#!/usr/bin/env python3

from math import ceil
import time

import numpy as np
import dynet as dy

import dynn
from dynn.layers.dense_layers import DenseLayer
from dynn.layers.embedding_layers import EmbeddingLayer
from dynn.layers.recurrent_layers import LSTM
from dynn.layers.transduction_layers import (
    FeedForwardTransductionLayer, UnidirectionalLayer
)
from dynn.activations import identity
# from dynn.operations import stack

from dynn.data import ptb
from dynn.data.dictionary import Dictionary
from dynn.data.batching import BPTTBatchIterator

# For reproducibility
dynn.set_random_seed(31415)

# Data
# ====

# Download SST
ptb.download_ptb(".")

# Load the data
print("Loading the PTB data")
train, valid, test = ptb.load_ptb(".")

# Learn the dictionary
print("Building the dictionary")
dic = Dictionary.from_data(train)
dic.freeze()

# Numberize the data
print("Numberizing")
train = dic.numberize(train)
valid = dic.numberize(valid)
test = dic.numberize(test)

# Create the batch iterators
print("Creating batch iterators")
train_batches = BPTTBatchIterator(
    train, batch_size=32, seq_length=30
)
valid_batches = BPTTBatchIterator(
    valid, batch_size=1, seq_length=30
)
test_batches = BPTTBatchIterator(
    test, batch_size=1, seq_length=30
)

# Model
# =====

# Hyper-parameters
EMBED_DIM = 32
HIDDEN_DIM = 32
VOC_SIZE = len(dic)

# Define the network as a custom layer


class RNNLM(object):

    def __init__(self, embed_dim, hidden_dim, voc_size):
        # Master parameter collection
        self.pc = dy.ParameterCollection()
        # Word embeddings
        self.embed = FeedForwardTransductionLayer(
            EmbeddingLayer(self.pc, dic, embed_dim, pad_mask=0.0)
        )
        # RNNLM
        self.rnn = UnidirectionalLayer(
            LSTM(self.pc, embed_dim, hidden_dim,
                 dropout_h=0.5, dropout_x=0.5)
        )
        # Final projection layer
        self.project = FeedForwardTransductionLayer(
            DenseLayer(
                self.pc, hidden_dim, voc_size, activation=identity, dropout=0.5
            )
        )

    def init(self, test=False, update=True):
        self.embed.init(test=test, update=update)
        self.rnn.init(test=test, update=update)
        self.project.init(test=test, update=update)

    def __call__(self, input_sequence, initial_state=None):
        # Embed input words
        w_embeds = self.embed(input_sequence)
        # Initialize hidden state
        if initial_state is not None:
            initial_state = dy.inputTensor(initial_state, batched=True)
        # Run the bilstm
        states = self.rnn(w_embeds, initial_state=initial_state)
        # Retrieve the network outputs
        hs = [self.rnn.cell.get_output(state) for state in states]
        # Get the logits
        logits = self.project(hs)
        # Retrieve last state value
        last_state = [h.npvalue() for h in states[-1]]
        # Return logits and last state
        return logits, last_state


# Instantiate the network
network = RNNLM(EMBED_DIM, HIDDEN_DIM, VOC_SIZE)

# Optimizer
trainer = dy.AdamTrainer(network.pc, alpha=0.001)


# Training
# ========

# Start training
print("Starting training")
best_ppl = np.inf
# Start training
for epoch in range(10):
    # Time the epoch
    start_time = time.time()
    # This state will be passed around for truncated BPTT
    state_val = None
    for x, targets in train_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=False, update=True)
        # Compute logits
        logits, state_val = network(x, initial_state=None)
        # Loss at each step
        nlls = [dy.pickneglogsoftmax_batch(logit, y)
                for logit, y in zip(logits, targets)]
        # Reduce losses
        nll = dy.mean_batches(dy.average(nlls))
        # Backward pass
        nll.backward()
        # Update the parameters
        trainer.update()
        # Print the current loss from time to time
        if train_batches.just_passed_multiple(ceil(len(train_batches)/10)):
            print(
                f"Epoch {epoch+1}@{train_batches.percentage_done():.0f}%: "
                f"NLL={nll.value():.3f}"
            )
    # End of epoch logging
    print(f"Epoch {epoch+1}@100%: NLL={nll.value():.3f}")
    print(f"Took {time.time()-start_time:.1f}s")
    print("=" * 20)
    # Validate
    nll = 0
    # This state will be passed around for truncated BPTT
    state_val = None
    for x, y in valid_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Compute logits
        logits, state_val = network(x, initial_state=None)
        # Loss at each step
        nlls = [dy.pickneglogsoftmax_batch(logit, y)
                for logit, y in zip(logits, targets)]
        # Accuracy
        nll += dy.sum_batches(dy.esum(nlls)).value()
    # Average NLL
    nll /= valid_batches.num_samples
    # Perplexity
    ppl = np.exp(-nll)
    # Print final result
    print(f"Valid ppl: {ppl:.2f}%")
    # Early stopping
    if ppl < best_ppl:
        best_ppl = ppl
        network.pc.save("ptb_rnnlm.model")
    else:
        print(f"Early stopping with best ppl {best_ppl:.2f}%")
        break

# Testing
# =======

# Load model
print("Reloading best model")
network.pc.populate("sst_bilstm.model")

# Test
nll = 0
# This state will be passed around for truncated BPTT
state_val = None
for x, y in test_batches:
    # Renew the computation graph
    dy.renew_cg()
    # Initialize layers
    network.init(test=True, update=False)
    # Compute logits
    logits, state_val = network(x, initial_state=None)
    # Loss at each step
    nlls = [dy.pickneglogsoftmax_batch(logit, y)
            for logit, y in zip(logits, targets)]
    # Accuracy
    nll += dy.sum_batches(dy.esum(nlls)).value()
# Average NLL
nll /= test_batches.num_samples
# Perplexity
ppl = np.exp(-nll)
# Print final result
print(f"Test ppl: {ppl:.2f}%")
