#!/usr/bin/env python3

from math import ceil
import time

import numpy as np
import dynet as dy

import dynn
from dynn.layers.dense_layers import Affine
from dynn.layers.embedding_layers import Embeddings
from dynn.layers.recurrent_layers import StackedLSTM
from dynn.layers.transduction_layers import (
    Transduction, Unidirectional
)

from dynn.parameter_initialization import UniformInit

from dynn.data import ptb
from dynn.data.dictionary import Dictionary
from dynn.data.batching import BPTTBatchIterator

# For reproducibility
dynn.set_random_seed(31415)

# Data
# ====

# Download SST
ptb.download_ptb("data")

# Load the data
print("Loading the PTB data")
data = ptb.load_ptb("data", eos="<eos>")

# Learn the dictionary
print("Building the dictionary")
dic = Dictionary.from_data(data["train"])
dic.freeze()
dic.save("ptb.dic")

# Numberize the data
print("Numberizing")
data = dic.numberize(data)

# Model
# =====

# Hyper-parameters
LEARNING_RATE = 20
LEARNING_RATE_DECAY = 4.0
CLIP_NORM = 0.25
BPTT_LENGTH = 35
BATCH_SIZE = 20
N_LAYERS = 2
EMBED_DIM = 200
HIDDEN_DIM = 200
VOC_SIZE = len(dic)
DROPOUT = 0.2
N_EPOCHS = 40

# Define the network as a custom layer


class RNNLM(object):

    def __init__(self, nl, dx, dh):
        # Master parameter collection
        self.pc = dy.ParameterCollection()
        # Word embeddings
        embed_init = UniformInit(0.1)
        E = self.pc.add_parameters((len(dic), dx), init=embed_init)
        # Embedding layer
        embed = Embeddings(self.pc, dic, dx, params=E)
        self.embed = Transduction(embed)
        # RNNLM
        self.lstm_cell = StackedLSTM(self.pc, nl, dx, dh, DROPOUT, DROPOUT)
        self.rnn = Unidirectional(self.lstm_cell)
        # Final projection layer
        proj_layer = Affine(self.pc, dh, len(dic), dropout=DROPOUT, W_p=E)
        self.project = Transduction(proj_layer)

    def init(self, test=False, update=True):
        self.embed.init(test=test, update=update)
        self.rnn.init(test=test, update=update)
        self.project.init(test=test, update=update)

    def __call__(self, input_sequence, initial_state=None):
        # Embed input words
        w_embeds = self.embed(input_sequence)
        # Initialize hidden state
        if initial_state is not None:
            batched = w_embeds[0].dim()[1] > 1
            initial_state = [dy.inputTensor(s, batched=batched)
                             for s in initial_state]
        # Run the bilstm
        states = self.rnn(w_embeds, initial_state=initial_state)
        # Retrieve the network outputs
        hs = [self.rnn.cell.get_output(state) for state in states]
        # Get the logits
        logits = self.project(hs)
        # Retrieve last state value
        last_state = [s.npvalue() for s in states[-1]]
        # Return logits and last state
        return logits, last_state


# Instantiate the network
network = RNNLM(N_LAYERS, EMBED_DIM, HIDDEN_DIM)

# Optimizer
trainer = dy.SimpleSGDTrainer(network.pc, learning_rate=LEARNING_RATE)
trainer.set_clip_threshold(CLIP_NORM)


# Training
# ========

# Create the batch iterators
print("Creating batch iterators")
train_batches = BPTTBatchIterator(
    data["train"], batch_size=BATCH_SIZE, seq_length=BPTT_LENGTH
)
valid_batches = BPTTBatchIterator(data["valid"], batch_size=1, seq_length=200)
test_batches = BPTTBatchIterator(data["test"], batch_size=1, seq_length=200)
print(f"{len(train_batches)} training batches")


# Start training
print("Starting training")
best_ppl = np.inf
# Start training
for epoch in range(N_EPOCHS):
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
        logits, state_val = network(x, initial_state=state_val)
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
                f"NLL={nll.value():.3f} ppl={np.exp(nll.value()):.2f}"
            )
    # End of epoch logging
    print(f"Epoch {epoch+1}@100%: "
          f"NLL={nll.value():.3f} ppl={np.exp(nll.value()):.2f}")
    print(f"Took {time.time()-start_time:.1f}s")
    print("=" * 20)
    # Validate
    nll = 0
    # This state will be passed around for truncated BPTT
    state_val = None
    for x, targets in valid_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Compute logits
        logits, state_val = network(x, initial_state=state_val)
        # Loss at each step
        nlls = [dy.pickneglogsoftmax_batch(logit, y)
                for logit, y in zip(logits, targets)]
        # Aggregate NLL
        nll += dy.sum_batches(dy.esum(nlls)).value()
    # Average NLL
    nll /= valid_batches.num_samples
    # Perplexity
    ppl = np.exp(nll)
    # Print final result
    print(f"Valid ppl: {ppl:.2f}")
    # Early stopping
    if ppl < best_ppl:
        best_ppl = ppl
        network.pc.save("ptb_rnnlm.model")
    else:
        print("Decreasing learning rate")
        trainer.learning_rate /= LEARNING_RATE_DECAY
        print(f"New learning rate: {trainer.learning_rate}")

# Testing
# =======

# Load model
print("Reloading best model")
network.pc.populate("ptb_rnnlm.model")

# Test
nll = 0
# This state will be passed around for truncated BPTT
state_val = None
for x, targets in test_batches:
    # Renew the computation graph
    dy.renew_cg()
    # Initialize layers
    network.init(test=True, update=False)
    # Compute logits
    logits, state_val = network(x, initial_state=state_val)
    # Loss at each step
    nlls = [dy.pickneglogsoftmax_batch(logit, y)
            for logit, y in zip(logits, targets)]
    # Aggregate NLL
    nll += dy.sum_batches(dy.esum(nlls)).value()
# Average NLL
nll /= test_batches.num_samples
# Perplexity
ppl = np.exp(nll)
# Print final result
print(f"Test ppl: {ppl:.2f}")
