#!/usr/bin/env python3

from math import ceil
import time

import dynet as dy

import dynn
from dynn.layers.dense_layers import DenseLayer
from dynn.layers.embedding_layers import EmbeddingLayer
from dynn.layers.pooling_layers import MaxPooling1DLayer
from dynn.layers.recurrent_layers import LSTM
from dynn.layers.transduction_layers import (
    FeedForwardTransductionLayer, SequenceMaskingLayer, BidirectionalLayer
)
from dynn.layers.combination_layers import StackedLayers
from dynn.activations import identity
from dynn.operations import stack

from dynn.data import sst
from dynn.data import preprocess
from dynn.data.dictionary import Dictionary
from dynn.data.batching import PaddedSequenceBatchIterator

# For reproducibility
dynn.set_random_seed(31415)

# Data
# ====

# Download SST
sst.download_sst(".")

# Load the data
print("Loading the SST data")
(
    (train_x, train_y),
    (dev_x, dev_y),
    (test_x, test_y),
) = sst.load_sst(".", terminals_only=True, binary=True)

# Lowercase
print("Lowercasing")
train_x, dev_x, test_x = preprocess.lowercase([train_x, dev_x, test_x])

# Learn the dictionary
print("Building the dictionary")
dic = Dictionary.from_data(train_x)
dic.freeze()

# Numberize the data
print("Numberizing")
train_x = dic.numberize(train_x)
dev_x = dic.numberize(dev_x)
test_x = dic.numberize(test_x)


# Create the batch iterators
print("Creating batch iterators")
train_batches = PaddedSequenceBatchIterator(
    train_x, train_y, dic, max_samples=32
)
dev_batches = PaddedSequenceBatchIterator(
    dev_x, dev_y, dic, max_samples=1, shuffle=False
)
test_batches = PaddedSequenceBatchIterator(
    test_x, test_y, dic, max_samples=1, shuffle=False
)

# Model
# =====

# Hyper-parameters
EMBED_DIM = 100
HIDDEN_DIM = 512
MAX_WIDTH = 4
N_CLASSES = 2

# Define the network as a custom layer


class BiLSTM(object):

    def __init__(self, embed_dim, hidden_dim, num_classes):
        # Master parameter collection
        self.pc = dy.ParameterCollection()
        # Word embeddings
        self.embed = FeedForwardTransductionLayer(
            EmbeddingLayer(self.pc, dic, EMBED_DIM, pad_mask=0.0)
        )
        # BiLSTM
        self.bilstm = BidirectionalLayer(
            forward_cell=LSTM(self.pc, EMBED_DIM,
                              HIDDEN_DIM//2, dropout_h=0.1),
            backward_cell=LSTM(self.pc, EMBED_DIM,
                               HIDDEN_DIM//2, dropout_h=0.1),
        )
        # Masking for the pooling layer
        self.masking = SequenceMaskingLayer(mask_value=-9999)
        # Pool and predict
        self.pool_and_predict = StackedLayers(
            # Max pooling
            MaxPooling1DLayer(),
            # Softmax layer
            DenseLayer(self.pc, HIDDEN_DIM, N_CLASSES,
                       activation=identity, dropout=0.5),
        )

    def init(self, test=False, update=True):
        self.embed.init(test=test, update=update)
        self.bilstm.init(test=test, update=update)
        self.masking.init(test=test, update=update)
        self.pool_and_predict.init(test=test, update=update)

    def __call__(self, batch):
        # Embed the f out of the inputs
        w_embeds = self.embed(batch.sequences)
        # Run the bilstm
        fwd_states, bwd_states = self.bilstm(w_embeds, lengths=batch.lengths)
        H = [dy.concatenate([fwd_h, bwd_h])
             for (fwd_h, bwd_h), _ in zip(fwd_states, bwd_states)]
        # Mask and stack to a matrix
        masked_H = stack(self.masking(H, lengths=batch.lengths), d=0)
        # Maxpool and get the logits
        logits = self.pool_and_predict(masked_H)
        return logits


# Instantiate the network
network = BiLSTM(EMBED_DIM, HIDDEN_DIM, N_CLASSES)

# Optimizer
trainer = dy.AdamTrainer(network.pc, alpha=0.001)


# Training
# ========

print("Starting training")
# Start training
for epoch in range(10):
    # Time the epoch
    start_time = time.time()
    for batch, y in train_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=False, update=True)
        # Compute logits
        logits = network(batch)
        # Loss function
        nll = dy.mean_batches(dy.pickneglogsoftmax_batch(logits, y))
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
    accuracy = 0
    for batch, y in dev_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Compute logits
        logits = network(batch)
        # Get prediction
        predicted = logits.npvalue().argmax(axis=0)
        # Accuracy
        accuracy += (predicted == y).sum()
    # Average accuracy
    accuracy /= dev_batches.num_samples
    # Print final result
    print(f"Dev accuracy: {accuracy*100:.2f}%")

# Testing
# =======

# Test
accuracy = 0
for batch, y in test_batches:
    # Renew the computation graph
    dy.renew_cg()
    # Initialize layers
    network.init(test=True, update=False)
    # Compute logits
    logits = network(batch)
    # Get prediction
    predicted = logits.npvalue().argmax(axis=0)
    # Accuracy
    accuracy += (predicted == y).sum()
# Average accuracy
accuracy /= test_batches.num_samples
# Print final result
print(f"Test accuracy: {accuracy*100:.2f}%")
