#!/usr/bin/env python3

from math import ceil
import time

import dynet as dy

import dynn
from dynn.layers.dense_layers import Affine
from dynn.layers.embedding_layers import Embeddings
from dynn.layers.pooling_layers import MeanPooling1DLayer
from dynn.layers.recurrent_layers import LSTM
from dynn.layers.transduction_layers import (
    Transduction, Bidirectional
)

from dynn.data import sst
from dynn.data import preprocess
from dynn.data.dictionary import Dictionary
from dynn.data.batching import PaddedSequenceBatchIterator

# For reproducibility
dynn.set_random_seed(31415)

# Data
# ====

# Download SST
sst.download_sst("data")

# Load the data
print("Loading the SST data")
(
    (train_x, train_y),
    (dev_x, dev_y),
    (test_x, test_y),
) = sst.load_sst("data", terminals_only=True, binary=True)

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
    train_x, train_y, dic, max_samples=32, group_by_length=True
)
dev_batches = PaddedSequenceBatchIterator(
    dev_x, dev_y, dic, max_samples=32, shuffle=False
)
test_batches = PaddedSequenceBatchIterator(
    test_x, test_y, dic, max_samples=32, shuffle=False
)

# Model
# =====

# Hyper-parameters
EMBED_DIM = 100
HIDDEN_DIM = 512
N_CLASSES = 2
DROPOUT = 0.5

# Define the network as a custom layer


class BiLSTM(object):

    def __init__(self, dx, dh):
        # Master parameter collection
        self.pc = dy.ParameterCollection()
        # Word embeddings
        self.embed = Transduction(
            Embeddings(self.pc, dic, dx, pad_mask=0.0)
        )
        # BiLSTM
        self.bilstm = Bidirectional(
            forward_cell=LSTM(self.pc, dx, dh, DROPOUT, DROPOUT),
            backward_cell=LSTM(self.pc, dx, dh, DROPOUT, DROPOUT),
            output_only=True,
        )
        # Pooling layer
        self.mean_pool = MeanPooling1DLayer()
        # Softmax layer
        self.softmax = Affine(self.pc, dh, N_CLASSES, dropout=DROPOUT)

    def init(self, test=False, update=True):
        self.embed.init(test=test, update=update)
        self.bilstm.init(test=test, update=update)
        self.mean_pool.init(test=test, update=update)
        self.softmax.init(test=test, update=update)

    def __call__(self, batch):
        # Embed the f out of the inputs
        w_embeds = self.embed(batch.sequences)
        # Run the bilstm
        fwd_H, bwd_H = self.bilstm(w_embeds, lengths=batch.lengths)
        H = [0.5 * (fh + bh) for fh, bh in zip(fwd_H, bwd_H)]
        # Mask and stack to a matrix
        pooled_H = self.mean_pool(H, lengths=batch.lengths)
        # Maxpool and get the logits
        logits = self.softmax(pooled_H)
        return logits


# Instantiate the network
network = BiLSTM(EMBED_DIM, HIDDEN_DIM)

# Optimizer
trainer = dy.AdamTrainer(network.pc, alpha=0.001)


# Training
# ========

# Start training
print("Starting training")
best_accuracy = 0
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
    # Early stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        network.pc.save("sst_bilstm.model")
    else:
        print(f"Early stopping with best accuracy {best_accuracy*100:.2f}%")
        break

# Testing
# =======

# Load model
print("Reloading best model")
network.pc.populate("sst_bilstm.model")

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
