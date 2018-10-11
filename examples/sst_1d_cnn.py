#!/usr/bin/env python3

from math import ceil
import time

import dynet as dy

import dynn
from dynn.layers.dense_layers import Affine
from dynn.layers.embedding_layers import Embeddings
from dynn.layers.convolution_layers import Conv1D
from dynn.layers.pooling_layers import MaxPool1D
from dynn.layers.combination_layers import Sequential, Parallel
from dynn.activations import relu

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
    train_x, train_y, dic, max_samples=64, group_by_length=True
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
EMBED_DIM = 300
FILTERS = {1: 128, 2: 256, 3: 256}
HIDDEN_DIM = sum(FILTERS.values())
N_CLASSES = 2

# Master parameter collection
pc = dy.ParameterCollection()


# Embeddings Layer
embeddings = Embeddings(pc, dic, EMBED_DIM, pad_mask=0.0)
# Convolutions
conv1d = [
    Conv1D(pc, EMBED_DIM, number, width, activation=relu)
    for width, number in FILTERS.items()
]
# Network
network = Sequential(
    embeddings,
    # Convolution layer
    Parallel(
        *conv1d,
        dim=-1,
    ),
    # Max pooling
    MaxPool1D(),
    # Softmax layer
    Affine(pc, HIDDEN_DIM, N_CLASSES, dropout=0.5),
)

# Optimizer
trainer = dy.MomentumSGDTrainer(pc, learning_rate=0.01, mom=0.9)


# Training
# ========

# Start training
print("Starting training")
best_accuracy = 0
for epoch in range(10):
    # Time the epoch
    start_time = time.time()
    for batch, y in train_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=False, update=True)
        # Run the cnn
        logits = network(batch.sequences)
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
    # Lower learning rate
    trainer.learning_rate *= 0.5
    # Validate
    accuracy = 0
    for batch, y in dev_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Run the cnn
        logits = network(batch.sequences)
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
        pc.save("sst_1d_cnn.model")
    else:
        print(f"Early stopping with best accuracy {best_accuracy*100:.2f}%")
        break

# Testing
# =======

# Load model
print("Reloading best model")
pc.populate("sst_1d_cnn.model")

# Test
accuracy = 0
for batch, y in test_batches:
    # Renew the computation graph
    dy.renew_cg()
    # Initialize layers
    network.init(test=True, update=False)
    # Run the cnn
    logits = network(batch.sequences)
    # Get prediction
    predicted = logits.npvalue().argmax(axis=0)
    # Accuracy
    accuracy += (predicted == y).sum()
# Average accuracy
accuracy /= test_batches.num_samples
# Print final result
print(f"Test accuracy: {accuracy*100:.2f}%")
