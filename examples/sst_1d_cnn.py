#!/usr/bin/env python3

from math import ceil
import time

import numpy as np
import dynet as dy

from dynn.layers.dense_layers import DenseLayer
from dynn.layers.embedding_layers import EmbeddingLayer
from dynn.layers.convolution_layers import Conv1DLayer
from dynn.layers.pooling_layers import MaxPooling1DLayer
from dynn.layers.combination_layers import StackedLayers, ConcatenatedLayers
from dynn.activations import relu, identity

from dynn.data import sst
from dynn.data.dictionary import Dictionary
from dynn.data.batching import PaddedSequenceBatchIterator

# For reproducibility
dy.reset_random_seed(31415)
np.random.seed(51413)

# Data
# ====

# Download SST
sst.download_sst(".")

# Load the data
(
    (train_x, train_y),
    (dev_x, dev_y),
    (test_x, test_y),
) = sst.load_sst(".", terminals_only=True, binary=True)

# Learn the dictionary
dic = Dictionary.from_data(train_x)
dic.freeze()

# Numberize the data
train_x = dic.numberize(train_x)
dev_x = dic.numberize(dev_x)
test_x = dic.numberize(test_x)

# Create the batch iterators
train_batches = PaddedSequenceBatchIterator(
    train_x, train_y, dic, max_samples=64, max_tokens=2000
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
N_CLASSES = 2

# Master parameter collection
pc = dy.ParameterCollection()


# Embeddings Layer
embeddings = EmbeddingLayer(pc, dic, EMBED_DIM, pad_mask=0.0)
# Convolutions
conv1d = [
    Conv1DLayer(pc, EMBED_DIM, HIDDEN_DIM//4, 1, activation=relu),
    Conv1DLayer(pc, EMBED_DIM, HIDDEN_DIM//4, 2, activation=relu),
    Conv1DLayer(pc, EMBED_DIM, HIDDEN_DIM//4, 3, activation=relu),
    Conv1DLayer(pc, EMBED_DIM, HIDDEN_DIM//4, 4, activation=relu),
]
# Network
network = StackedLayers(
    embeddings,
    # Convolution layer
    ConcatenatedLayers(
        *conv1d,
        dim=-1,
    ),
    # Max pooling
    MaxPooling1DLayer(),
    # Softmax layer
    DenseLayer(pc, HIDDEN_DIM, N_CLASSES, activation=identity, dropout=0.5),
)

# Optimizer
trainer = dy.RMSPropTrainer(pc, learning_rate=0.001)


# Training
# ========

# Start training
for epoch in range(5):
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

# Testing
# =======

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

# Interpretation
# ==============

# Retrieve softmax weights
softmax_layer = network.layers[-1]
softmax_weights = softmax_layer.W_p.expr(update=False)

# Feature values
feature_weights = (softmax_weights[0] - softmax_weights[1]).npvalue()

# Word embeddings
embed_layer = network.layers[0]
embeddings = embed_layer.params


def top_k_ngrams(ngrams, k=10):
    """Helper function to select the top k scoring ngrams"""
    scores = np.zeros(len(ngrams))
    for i in range(len(ngrams)):
        scores[i] = sum(score for unigram, score in ngrams[i])
    argtopk = np.argsort(scores)[-k:]
    return [
        (" ".join(dic[unigram] for unigram, score in ngrams[idx]), scores[idx])
        for idx in argtopk
    ]


def top_ngrams_feature(feature_number, k=10, weight=1.0):
    """Helper function to retrieve the highest scoring ngrams for a given
    feature"""
    dy.renew_cg()
    # Retrieve the filter from the feature number
    kernel_size = feature_number // (HIDDEN_DIM // 4) + 1
    kernels = conv1d[kernel_size - 1].K_p.as_array()
    idx = feature_number % (HIDDEN_DIM // 4)
    kernel = dy.inputTensor(kernels[:, 0, :, idx])
    # Get the embedding matrix
    E = dy.transpose(embeddings.expr())
    # Get the k^filter_width top filters
    top_ngrams = [[]]
    for i in range(kernel_size):
        unigram_scores = (E * kernel[i]).npvalue()
        top_unigrams = unigram_scores.argsort()[-k:]
        new_top_ngrams = []
        for ngram in top_ngrams:
            for unigram in top_unigrams:
                new_ngram = ngram[:]
                new_ngram.append((unigram, unigram_scores[unigram] * weight))
                new_top_ngrams.append(new_ngram)
        top_ngrams = new_top_ngrams[:]
    # Only keep the top k
    topk = top_k_ngrams(top_ngrams, k=k)

    return topk


# Top 10 negative ngrams
negative_ngrams = []
most_negative_features = feature_weights.argsort()[:10]
for feature in most_negative_features:
    negative_ngrams.extend(
        top_ngrams_feature(feature, 10, -feature_weights[feature])
    )

topk_negative_ngrams = sorted(negative_ngrams, key=lambda x: x[1])[-10:]


# Top 10 positive ngrams
positive_ngrams = []
most_positive_features = feature_weights.argsort()[-10:]
for feature in most_positive_features:
    positive_ngrams.extend(
        top_ngrams_feature(feature, 10, feature_weights[feature])
    )

topk_positive_ngrams = sorted(positive_ngrams, key=lambda x: x[1])[-10:]

# Print the result
print("Top 10 negative n-grams:")
for negative_ngram, score in topk_negative_ngrams:
    print(f" - score={score} : \"{negative_ngram}\"")
print("")
print("Top 10 positive n-grams:")
for positive_ngram, score in topk_positive_ngrams:
    print(f" - score={score} : \"{positive_ngram}\"")
