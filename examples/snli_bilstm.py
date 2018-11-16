#!/usr/bin/env python3

from math import ceil
import time
import os
import pickle

import dynet as dy

import dynn
from dynn.layers import Affine
from dynn.layers import Sequential
from dynn.layers import Embeddings
from dynn.layers import MeanPool1D
from dynn.layers import LSTM
from dynn.layers import Transduction, Bidirectional

from dynn.data import snli
from dynn.data import preprocess
from dynn.data import Dictionary
from dynn.data.batching import SequencePairsBatches

from dynn.activations import relu

# For reproducibility
dynn.set_random_seed(31415)

# Data
# ====

# Cache the data (because loading and preprocessing is slow)
if not os.path.isfile("data/snli.bin"):
    # Download SNLI
    snli.download_snli("data")

    # Load the data
    print("Loading the SNLI data")
    data = snli.load_snli("data", terminals_only=True, binary=True)

    # Lowercase
    print("Lowercasing")
    data = preprocess.lowercase(data)

    train_p, train_h, train_y = data["train"]
    dev_p, dev_h, dev_y = data["dev"]
    test_p, test_h, test_y = data["test"]

    # Learn the dictionary
    print("Building the dictionary")
    dic = Dictionary.from_data(train_p + train_h)
    dic.freeze()
    dic.save("snli.dic")
    # Labels dictionary
    label_dic = Dictionary.from_data([train_y], no_specials=True)
    label_dic.save("snli.labels.dic")

    # Numberize the data
    print("Numberizing")
    train_p, train_h = dic.numberize([train_p, train_h])
    dev_p, dev_h = dic.numberize([dev_p, dev_h])
    test_p, test_h = dic.numberize([test_p, test_h])
    # Numberize labels
    train_y, dev_y, test_y = label_dic.numberize([train_y, dev_y, test_y])
    # Save to file
    with open("data/snli.bin", "wb") as f:
        pickle.dump([train_p, train_h, train_y,
                     dev_p, dev_h, dev_y,
                     test_p, test_h, test_y, dic, label_dic], f)
else:
    print("Loading cached dataset")
    with open("data/snli.bin", "rb") as f:
        [train_p, train_h, train_y,
         dev_p, dev_h, dev_y,
         test_p, test_h, test_y, dic, label_dic] = pickle.load(f)


# Create the batch iterators
print("Creating batch iterators")
train_batches = SequencePairsBatches(
    train_p,
    train_h,
    dic,
    labels=train_y,
    max_samples=32,
    group_by_length=None,
)
dev_batches = SequencePairsBatches(
    dev_p, dev_h, dic, labels=dev_y, max_samples=32, shuffle=False
)
test_batches = SequencePairsBatches(
    test_p, test_h, dic, labels=test_y, max_samples=32, shuffle=False
)

# Model
# =====

# Hyper-parameters
EMBED_DIM = 100
HIDDEN_DIM = 256
N_CLASSES = len(label_dic)
DROPOUT = 0.5
N_EPOCHS = 10

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
        self.encoder = Bidirectional(
            forward_cell=LSTM(self.pc, dx, dh, DROPOUT, DROPOUT),
            backward_cell=LSTM(self.pc, dx, dh, DROPOUT, DROPOUT),
            output_only=True,
        )
        # Pooling layer
        self.pool = MeanPool1D()
        # Softmax layer
        self.softmax = Sequential(
            Affine(self.pc, 4*dh, dh, activation=relu),
            Affine(self.pc, dh, N_CLASSES, dropout=DROPOUT),
        )

    def init(self, test=False, update=True):
        self.embed.init(test=test, update=update)
        self.encoder.init(test=test, update=update)
        self.pool.init(test=test, update=update)
        self.softmax.init(test=test, update=update)

    def encode(self, batch):
        # Embed the f out of the inputs
        w_embeds = self.embed(batch.sequences)
        # Run the bilstm
        fwd_H, bwd_H = self.encoder(w_embeds, lengths=batch.lengths)
        H = [0.5 * (fh + bh) for fh, bh in zip(fwd_H, bwd_H)]
        # Mask and stack to a matrix
        pooled_H = self.pool(H, lengths=batch.lengths)
        return pooled_H

    def __call__(self, premise, hypothesis):
        # Get encodings for premise and hypothesis
        h_premise = self.encode(premise)
        h_hypothesis = self.encode(hypothesis)
        # Interaction vector
        h_interact = dy.concatenate([
            h_premise - h_hypothesis,
            dy.cmult(h_premise, h_hypothesis),
            h_premise,
            h_hypothesis,
        ])
        # Logits
        logits = self.softmax(h_interact)
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
for epoch in range(N_EPOCHS):
    # Time the epoch
    start_time = time.time()
    for premise, hypothesis, y in train_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=False, update=True)
        # Compute logits
        logits = network(premise, hypothesis)
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
    for premise, hypothesis, y in dev_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Compute logits
        logits = network(premise, hypothesis)
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
        dynn.io.save(network.pc, "snli_bilstm.model.npz")
    else:
        print(f"Early stopping with best accuracy {best_accuracy*100:.2f}%")
        break

# Testing
# =======

# Load model
print("Reloading best model")
dynn.io.populate(network.pc, "snli_bilstm.model.npz")

# Test
accuracy = 0
for premise, hypothesis, y in test_batches:
    # Renew the computation graph
    dy.renew_cg()
    # Initialize layers
    network.init(test=True, update=False)
    # Compute logits
    logits = network(premise, hypothesis)
    # Get prediction
    predicted = logits.npvalue().argmax(axis=0)
    # Accuracy
    accuracy += (predicted == y).sum()
# Average accuracy
accuracy /= test_batches.num_samples
# Print final result
print(f"Test accuracy: {accuracy*100:.2f}%")
