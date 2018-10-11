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
    Transduction, Bidirectional
)
from dynn.layers.attention_layers import MLPAttentionLayer
from dynn.layers.combination_layers import Sequential

from dynn.operations import stack
from dynn.parameter_initialization import UniformInit

from dynn.data import iwslt
from dynn.data import preprocess
from dynn.data.dictionary import Dictionary
from dynn.data.batching import SequencePairsBatchIterator

# For reproducibility
dynn.set_random_seed(31415)


# Hyper-parameters
# ================

VOC_SIZE = 30000
LEARNING_RATE = 20
LEARNING_RATE_DECAY = 4.0
CLIP_NORM = 0.25
BATCH_SIZE = 20
N_LAYERS = 1
EMBED_DIM = 200
HIDDEN_DIM = 200
DROPOUT = 0.2
LABEL_SMOOTHING = 1
N_EPOCHS = 1


# Data
# ====

# Download SST
iwslt.download_iwslt("data", year="2016", langpair="fr-en")

# Load the data
print("Loading the IWSLT data")
(
    (train_src, train_tgt),
    (dev_src, dev_tgt),
    (test_src, test_tgt),
) = iwslt.load_iwslt("data", year="2016", langpair="fr-en", eos="<eos>")
print(f"{len(train_src)} training samples")
print(f"{len(dev_src)} dev samples")
print(f"{len(test_src)} test samples")

# Prepare French data
print("Preparing French data...")
print("Lowercasing")
train_src = preprocess.lowercase(train_src)
dev_src = preprocess.lowercase(dev_src)
test_src = preprocess.lowercase(test_src)

# Learn the dictionaries
print("Building the dictionary")
dic_src = Dictionary.from_data(train_src, max_size=VOC_SIZE)
dic_src.freeze()

# Numberize the data
print("Numberizing")
train_src = dic_src.numberize(train_src)
dev_src = dic_src.numberize(dev_src)
test_src = dic_src.numberize(test_src)

# Prepare English data
print("Preparing English data...")
print("Lowercasing")
train_tgt = preprocess.lowercase(train_tgt)
dev_tgt = preprocess.lowercase(dev_tgt)
test_tgt = preprocess.lowercase(test_tgt)

# Learn the dictionaries
print("Building the dictionary")
dic_tgt = Dictionary.from_data(train_tgt, max_size=VOC_SIZE)
dic_tgt.freeze()

# Numberize the data
print("Numberizing")
train_tgt = dic_tgt.numberize(train_tgt)
dev_tgt = dic_tgt.numberize(dev_tgt)
test_tgt = dic_tgt.numberize(test_tgt)


# Model
# =====

# Define the network as a custom layer


class AttBiLSTM(object):

    def __init__(self, nl, dx, dh):
        # Master parameter collection
        self.pc = dy.ParameterCollection()
        # Encoder
        # -------
        # Source Word embeddings
        embed_init = UniformInit(0.1)
        E_src = self.pc.add_parameters((len(dic_src), dx), init=embed_init)
        self.src_embed = Embeddings(self.pc, dic_src, dx, params=E_src)
        self.src_embed_all = Transduction(self.src_embed)
        # BiLSTM
        self.enc_fwd = StackedLSTM(self.pc, nl, dx, dh, DROPOUT, DROPOUT)
        self.enc_bwd = StackedLSTM(self.pc, nl, dx, dh, DROPOUT, DROPOUT)
        self.bilstm = Bidirectional(self.enc_fwd, self.enc_bwd)
        # Attention
        # --------
        self.attend = MLPAttentionLayer(self.pc, dh+dx, dh, dh)
        # Decoder
        # -------
        # Target word embeddings
        embed_init = UniformInit(0.1)
        E_tgt = self.pc.add_parameters((len(dic_tgt), dx), init=embed_init)
        self.tgt_embed = Embeddings(self.pc, dic_tgt, dx, params=E_tgt)
        self.tgt_embed_all = Transduction(self.tgt_embed)
        # Start of sentence embedding
        self.sos = self.pc.add_lookup_parameters((1, dx), init=embed_init)
        # Recurrent decoder
        self.dec_cell = StackedLSTM(self.pc, nl, dx+dh, dh, DROPOUT, DROPOUT)
        # Final projection layers
        self.project = Sequential(
            # First project to embedding dim
            Affine(self.pc, dh, dx),
            # Then logit layer with weights tied to the word embeddings
            Affine(self.pc, dh, len(dic_tgt), dropout=DROPOUT, W_p=E_tgt)
        )
        self.project_all = Transduction(self.project)

    def init(self, test=False, update=True):
        self.src_embed_all.init(test=test, update=update)
        self.bilstm.init(test=test, update=update)
        self.attend.init(test=test, update=update)
        self.tgt_embed_all.init(test=test, update=update)
        self.dec_cell.init(test=test, update=update)
        self.project_all.init(test=test, update=update)

    def encode(self, src):
        # Embed input words
        src_embs = self.src_embed_all(src.sequences)
        # Encode
        fwd, bwd = self.bilstm(src_embs, lengths=src.lengths, output_only=True)
        # Sum forward and backward and concatenate all to a dh x L expression
        return stack(fwd, d=-1) + stack(bwd, d=-1)

    def __call__(self, src, tgt):
        # Encode
        # ------
        X = self.encode(src)
        # Decode
        # ------
        # Mask for attention
        attn_mask = src.get_mask(base_val=0, mask_val=-np.inf)
        # Embed all words (except EOS)
        tgt_embs = [self.sos.batch([0] * tgt.batch_size)]
        tgt_embs.extend(self.tgt_embed_all([w for w in tgt.sequences[:-1]]))
        # Initialize decoder state
        dec_state = self.dec_cell.initial_value(tgt.batch_size)
        # Iterate over target words
        dec_outputs = []
        for x in tgt_embs:
            # Attention query: previous hidden state and current word embedding
            query = dy.concatenate([x, self.dec_cell.get_output(dec_state)])
            # Attend
            ctx, _ = self.attend(query, X, X, mask=attn_mask)
            # Both context and target word embedding will be fed to the decoder
            dec_input = dy.concatenate([x, ctx])
            # Update decoder state
            dec_state = self.dec_cell(dec_input, *dec_state)
            # Save output
            dec_outputs.append(self.dec_cell.get_output(dec_state))
        # Compute logits
        logits = self.project_all(dec_outputs)

        return logits

    def sample(self, src, tau=1.0):
        batch_size = src.batch_size
        # Encode
        # ------
        X = self.encode(src)
        # Decode
        # ------
        # Mask for attention
        attn_mask = src.get_mask(base_val=0, mask_val=-np.inf)
        # Max length
        max_len = 2 * src.max_length
        # Generated words
        sents = [[] for _ in range(batch_size)]
        x = self.sos.batch([0] * batch_size)
        # Initialize decoder state
        dec_state = self.dec_cell.initial_value(1)
        # Start decoding
        is_over = [False for _ in range(batch_size)]
        while not all(is_over) and max(len(sent) for sent in sents) < max_len:
            # Attention query: previous hidden state and current word embedding
            query = dy.concatenate([x, self.dec_cell.get_output(dec_state)])
            # Attend
            ctx, _ = self.attend(query, X, X, mask=attn_mask)
            # Both context and target word embedding will be fed to the decoder
            dec_input = dy.concatenate([x, ctx])
            # Update decoder state
            dec_state = self.dec_cell(dec_input, *dec_state)
            # Save output
            h = self.dec_cell.get_output(dec_state)
            # Get log_probs
            logits = self.project(h)
            log_p = dy.log_softmax(logits)
            # Add gumbel noise for sampling
            noise = dy.random_gumbel(len(dic_tgt), batch_size=batch_size)
            log_p += tau * noise
            # Sample
            next_word = log_p.npvalue().reshape(-1, batch_size).argmax(axis=0)
            # Check for EOS in each batch element and add the word to the
            # output sentences accordingly
            for b, w in enumerate(next_word):
                if is_over[b]:
                    continue
                elif w == dic_tgt.eos_idx:
                    is_over[b] = True
                else:
                    sents[b].append(w)
            # Embed the last word
            x = self.tgt_embed([sent[-1] for sent in sents])
        # Return the sentences
        return sents


# Instantiate the network
network = AttBiLSTM(N_LAYERS, EMBED_DIM, HIDDEN_DIM)
# network.pc.save("iwslt_att.model")

# Optimizer
trainer = dy.SimpleSGDTrainer(network.pc, learning_rate=LEARNING_RATE)
trainer.set_clip_threshold(CLIP_NORM)


# Training
# ========

# Create the batch iterators
print("Creating batch iterators")
train_batches = SequencePairsBatchIterator(
    train_src, train_tgt, dic_src, dic_tgt, max_samples=100, max_tokens=4000,
)
dev_batches = SequencePairsBatchIterator(
    dev_src, dev_tgt, dic_src, dic_tgt, max_samples=10)
test_batches = SequencePairsBatchIterator(
    test_src, test_tgt, dic_src, dic_tgt, max_samples=10)
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
    for src, tgt in train_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=False, update=True)
        # Compute logits
        logits = network(src, tgt)
        # log prob at each timestep
        logprobs = [dy.log_softmax(logit) for logit in logits]
        # Label smoothed log likelihoods
        lls = [dy.pick_batch(lp, y) * (1-LABEL_SMOOTHING) +
               dy.mean_elems(lp) * LABEL_SMOOTHING
               for lp, y in zip(logprobs, tgt.sequences)]
        # Mask losses and reduce
        masked_nll = - stack(lls, d=-1) * tgt.get_mask()
        # Reduce losses
        nll = dy.mean_batches(masked_nll)
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
    for src, tgt in dev_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Compute logits
        logits = network(src, tgt)
        # log prob at each timestep
        logprobs = [dy.log_softmax(logit) for logit in logits]
        # Label smoothed log likelihoods
        lls = [dy.pick_batch(lp, y) * (1-LABEL_SMOOTHING) +
               dy.mean_elems(lp) * LABEL_SMOOTHING
               for lp, y in zip(logprobs, tgt.sequences)]
        # Mask losses and reduce
        masked_nll = - stack(lls, d=-1) * tgt.get_mask()
        # Aggregate NLL
        nll += dy.sum_batches(masked_nll).value()
    # Average NLL
    nll /= dev_batches.num_samples
    # Perplexity
    ppl = np.exp(nll)
    # Print final result
    print(f"Valid ppl: {ppl:.2f}")
    # Early stopping
    if ppl < best_ppl:
        best_ppl = ppl
        network.pc.save("iwslt_att.model")
    else:
        print("Decreasing learning rate")
        trainer.learning_rate /= LEARNING_RATE_DECAY
        print(f"New learning rate: {trainer.learning_rate}")

# Testing
# =======

# Load model
print("Reloading best model")
# network.pc.populate("iwslt_att.model")

# Generate from dev data
for src, tgt in dev_batches:
    # Renew the computation graph
    dy.renew_cg()
    # Initialize layers
    network.init(test=True, update=False)
    # Compute logits
    hyp = network.sample(src)
    # Print
    for b in range(src.batch_size):
        print("-"*80)
        src_sent = dic_src.string(src.sequences[:, b], join_with=" ")
        print(f"src:\t{src_sent}")
        hyp_sent = dic_src.string(hyp[b], join_with=" ")
        print(f"hyp:\t{hyp_sent}")
        tgt_sent = dic_tgt.string(tgt.sequences[:, b], join_with=" ")
        print(f"tgt:\t{tgt_sent}")
