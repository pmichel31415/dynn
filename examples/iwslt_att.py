#!/usr/bin/env python3

from math import ceil
import time

import numpy as np
import dynet as dy
import sacrebleu

import dynn
from dynn.layers import Affine
from dynn.layers import Embeddings
from dynn.layers import StackedLSTM
from dynn.layers import Transduction, Bidirectional
from dynn.layers import MLPAttention
from dynn.layers import Sequential

from dynn.operations import stack
from dynn.parameter_initialization import UniformInit

from dynn.data import iwslt
from dynn.data import preprocess
from dynn.data import Dictionary
from dynn.data.batching import SequencePairsBatches

# For reproducibility
dynn.set_random_seed(31415)


# Hyper-parameters
# ================

VOC_SIZE = 30000
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 2.0
CLIP_NORM = 5.0
N_LAYERS = 1
EMBED_DIM = 256
HIDDEN_DIM = 512
DROPOUT = 0.2
LABEL_SMOOTHING = 0.1
N_EPOCHS = 10
BEAM_SIZE = 4
LENPEN = 1.0


# Data
# ====

# Download IWSLT
iwslt.download_iwslt("data", year="2016", langpair="fr-en")

# Load the data
print("Loading the IWSLT data")
train, dev, test = iwslt.load_iwslt("data", year="2016", langpair="fr-en")
print(f"{len(train[0])} training samples")
print(f"{len(dev[0])} dev samples")
print(f"{len(test[0])} test samples")

print("Lowercasing")
train, dev, test = preprocess.lowercase([train, dev, test])

# Learn the dictionaries
print("Building the dictionaries")
dic_src = Dictionary.from_data(train[0], max_size=VOC_SIZE)
dic_src.freeze()
dic_src.save("iwslt_att.dic.src")
dic_tgt = Dictionary.from_data(train[1], max_size=VOC_SIZE)
dic_tgt.freeze()
dic_tgt.save("iwslt_att.dic.tgt")

# Numberize the data
print("Numberizing")
train_src, dev_src, test_src = dic_src.numberize([train[0], dev[0], test[0]])
train_tgt, dev_tgt, test_tgt = dic_tgt.numberize([train[1], dev[1], test[1]])


# Model
# =====


class AttBiLSTM(object):
    """This custom layer implements an attention BiLSTM model"""

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
        self.attend = MLPAttention(self.pc, dh+dx, dh, dh)
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
            Affine(self.pc, dx, len(dic_tgt), dropout=DROPOUT, W_p=E_tgt)
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

    def decode(self, src, beam_size=3):
        """Find the best translation using beam search"""
        batch_size = src.batch_size
        # Defer batch size > 1 to multiple calls
        if batch_size > 1:
            sents, aligns = [], []
            for b in range(batch_size):
                sent, align = self.decode(src[b], beam_size)
                sents.append(sent[0])
                aligns.append(align[0])
            return sents, aligns
        # Encode
        # ------
        X = self.encode(src)
        # Decode
        # ------
        # Mask for attention
        attn_mask = src.get_mask(base_val=0, mask_val=-np.inf)
        # Max length
        max_len = 2 * src.max_length
        # Initialize beams
        first_beam = {
            "wemb": self.sos[0],  # Previous word embedding
            "state": self.dec_cell.initial_value(1),  # Decoder state
            "score": 0.0,  # score
            "words": [],  # generated words
            "align": [],  # Alignments given by attention
            "is_over": False,  # is over
        }
        beams = [first_beam]
        # Start decoding
        while not beams[-1]["is_over"] and len(beams[-1]["words"]) < max_len:
            new_beams = []
            for beam in beams:
                # Don't do anything if the beam is over
                if beam["is_over"]:
                    continue
                # Attention query: previous hidden state and current
                # word embedding
                prev_h = self.dec_cell.get_output(beam["state"])
                query = dy.concatenate([beam["wemb"], prev_h])
                # Attend
                ctx, attn_weights = self.attend(query, X, X, mask=attn_mask)
                # Both context and target word embedding will be fed
                # to the decoder
                dec_input = dy.concatenate([beam["wemb"], ctx])
                # Update decoder state
                dec_state = self.dec_cell(dec_input, *beam["state"])
                # Save output
                h = self.dec_cell.get_output(dec_state)
                # Get log_probs
                log_p = dy.log_softmax(self.project(h)).npvalue()
                # top k words
                next_words = log_p.argsort()[-beam_size:]
                # alignments from attention
                align = attn_weights.npvalue().argmax()
                # Add to new beam
                for word in next_words:
                    # Handle stop condition
                    if word == dic_tgt.eos_idx:
                        new_beam = {
                            "words": beam["words"],
                            "score": beam["score"] + log_p[word],
                            "align": beam["align"],
                            "is_over": True,
                        }
                    else:
                        new_beam = {
                            "wemb": self.tgt_embed(word),
                            "state": dec_state,
                            "words": beam["words"] + [word],
                            "score": beam["score"] + log_p[word],
                            "align": beam["align"] + [align],
                            "is_over": False,
                        }
                    new_beams.append(new_beam)

            def beam_score(beam):
                """Helper to score a beam with length penalty"""
                return beam["score"] / (len(beam["words"])+1)**LENPEN
            # Only keep topk new beams
            beams = sorted(new_beams, key=beam_score)[-beam_size:]

        # Return top beam
        return [beams[-1]["words"]], [beams[-1]["align"]]


# Instantiate the network
network = AttBiLSTM(N_LAYERS, EMBED_DIM, HIDDEN_DIM)
# network.pc.save("iwslt_att.model")

# Optimizer
trainer = dy.AdamTrainer(network.pc, alpha=LEARNING_RATE)
trainer.set_clip_threshold(CLIP_NORM)


# Training
# ========

# Create the batch iterators
print("Creating batch iterators")
train_batches = SequencePairsBatches(
    train_src, train_tgt, dic_src, dic_tgt, max_samples=64, max_tokens=2000,
)
dev_batches = SequencePairsBatches(
    dev_src, dev_tgt, dic_src, dic_tgt, max_samples=10
)
test_batches = SequencePairsBatches(
    test_src, test_tgt, dic_src, dic_tgt, max_samples=10
)
print(f"{len(train_batches)} training batches")


# Start training
print("Starting training")
best_ppl = np.inf
# Start training
for epoch in range(N_EPOCHS):
    # Time the epoch
    start_time = time.time()
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
        # Rescale by inverse length
        masked_nll = dy.cdiv(
            masked_nll, dy.inputTensor(tgt.lengths, batched=True))
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
        lls = [dy.pick_batch(lp, y)
               for lp, y in zip(logprobs, tgt.sequences)]
        # Mask losses and reduce
        masked_nll = - stack(lls, d=-1) * tgt.get_mask()
        # Aggregate NLL
        nll += dy.sum_batches(masked_nll).value()
    # Average NLL
    nll /= dev_batches.tgt_size
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

# Evaluation
# ==========

# Load model
print("Reloading best model")
network.pc.populate("iwslt_att.model")


def eval_bleu(batch_iterator, src_sents, tgt_sents, verbose=False):
    """Compute BLEU score over a given dataset"""
    hyps = []
    refs = []
    # Generate from the source data
    for src, tgt in batch_iterator:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Compute logits
        hyp, aligns = network.decode(src, beam_size=BEAM_SIZE)
        # Print
        for b in range(tgt.batch_size):
            # Get original source words
            src_words = src_sents[src.original_idxs[b]]
            hyp_words = dic_tgt.string(hyp[b], join_with=None)
            # replace unks with the alignments given by attention
            for i, w in enumerate(hyp_words):
                if w == dic_tgt.unk_tok:
                    hyp_words[i] = src_words[aligns[b][i]]
            # Join words
            src_sent = " ".join(src_words[:-1])
            hyp_sent = " ".join(hyp_words)
            ref_sent = " ".join(tgt_sents[tgt.original_idxs[b]][:-1])
            # Maybe print
            if verbose:
                print("-"*80)
                print(f"src:\t{src_sent}")
                print(f"hyp:\t{hyp_sent}")
                print(f"ref:\t{ref_sent}")
            # Keep track
            hyps.append(hyp_sent)
            refs.append(ref_sent)
    # BLEU
    return sacrebleu.corpus_bleu(hyps, [refs]).score


# Dev set
dev_bleu = eval_bleu(dev_batches, dev[0], dev[1])
print(f"Dev BLEU: {dev_bleu:.2f}")
# Test set
test_bleu = eval_bleu(test_batches, test[0], test[1])
print(f"Test BLEU: {test_bleu:.2f}")
