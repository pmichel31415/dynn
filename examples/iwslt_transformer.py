#!/usr/bin/env python3

import sys
from math import ceil
import time

import numpy as np
import dynet as dy
import sacrebleu
import sacremoses

import dynn
from dynn.layers import StackedTransformers, StackedCondTransformers
from dynn.layers import Embeddings
from dynn.layers import Affine
from dynn.layers import Sequential

from dynn.operations import stack
from dynn.util import sin_embeddings
from dynn.parameter_initialization import UniformInit
from dynn.training import inverse_sqrt_schedule

from dynn.data import iwslt, preprocess, Dictionary
from dynn.data.batching import SequencePairsBatches
from dynn.data.caching import cached_to_file

# For reproducibility
dynn.set_random_seed(31415)


# Hyper-parameters
# ================

VOC_SIZE = 30000
LEARNING_RATE_DECAY = 1.0
CLIP_NORM = 5.0
N_LAYERS = 4
MODEL_DIM = 512
N_HEADS = 4
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1
N_EPOCHS = 20
BEAM_SIZE = 4
LENPEN = 0.0


# Data
# ====

@cached_to_file("iwslt.mosestok.bin")
def prepare_data():
    # Download IWSLT
    iwslt.download_iwslt("data", year="2016", langpair="fr-en")

    # Load the data
    print("Loading the IWSLT data")
    train, dev, test = iwslt.load_iwslt("data", year="2016", langpair="fr-en", tgt_eos=None)
    print(f"{len(train[0])} training samples")
    print(f"{len(dev[0])} dev samples")
    print(f"{len(test[0])} test samples")

    print("Tokenizing")
    train, dev, test = preprocess.tokenize([train, dev, test], "moses", "en")
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
    # Append EOS token in the target
    train_tgt = [sent + [dic_tgt.eos_idx] for sent in train_tgt]
    dev_tgt = [sent + [dic_tgt.eos_idx] for sent in dev_tgt]
    test_tgt = [sent + [dic_tgt.eos_idx] for sent in test_tgt]
    return dic_src, dic_tgt, train, dev, test, train_src, dev_src, test_src, train_tgt, dev_tgt, test_tgt

(
    dic_src, dic_tgt, train, dev, test, train_src, dev_src, test_src, train_tgt, dev_tgt, test_tgt
) = prepare_data()


# Model
# =====


class TransformerNetwork(object):
    """This custom layer implements an attention BiLSTM model"""

    def __init__(self, nl, dh, nh, dr):
        # Master parameter collection
        self.pc = dy.ParameterCollection()
        # Hyper-parameters
        self.nl = nl
        self.dh = dh
        self.nh = nh
        self.dr = dr
        # Encoder
        # -------
        # Source Word embeddings
        embed_init = UniformInit(0.1)
        E_src = self.pc.add_parameters((len(dic_src), dh), name="E-src")
        self.src_embed = Embeddings(self.pc, dic_src, dh, params=E_src)
        # Position embeddings
        self.pos_embeds = sin_embeddings(2000, dh, transposed=True)
        # Encoder transformer
        self.enc = StackedTransformers(self.pc, nl, dh, nh, dropout=dr)
        # Decoder
        # --------
        # Word embeddings
        embed_init = UniformInit(0.1)
        E_tgt = self.pc.add_parameters((len(dic_tgt), dh), name="E-tgt")
        self.tgt_embed = Embeddings(self.pc, dic_tgt, dh, params=E_tgt)
        # Start of sentence embedding
        self.sos = self.pc.add_lookup_parameters((1, dh, 1))
        # Transformer
        self.dec = StackedCondTransformers(self.pc, nl, dh, dh, nh, dropout=dr)
        # Projection to logits
        self.project = Sequential(
            # First project to embedding dim
            Affine(self.pc, dh, dh),
            # Then logit layer with weights tied to the word embeddings
            Affine(self.pc, dh, len(dic_tgt), dropout=DROPOUT, W_p=E_tgt)
        )

    def init(self, test=False, update=True):
        self.src_embed.init(test=test, update=update)
        self.enc.init(test=test, update=update)
        self.tgt_embed.init(test=test, update=update)
        self.dec.init(test=test, update=update)
        self.project.init(test=test, update=update)

    def encode(self, src):
        # Embed input words
        src_embs = self.src_embed(src.sequences, length_dim=1) * np.sqrt(self.dh)
        # Add position encodings
        src_embs += dy.inputTensor(self.pos_embeds[:, :src.max_length])
        # Encode
        X = self.enc(src_embs, lengths=src.lengths)
        #  Return list of encodings for each layer
        return X

    def __call__(self, src, tgt):
        # Encode
        # ------
        # Each element of X has shape ``dh x l``
        X = self.encode(src)
        # Decode
        # ------
        L = tgt.max_length
        # Mask for attention
        attn_mask = src.get_mask(base_val=0, mask_val=-np.inf)
        # Embed all words (except EOS)
        tgt_embs = self.tgt_embed(tgt.sequences[:-1], length_dim=1)
        # Add SOS embedding
        sos_embed = self.sos.batch([0] * tgt.batch_size)
        tgt_embs = dy.concatenate([sos_embed, tgt_embs], d=1)
        # Scale embeddings
        tgt_embs = tgt_embs * np.sqrt(self.dh)
        # Add positional encoding (tgt_embs has shape ``dh x L``)
        tgt_embs += dy.inputTensor(self.pos_embeds[:, :L])
        # Decode (h_dec has shape ``dh x L``)
        h_dec = self.dec(tgt_embs, X, mask_c=attn_mask, triu=True)
        # Logits (shape |V| x L)
        logits = self.project(h_dec)
        # Return list of logits (one per position)
        return [dy.pick(logits, index=pos, dim=1) for pos in range(L)]

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
        mask = src.get_mask(base_val=0, mask_val=-np.inf)
        # Max length
        max_len = 2 * src.max_length
        # Initialize beams
        first_beam = {
            "wemb": self.sos[0],  # Previous word embedding
            "state": None,  # Previous state
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
                # Word embedding
                wemb = beam["wemb"] * np.sqrt(self.dh)
                wemb += dy.inputTensor(self.pos_embeds[:, len(beam["words"])])
                # Run one step of the decoder
                new_state, h, _, attn_weights = self.dec.step(
                    beam["state"],
                    wemb,
                    X,
                    mask_c=mask,
                    return_att=True
                )
                # Get log_probs
                log_p = dy.log_softmax(self.project(h)).npvalue()
                # top k words
                next_words = log_p.argsort()[-beam_size:]
                # Alignments from attention (average weights from each head)
                align = dy.average(attn_weights).npvalue()[:, -1].argmax()
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
                            "state": new_state,
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
network = TransformerNetwork(N_LAYERS, MODEL_DIM, N_HEADS, DROPOUT)

# Optimizer
trainer = dy.AdamTrainer(network.pc, beta_1=0.9, beta_2=0.98)
trainer.set_clip_threshold(CLIP_NORM)


learning_rate = inverse_sqrt_schedule(warmup=8000, lr0=1.0 / np.sqrt(MODEL_DIM))

# Training
# ========

# Create the batch iterators
print("Creating batch iterators")
train_batches = SequencePairsBatches(
    train_src, train_tgt, dic_src, dic_tgt, max_samples=128, max_tokens=4000,
)
dev_batches = SequencePairsBatches(
    dev_src, dev_tgt, dic_src, dic_tgt, max_samples=1
)
test_batches = SequencePairsBatches(
    test_src, test_tgt, dic_src, dic_tgt, max_samples=1
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
        trainer.learning_rate = next(learning_rate)
        trainer.update()
        # Print the current loss from time to time
        if train_batches.just_passed_multiple(ceil(len(train_batches)/10)):
            print(
                f"Epoch {epoch+1}@{train_batches.percentage_done():.0f}%: "
                f"NLL={nll.value():.3f} ppl={np.exp(nll.value()):.2f}"
            )
        sys.stdout.flush()

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
        dynn.io.save(network.pc, "iwslt_tf.model")
    elif LEARNING_RATE_DECAY != 1:
        print("Decreasing learning rate")
        trainer.learning_rate /= LEARNING_RATE_DECAY
        print(f"New learning rate: {trainer.learning_rate}")
    sys.stdout.flush()

# Evaluation
# ==========

# Load model
print("Reloading best model")
dynn.io.populate(network.pc, "iwslt_tf.model.npz")


def eval_bleu(batch_iterator, src_sents, tgt_sents, verbose=False):
    """Compute BLEU score over a given dataset"""
    hyps = []
    refs = []
    detok = sacremoses.MosesDetokenizer(lang="en")
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
            src_sent = detok.detokenize(src_words)
            hyp_sent = detok.detokenize(hyp_words)
            ref_sent = detok.detokenize(tgt_sents[tgt.original_idxs[b]])
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
    return sacrebleu.corpus_bleu(hyps, [refs])


# Dev set
dev_bleu = eval_bleu(dev_batches, dev[0], dev[1])
print(f"Dev BLEU: {dev_bleu.score:.2f} (BP={dev_bleu.bp:.2f}, {dev_bleu.sys_len/dev_bleu.ref_len:.2f})")
# Test set
test_bleu = eval_bleu(test_batches, test[0], test[1], verbose=True)
print(f"Dev BLEU: {test_bleu.score:.2f} (BP={test_bleu.bp:.2f}, {test_bleu.sys_len/test_bleu.ref_len:.2f})")
