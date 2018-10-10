# DyNN examples

- [CNN for MNIST Classification](cnn-mnist)
- [1D CNN for Sentiment Classification on SST](cnn-sst)
- [BiLSTM for Sentiment Classification on SST](bilstm-sst)
- [LSTM Language Modeling on the PennTreeBank](lstm-language-modeling-on-the-penntreebank)

## CNN MNIST

[`mnist_cnn.py`](mnist_cnn.py) implements a simple CNN architecture for MNIST classification. Run it with:

```bash
python mnist_cnn.py --dynet-gpus 1
```

On a Titan X GPU this runs in ~=4.6s per epoch. The test accuracy after 5 epochs is **99.10%**.

## CNN SST

[`sst_1d_cnn.py`](sst_1d_cnn.py) implements a simple 1D CNN architecture for sentiment classification on the Stanford Sentiment Treebank. Run it with:

```bash
python sst_1d_cnn.py --dynet-gpus 1
```

On a Titan X GPU this runs in ~=1.8s per epoch. The test accuracy after 5 epochs is **84.62%**.

## BiLSTM SST

[`sst_bilstm.py`](sst_bilstm.py) implements a simple BiLSTM + mean pooling architecture for sentiment classification on the Stanford Sentiment Treebank. Run it with:

```bash
python sst_bilstm.py --dynet-gpus 1
```

On a Titan X GPU this runs in ~=6.3s per epoch. The test accuracy after 2 epochs is **90.99%**.

## LSTM Language Modeling on the PennTreeBank

[`ptb_rnnlm.py`](ptb_rnnlm.py) implement a 2 layer LSTM language model with tied embeddings on the Penn TreeBank (PTB) dataset

On a GTX 1080 Ti GPU this runs in ~=1m per epoch (~=42s with autobatching). The test perplexity after 40 epochs is **84.78**

