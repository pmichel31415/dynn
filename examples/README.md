# DyNN examples

- [CNN for MNIST Classification](cnn-mnist)
- [1D CNN for Sentiment Classification on SST](cnn-sst)

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

On a GTX 1080 Ti GPU this runs in ~=1.9s per epoch. The test accuracy after 5 epochs is **81.55%**.
