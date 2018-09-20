# DyNN examples

- [CNN for MNIST Classification](cnn-mnist)

## CNN MNIST

[`mnist_cnn.py`](mnist_cnn.py) implements a simple CNN architecture for MNIST classification. Run it with:

```bash
python mnist_cnn.py --dynet-gpus 1
```

On a Titan X GPU this runs in ~=4.6s per epoch. The test accuracy after 5 epochs is **99.10%**.