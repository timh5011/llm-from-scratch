# Neural Networks: Zero to Hero - Series by Andrej Karpathy

Following [along](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) to build neural networks from scratch.

## Contents
 * [micrograd](#micrograd)
 * [makemore](#makemore)

## micrograd

micrograd is a simple autograd engine that aligns with the PyTorch API. It includes an implementation of backpropogation on scalar-valued neural networks. The class Value is defined such that basic algebraic operations can be carried out to form new Values while tracking the operations and Values that went into the creation of new Values, forming an expression graph for the purpose of backpropogration. This allows gradients to be computed more efficiently and accurately than through the use of numerical approximation. 

## makemore

makemore is an autoregressive character level language model. It generates fake words that are structurally similar to the words it was trained on. makemore includes various different models, including:

 * Bigram Language Model
    * trained through counting character pairs (bigrams) to form a character correlation matrix
    * simple neural network (linear layer + softmax layer) trained on bigrams
 * Multi-Layer Perceptron
    * MLP implementation following [Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) with batch normalization following [Ioffe et al., 2015](https://arxiv.org/pdf/1502.03167)
 * Convolutional Neural Network
   * CNN implementation modeled off [Google DeepMind's WaveNet, 2016](https://arxiv.org/abs/1609.03499) 
