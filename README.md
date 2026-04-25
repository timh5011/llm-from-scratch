# LLMs from scratch

Implementing LLMs from scratch, focused on encoder-decoder transformers.

## Contents
 * [Neural Networks: Hero to Zero](#neural-networks-hero-to-zero)
      * [micrograd](#micrograd)
      * [makemore](#makemore)
* [Representation Learning](#representation-learning)

## Neural Networks: Hero to Zero

Implementing neural networks from scratch based on [Andrej Karpathy's lecture series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

### micrograd

micrograd is a simple autograd engine that aligns with the PyTorch API. It includes an implementation of backpropogation on scalar-valued neural networks. The class Value is defined such that basic algebraic operations can be carried out to form new Values while tracking the operations and Values that went into the creation of new Values, forming an expression graph for the purpose of backpropogration. This allows gradients to be computed more efficiently and accurately than through the use of numerical approximation. 

### makemore

makemore is an autoregressive character level language model. It generates fake words that are structurally similar to the words it was trained on. Our implementation implements key parts of torch.nn using torch.tensor. makemore includes various different models, including:

 * Bigram Language Model
    * trained through counting character pairs (bigrams) to form a character correlation matrix
    * simple neural network (linear layer + softmax layer) trained on bigrams
 * Multi-Layer Perceptron
    * MLP implementation following [Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 
    * with Batch Normalization following [Ioffe et al., 2015](https://arxiv.org/pdf/1502.03167)
 * Convolutional Neural Network
   * CNN implementation modeled off [Google DeepMind's WaveNet, 2016](https://arxiv.org/abs/1609.03499) 
 * Transformer
   * Mini GPT-2 implementation following [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
   * with the following for optimizing deep networks: 
      * Residual Connections from [He et al., 2015](https://arxiv.org/abs/1512.03385)
      * Layer Normalization from [Ba, Kiros, and Hinton, 2016](https://arxiv.org/abs/1607.06450)
      * Dropout (to prevent overfitting) from [Srivastava, 2014](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b4)

## Representation Learning

 * BERT
      * https://arxiv.org/abs/1810.04805
 * Autoencoders and VAEs:
      * https://arxiv.org/abs/1312.6114
 * Representation Learning as Manifold Learning
      * https://arxiv.org/abs/1206.5538
 * Information Bottleneck Theory
      * General Rate-Distortion Theory
      * https://arxiv.org/abs/physics/0004057
      * https://arxiv.org/abs/1703.00810
 * Platonic Representation Hypothesis
      * https://arxiv.org/abs/2405.07987
      * Linear Representation Hypoethsesis: https://arxiv.org/html/2311.03658v2
      * https://arxiv.org/abs/2103.00020
 * Algebraic Study of Latent Space:
      * https://arxiv.org/abs/1003.4394

Also note: word2vec, RNNs, LSTMS, CNNs