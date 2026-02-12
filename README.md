# Neural Networks: Zero to Hero -- Series by Andrej Karpathy

Following [along](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) to build neural networks from scratch.

## Contents
 * [micrograd](#micrograd)
 * [makemore](#makemore)

## micrograd

micrograd is a simple autograd engine that enables the user to perform backpropogation on scalar-valued neural networks. The class Value is defined such that basic algebraic operations can be carried out to form new Values. The class tracks the operations and Values that went into the creation of new Values, forming an expression graph for the purpose of backpropogration. This allows gradients to be computed more efficiently and accurately than through the use of numerical approximation. 

## makemore

makemore is an autoregressive character level language model. It generates fake words that are structurally similar to the words it was trained on.