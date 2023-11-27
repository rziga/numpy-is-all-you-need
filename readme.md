# Transformer in numpy.

Transformer (from the paper [Attention is all you need](https://arxiv.org/abs/1706.03762)) written from scratch in numpy with manual backprop cuz why not.
This is not really meant to be used for anything serious as it was only written for a meme and runs on the CPU and is thus quite slow (although everything is fully vectorized in numpy so it's not that slow either😎).

## How it works

As there is no autograd functionality present in numpy, all gradients are manually backpropagated.
To make my life a bit easier, I use modules, which are similar to interface used in `pytorch.nn`.
But instead only implementing `forward()` in a layer, you need to implement `backward()` as well :).

Unlike Pytorch, however, saving calculations for efficient backprop are saved/handled in layers directly, without any ctx.
E.g. in softmax backprop you use the result of forward in backward to save calculations, so reusing a layer with no parameters is not possible here.

## Does it run?

Yes ... I guess?

Looking at loss curves in [`train example`]("src/train.ipynb"), the losses seem to be decreasing, although the validation loss is kinda suspect (decreasing a bit too well for a bit too long🤔).

![where fig? :c](figs/loss.png "Is this loss?")

I verified backprop of each layer with numerical gradient checking, and the gradients match so that is that.
After inspecting the model, the predictions are just the most common character, so it's likely just that the model is too simple (only 3.5k weights).
Larger models work, but I would like to not fry my laptop completely by repeatedly testing them.


## If you really really want to run

Install numpy:

`pip install numpy`

Optionally, to run train example notebook:

`pip install matplotlib ipykernel`

## TODO

* I verified backprop using a numerical approach, but the code is currently too messy to be included :) So uuhh trust?