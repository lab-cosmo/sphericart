PyTorch
-------

The `PyTorch` implementation follows closely the syntax and usage of the 
`Python` implementation, while also supporting backpropagation. 
The example shows how to compute gradients relative to the input
coordinates by using ``backward()``, and it also illustrates the computation
of second derivatives by reverse-mode autodifferentiation.
The :py:class:`sphericart.torch.SphericalHarmonics` object can also 
be used inside a :py:class:`torch.nn.Module`, that can then be 
compiled using `torchscript`. 

.. literalinclude:: ../../examples/pytorch/example.py
    :language: python
