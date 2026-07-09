Using sphericart with metatensor
--------------------------------

``sphericart`` can be used in conjunction with
`metatensor <https://docs.metatensor.org/latest/index.html>`_ in order to attach
metadata to inputs and outputs, as well as to naturally obtain spherical harmonics,
gradients and Hessians in a single object.

This example shows how to use the ``sphericart.metatensor`` module to compute
spherical harmonics, their gradients and their Hessians.

.. literalinclude:: ../../examples/metatensor/example.py
    :language: python
