Example of Inversion Data in crisPy2
====================================

The data structure for inversion data in ``crisPy2`` is built to work with inversions from the `RADYNVERSION <https://iopscience.iop.org/article/10.3847/1538-4357/ab07b4>`_ code, specificaly whole image inversions (WII). The code can be easily inherited to work for a variety of inversion codes, however. The main things we need is the inversion itself, the corresponding observation, and the height grid that the inversions are calculated on which is stored in the ``crisPy2.Radynversion`` submodule.

.. note:: The ``Inversion`` objects can also be indexed in a similar manner to the other data structures to making plotting easier and working with the data simpler.

.. jupyter-execute::

   from crisPy2.inversions import Inversion
   from crisPy2.Radynversion.utils import z
   from crisPy2.crisp import CRISP

   crisp = CRISP("../examples/2014/crisp_l2_20140906_152724_8542_r00447.fits")
   inversion = Inversion("../examples/inversions/Inversion_0447.hdf5", z=z, header=crisp.file.header)
   print(inversion)

Now that the object exists we can use the built-in plotting methods to first show the line-of-sight fluid velocity at a height roughly corresponding to the formation height of the core of both H :math:`\alpha` and Ca II 8542

.. jupyter-execute::
   :hide-output:

   inversion[15].vel_map()

.. figure:: images/InversionVel.png
   :align: center
   :figclass: align-center

Next we look at the line-of-sight velocity for a specific point on the eastern flare ribbon at (-755,-330)

.. jupyter-execute::

   inversion.from_lonlat(-755,-330)

.. jupyter-execute::
   :hide-output:

   inversion[:,408,298].plot_vel()

.. figure:: images/InversionVel_Height.png
   :align: center
   :figclass: align-center