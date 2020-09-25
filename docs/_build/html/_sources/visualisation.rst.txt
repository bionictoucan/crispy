Visualisers
===========

The following is describing a unique point and click adventure for imaging spectropolarimetry data available using a Jupyter notebook.

Image Viewer
------------

The following visualisation is for use when only the images are of interest.

.. autoclass:: crisPy2.visualisation.ImageViewer

Spectra Viewer
--------------

The following visualisation tool is for use on **imaging spectroscopic** data and allows the user to explore up to 2 spectral lines simultaneously across an extended field-of-view.

.. autoclass:: crisPy2.visualisation.SpectralViewer
   :members:

Wideband Lightcurve Viewer
--------------------------

For the wideband data, the ``WidebandViewer`` tool can be used similarly to the ``SpectralViewer`` tool described above.

.. autoclass:: crisPy2.visualisation.WidebandViewer
   :members:

Inversion Viewer
----------------

Inversions can be explored in this manner also using ``AtmosViewer`` which will plot the maps of electron density, electron temperature and bulk line-of-sight velocity along with the height profiles of these estimated quantities at given locations (with or without errorbars!).

.. autoclass:: crisPy2.visualisation.AtmosViewer
   :members: