Visualisers
===========

The following is describing a unique point and click adventure for imaging spectropolarimetry data available using a Jupyter notebook.

Image Viewer
------------

The following visualisation is for use when only the images are of interest.

.. autoclass:: crisPy.visualisation.ImageViewer
   :members:

Spectra Viewer
--------------

The following visualisation tool is for use on **imaging spectroscopic** data and allows the user to explore up to 2 spectral lines simultaneously across an extended field-of-view.

.. autoclass:: crisPy.visualisation.SpectralViewer
   :members:

Spectral Time Series Viewer
---------------------------

A time series of imaging spectroscopic observations can also be explored with the functionality of the ``crisPy2.visualisation.SpectralViewer`` visualiser in that it can be used to see spectral lines with the added view of the intensity over time for the selected plot at the chosen wavelength.

.. autoclass:: crisPy.visualisation.SpectralTimeViewer
   :members:

Polarimetric Viewer
-------------------

There is also ``crisPy2.visualisation.PolarimetricViewer`` which allows the user to explore imaging spectropolarimetric data in one spectral line similar to how ``crisPy2.visualisation.SpectralViewer`` works.

.. autoclass:: crisPy.visualisation.PolarimetricViewer
   :members:

Polarimetric Time Viewer
------------------------

Similarly, there is a viewer for time series of imaging spectropolarimetric observations.

.. autoclass:: crisPy.visualisation.PolarimetricTimeViewer
   :members:

Wideband Lightcurve Viewer
--------------------------

For the wideband data, the ``WidebandViewer`` tool can be used similarly to the ``SpectralViewer`` tool described above.

.. autoclass:: crisPy.visualisation.WidebandViewer
   :members:

Inversion Viewer
----------------

Inversions can be explored in this manner also using ``AtmosViewer`` which will plot the maps of electron density, electron temperature and bulk line-of-sight velocity along with the height profiles of these estimated quantities at given locations (with or without errorbars!).

.. autoclass:: crisPy.visualisation.AtmosViewer
   :members: