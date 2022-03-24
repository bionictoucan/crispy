.. _io:

I/O
===

The following is the documentation for the input/output (I/O) functions for making use of various different types of CRISP data in Python.

CRISP observations are normally shipped in the legacy La Palma data cube format which is not always the most helpful for data analysis and does not have a Python interface. As a result, we have developed a function to memory map these cubes and a secondary function to be able to save these cubes as `zarr <https://zarr.readthedocs.io/en/stable/>`_ files similar to fits files format you might find the data in.

.. autofunction:: crispy.io.memmap_crisp_cube

.. autofunction:: crispy.io.la_palma_cube_to_zarr

The last function in the I/O section is mandatory for using the produced zarr files with the ``crispy.crisp`` data classes. This function converts the header information from the zarr files into a WCS object usable by these data classes.

.. autofunction:: crispy.io.zarr_header_to_wcs