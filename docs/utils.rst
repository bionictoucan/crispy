Utilities
=========

Sequence Construction
---------------------

The way the input for the sequence data classes works is not the most opaque thing in the world. So we provide a function to generate a list of dictionaries to consturct the sequence data class.

.. autofunction:: crisPy2.utils.CRISP_sequence_constructor

Image Utilities
---------------

Oftentimes, the CRISP images need to be augmented in some way for their intended purpose. This could be rotating the images in the image plane to get rid of the background or chopping up the image into segments for work on smaller pieces of the field-of-view. Below is a list of functions to help do these things.

.. autofunction:: crisPy2.utils.find_corners

.. autofunction:: crisPy2.utils.im_rotate

.. autofunction:: crisPy2.utils.segmentation

.. autofunction:: crisPy2.utils.segment_cube

.. autofunction:: crisPy2.utils.mosaic

.. autofunction:: crisPy2.utils.mosaic_cube