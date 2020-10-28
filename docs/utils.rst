Utilities
=========

Sequence Construction
---------------------

The way the input for the sequence data classes works is not the most opaque thing in the world. So we provide a function to generate a list of dictionaries to consturct the sequence data class.

.. autofunction:: crisPy.utils.CRISP_sequence_constructor

Image Utilities
---------------

Oftentimes, the CRISP images need to be augmented in some way for their intended purpose. This could be rotating the images in the image plane to get rid of the background or chopping up the image into segments for work on smaller pieces of the field-of-view. Below is a list of functions to help do these things.

.. autofunction:: crisPy.utils.scanline_search_corners

.. autofunction:: crisPy.utils.refine_corners

.. autofunction:: crisPy.utils.unify_boxes

.. autofunction:: crisPy.utils.find_unified_bb

.. autofunction:: crisPy.utils.rotate_crop_data

.. autofunction:: crisPy.utils.rotate_crop_aligned_data

.. autofunction:: crisPy.utils.reconstruct_full_frame

.. autofunction:: crisPy.utils.segmentation

.. autofunction:: crisPy.utils.segment_cube

.. autofunction:: crisPy.utils.mosaic

.. autofunction:: crisPy.utils.mosaic_cube