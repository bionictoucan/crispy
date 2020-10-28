Spectrum Properties
===================

The following is a collection of functions to estimate simple statistics for the spectral lines.

Moments of a Spectral Line
---------------------------

We provide functions to calculate the 0th, 1st and 2nd moments of a spectral line. This can be useful for looking at asymmetries in spectral lines and getting a rough idea of line-of-sight Doppler shifts.

.. autofunction:: crisPy.spectral.integrated_intensity

.. autofunction:: crisPy.spectral.bar_lambda

.. autofunction:: crisPy.spectral.variance

Line Asymmetries
----------------

It is often important to look at the asymmetry being the wings of spectral lines as this can help interpret the motion of the plasma that produces these lines. We provide useful functions here for working with line wing.

.. autofunction:: crisPy.spectral.wing_idxs

.. autofunction:: crisPy.spectral.delta_lambda

.. autofunction:: crisPy.spectral.lambda_0_wing

.. autofunction:: crisPy.spectral.intensity_ratio

Miscellaneous
-------------

Here we present some other useful functions when working with spectral lines. Namely, a function to interpolate the line onto a finer (even uniform) grid for more accurate calculations of moments and wings; a function to calculate the line-of-sight Doppler velocity for the intensity-averaged wavelength; and a function to find the azimuthally-averaged power spectrum across an image.

.. autofunction:: crisPy.spectral.interp_fine

.. autofunction:: crisPy.spectral.doppler_vel

.. autofunction:: crisPy.spectral.power_spectrum