import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumtrapz

def integrated_intensity(intensity_vector, wavelengths, idx_range="all"):
    """
    A function to find the integrated intensity over a wavelength range of a spectral line.

    .. math::
       I = \int_{\Delta \lambda} I_{\lambda} ~d\lambda

    Parameters
    ----------
    intensity_vector : numpy.ndarray
        The vector of spectral line intensities.
    wavelengths : numpy.ndarray
        The wavelengths to integrate over.
    idx_range : range, optional
        The range of indices to integrate over. Default is "all", integrates over the whole range.
    """

    if idx_range == "all":
        return trapz(intensity_vector, wavelengths)
    else:
        return cumtrapz(integrated_intensity, wavelengths)[idx_range[-1]]

def intensity_ratio(I_1, I_2):
    """
    A function that calculates the intensity ratio of two previously integrated intensities.
    """

    return I_1 / I_2

def doppler_vel(l, del_l):
    """
    A function to calculate the Doppler shift of a line core in km/s.

    .. math::
       v = \frac{\Delta \lambda}{\lambda_{0}} c
    """

    return (del_l / l) * 3e5

def bar_lambda(intensity_vector, wavelengths):
    """
    Calculates the intensity-averaged line core.

    .. math::
       \\bar{\lambda} = \\frac{\int I(\lambda) ~\lambda ~d\lambda}{\int I (\lambda) ~ d\lambda}
    """

    num = trapz(intensity_vector*wavelengths, wavelengths)
    den = trapz(intensity_vector, wavelengths)

    return num / den

def variance(intensity_vector, wavelengths, bar_l):
    """
    Calculates the variance of the spectral line around the intensity-averaged line core.

    .. math::
       \sigma^{2} = \\frac{\int I (\lambda) ~ (\lambda - \\bar{\lambda})^{2} ~ d\lambda}{\int I (\lambda) ~ d\lambda}
    """

    num = trapz(intensity_vector*(wavelengths-bar_l)**2, wavelengths)
    den = trapz(intensity_vector, wavelengths)

    return num / den

def wing_idxs(intensity_vector, wavelengths, var, bar_l):
    """
    A function to work out the index range for the wings of the spectral line. This is working on the definition of wings that says the wings are defined as being one standard deviation away from the intensity-averaged line core.
    """

    blue_wing_start = 0 #blue wing starts at shortest wavelength
    red_wing_end = wavelengths.shape[0] - 1 #red wing ends at longest wavelength

    blue_end_wvl = bar_l - np.sqrt(var)
    red_start_wvl = bar_l + np.sqrt(var)

    blue_wing_end = np.argmin(np.abs(wavelengths - blue_end_wvl))
    red_wing_start = np.argmin(np.abs(wavelengths - red_start_wvl))

    return range(blue_wing_start, blue_wing_end+1), range(red_wing_start, red_wing_end+1)

def delta_lambda(wing_idxs, wavelengths):
    """
    Calculates the half-width in an intensity range i.e. there are N indices spanning a wing then the half-width is half of those indices multiplied by the change in wavelength between two entries. N.B. this is only the case when the space in the spectral axis is uniform.

    .. math::
       \delta \lambda = \\frac{N}{2} \\times (\Delta \lambda)
    """

    return len(wing_idxs) * (wavelengths[1] - wavelengths[0]) / 2

def lambda_0_wing(wing_idxs, wavelengths, delta_lambda):
    """
    Calculates the centre wavelength of an intensity range.
    """

    return wavelengths[wing_idxs[-1]] - delta_lambda

def interp_fine(spec_line):
    """
    Interpolates the spectral line onto a finer grid for more accurate calculations for the wing properties.
    """

    x, y = spec_line
    x_new = np.linspace(x[0], x[-1], num=1001)
    y_new = interp1d(x, y)(x_new)

    return np.array([x_new, y_new])