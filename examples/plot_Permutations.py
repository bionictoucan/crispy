"""
Non-exhaustive plotting sanity checks
=====================================

The example seeks to test most combination of plot arguments as a smoke test to
catch likely bugs. It isn't a substitute for proper testing, but all I have time
for now.
"""

from crispy import CRISP, CRISPWideband
import warnings

warnings.filterwarnings("ignore")
import pytest
import astropy.units as u

# %%
crisp = CRISP("example_data/2014/crisp_l2_20140906_152724_6563_r00447.fits")
y, x = crisp.from_lonlat(-720, -300)
crisp[:, y, x].plot_spectrum()

# %%
crisp[:, y, x].plot_stokes(stokes="I")
# %%
crisp[:, y, x].plot_spectrum(air=True)
# %%
crisp[:, y, x].plot_spectrum(d=True)
# %%
crisp[:, y, x].plot_spectrum(air=True, d=True)
# %%
crisp[:, y, x].plot_spectrum(unit=u.nm)
# %%
crisp[:, y, x].plot_spectrum(unit=u.nm, d=True)

# %%
with pytest.raises(ValueError):
    crisp[:, y, x].plot_stokes(stokes="all")

# %%
crisp[7].intensity_map()
# %%
crisp[7].intensity_map(frame="pix")
# %%
crisp[7].intensity_map(frame="arcsec")
# %%
crisp[7, 400:500, :].intensity_map()
# %%
crisp[7, :, 400:500].intensity_map()

#%%
crisp = CRISP("example_data/2014/ca_00025.zarr")
y, x = crisp.from_lonlat(507, 264)
crisp[0, 7].intensity_map()
#%%
crisp[0, :, y, x].plot_spectrum()

# %%
crisp[:, :, y, x].plot_stokes(stokes="IQU")
# %%
crisp[0, :, y, x].plot_spectrum(air=True)
# %%
crisp[0, :, y, x].plot_spectrum(d=True)
# %%
crisp[0, :, y, x].plot_spectrum(air=True, d=True)
# %%
crisp[0, :, y, x].plot_spectrum(unit=u.nm)
# %%
crisp[0, :, y, x].plot_spectrum(unit=u.nm, d=True)

# %%
crisp[0, 7].intensity_map()
# %%
crisp[0, 7].intensity_map(frame="pix")
# %%
crisp[0, 3].intensity_map(frame="arcsec")
# %%
crisp[0, 7, 200:250, :].intensity_map()
# %%
crisp[0, 7, :, 400:500].intensity_map()


# %%
crispnonu = CRISP("example_data/2017/ca_00001.zarr")
y, x = crispnonu.from_lonlat(510, -265)
crispnonu[0, :, y, x].plot_spectrum()
# %%
# The conversion is incorrect as the wavelengths were originally provided in air.
crispnonu[0, :, y, x].plot_spectrum(air=True)
# %%
crispnonu[0, :, y, x].plot_spectrum(d=True)

# %%
with pytest.raises(ValueError):
    crispnonu[0, :, y, x].plot_stokes(stokes="IQ")

crispnonu[:, :, y, x].plot_stokes(stokes="IQ")

# %%
crispwideband = CRISPWideband("example_data/2017/wideband/ca_00001.zarr")
crispwideband.intensity_map()
# %%
crispwideband.intensity_map(frame="pix")
# %%
crispwideband.intensity_map(frame="arcsec")
