"""
Basic coordinate testing
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
print(y, x)

assert crisp[:, 10:, :-10].from_lonlat(-720, -300) == (y-10, x)
assert crisp[0, 10:].from_lonlat(-720, -300) == (y-10, x)

print(crisp.to_lonlat(y, x))
print(crisp[:, 10:, -10:].to_lonlat(y-10, x))
print(crisp[0, 10:].to_lonlat(y-10, x))

#%%
crispnonu = CRISP("example_data/2017/ca_00001.zarr")
y, x = crispnonu.from_lonlat(510, -265)
print(y, x)
assert crispnonu[:, :, 10:, :-10].from_lonlat(510, -265) == (y-10, x)
assert crispnonu[:, 1, 10:].from_lonlat(510, -265) == (y-10, x)
assert crispnonu[1, 4, :, 23:].from_lonlat(510, -265) == (y, x-23)

print(crispnonu.to_lonlat(y, x))
print(crispnonu[:, :, 10:, :-10].to_lonlat(y-10, x))
print(crispnonu[:, 1, 10:].to_lonlat(y-10, x))
print(crispnonu[1, 4, :, 23:].to_lonlat(y, x-23))
