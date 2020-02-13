import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import os, h5py, yaml
import astropy.units as u
from astropy.io import fits

class ObjDict(dict):
    '''
    This is an abstract class for allowing the keys of a dictionary to be accessed like class attributes.
    '''

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: "+name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: "+name)

class CRISP:
    '''
    This is a class for CRISP data cubes to make it easy to plot and obtain spectral line profiles from the data.

    Parameters
    ----------
    files: str or list of str
        The files to be read into the data structure. Can be one file or co-temporal files of different wavelengths.
    align : bool, optional
        Whether or not to align two images of different wavelengths as there are slight discrepancies in positional information. This results in a slight crop of the images. Default is False.
    '''

    def __init__(self, files, align=False):
        self.mm_sun = 1391 * u.megameter
        self.ang_sun = 1920 * u.arcsec
        if type(files) == str:
            if ".fits" in files:
                if "ca" in files:
                    self.ca = fits.open(files)[0]
                    try:
                        self.ca_wvls = self.ca.header["spect_pos"]
                    except KeyError:
                        self.ca_wvls = fits.open(files)[1].data
                else:
                    self.ha = fits.open(files)[0]
                    try:
                        self.ha_wvls = self.ha.header["spect_pos"]
                    except KeyError:
                        self.ha_wvls = fits.open(files)[1].data
            elif ".h5" or ".hdf5" in files:
                if "ca" in files:
                    ca = h5py.File(files, "r")
                    self.ca = ObjDict({})
                    self.ca["data"] = ca["data"]
                    self.ca["header"] = yaml.load(ca["header"][0], Loader=yaml.Loader)
                    self.ca_wvls = self.ca.header["spect_pos"]
                else:
                    ha = h5py.File(files, "r")
                    self.ha = ObjDict({})
                    self.ha["data"] = ha["data"]
                    self.ha["header"] = yaml.load(ha["header"][0], Loader=yaml.Loader)
                    self.ha_wvls = self.ha.header["spect_pos"]
        else:
            for f in files:
                if ".fits" in f:
                    if "ca" in f:
                        self.ca = fits.open(f)[0]
                        try:
                            self.ca_wvls = self.ca.header["spect_pos"]
                        except KeyError:
                            self.ca_wvls = fits.open(f)[1].data
                    else:
                        self.ha = fits.open(f)[0]
                        try:
                            self.ha_wvls = self.ha.header["spect_pos"]
                        except KeyError:
                            self.ha_wvls = fits.open(f)[1].data
                elif ".h5" or ".hdf5" in f:
                    if "ca" in f:
                        ca = h5py.File(f, "r")
                        self.ca = ObjDict({})
                        self.ca["data"] = ca["data"]
                        self.ca["header"] = yaml.load(ca["header"][0], Loader=yaml.Loader)
                        self.ca_wvls = self.ca.header["spect_pos"]
                    else:
                        ha = h5py.File(f, "r")
                        self.ha = ObjDict({})
                        self.ha["data"] = ha["data"]
                        self.ha["header"] = yaml.load(ha["header"][0], Loader=yaml.Loader)
                        self.ha_wvls = self.ha.header["spect_pos"]

    def __str__(self):
        if "ca" and "ha" in self.__dict__:
            date = self.ca.header["DATE-AVG"][:10]
            time = self.ca.header["DATE-AVG"][11:-4]
            el_1 = self.ca.header["WDESC1"]
            samp_wvl_1 = str(self.ca.header["NAXIS3"])
            el_2 = self.ha.header["WDESC1"]
            samp_wvl_2 = str(self.ha.header["NAXIS3"])
            pointing = (str(round(self.ca.header["CRVAL1"],3)), str(round(self.ca.header["CRVAL2"],3)))
            if len(self.ca.data.shape) == 3:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the elements {el_1} angstroms and {el_2} angstroms with {samp_wvl_1} and {samp_wvl_2} wavelengths sampled, respectively. Heliocentric coodinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. Only Stokes I present in these observations."
            elif len(self.ca.data.shape) == 4:
               return f"CRISP observation from {date} {time} UTC with measurements taken in the elements {el_1} angstroms and {el_2} angstroms with {samp_wvl_1} and {samp_wvl_2} wavelengths sampled, respectively. Heliocentric coodinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. All Stokes parameters present in these observations."
        elif "ca" and not "ha" in self.__dict__:
            date = self.ca.header["DATE-AVG"][:10]
            time = self.ca.header["DATE-AVG"][11:-4]
            el_1 = self.ca.header["WDESC1"]
            samp_wvl_1 = str(self.ca.header["NAXIS3"])
            pointing = (str(round(self.ca.header["CRVAL1"],3)), str(round(self.ca.header["CRVAL2"],3)))
            if len(self.ca.data.shape) == 3:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the elements {el_1} angstroms with {samp_wvl_1} wavelengths sampled and with heliocentric coordinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. Only Stokes I present in this observation."
            elif len(self.ca.data.shape) == 4:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the elements {el_1} angstroms with {samp_wvl_1} wavelengths sampled and with heliocentric coordinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. All Stokes parameters present in these observations."
        elif not "ca" and "ha" in self.__dict__:
            date = self.ha.header["DATE-AVG"][:10]
            time = self.ha.header["DATE-AVG"][11:-4]
            el_1 = self.ha.header["WDESC1"]
            samp_wvl_1 = str(self.ha.header["NAXIS3"])
            pointing = (str(round(self.ha.header["CRVAL1"],3)), str(round(self.ha.header["CRVAL2"],3)))
            if len(self.ha.data.shape) == 3:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the element {el_1} angstroms with {samp_wvl_1} wavelengths sampled and with heliocentric coordinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. Only Stokes I present in this observation."
            elif len(self.ha.data.shape) == 4:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the element {el_1} angstroms with {samp_wvl_1} wavelengths sampled and with heliocentric coordinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. All Stokes parameters present in these observations."

    @staticmethod
    def unit_conversion(coord, unit_to, centre=False):
        