import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import os, h5py, yaml, html
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective
from sunpy.time import parse_time
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from utils import *

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
        self.mm_sun = 1391 << u.megameter
        self.ang_sun = 1920 << u.arcsec
        if type(files) == str:
            if ".fits" in files:
                if "ca" in files:
                    self.ca = fits.open(files)[0]
                    try:
                        self.ca_wvls = self.ca.header["spect_pos"] << u.Angstrom
                    except KeyError:
                        self.ca_wvls = fits.open(files)[1].data << u.Angstrom
                    self.px_res = self.ca.header["CDELT1"] << u.arcsec / u.pixel
                    self.pointing = (self.ca.header["CRVAL2"], self.ca.header["CRVAL1"]) << u.arcsec
                    self.mid = (self.ca.header["NAXIS2"] // 2, self.ca.header["NAXIS1"] // 2) << u.pixel
                else:
                    self.ha = fits.open(files)[0]
                    try:
                        self.ha_wvls = self.ha.header["spect_pos"] << u.Angstrom
                    except KeyError:
                        self.ha_wvls = fits.open(files)[1].data << u.Angstrom
                    self.px_res = self.ha.header["CDELT1"] << u.arcsec / u.pixel
                    self.pointing = (self.ha.header["CRVAL2"], self.ha.header["CRVAL1"]) << u.arcsec
                    self.mid = (self.ha.header["NAXIS2"] // 2, self.ha.header["NAXIS1"] // 2) << u.pixel
            elif ".h5" or ".hdf5" in files:
                if "ca" in files:
                    ca = h5py.File(files, "r")
                    self.ca = ObjDict({})
                    self.ca["data"] = ca["data"]
                    self.ca["header"] = yaml.load(ca["header"][0], Loader=yaml.Loader)
                    self.ca_wvls = self.ca.header["spect_pos"] << u.Angstrom
                    self.px_res = self.ca.header["pixel_scale"] << u.arcsec / u.pixel
                    self.pointing = (self.ca.header["crval"][-2], self.ca.header["crval"][-1]) << u.arcsec
                    self.mid = (self.ca.header["dimensions"][-2] // 2, self.ca.header["dimensions"][-1] // 2) << u.pixel
                else:
                    ha = h5py.File(files, "r")
                    self.ha = ObjDict({})
                    self.ha["data"] = ha["data"]
                    self.ha["header"] = yaml.load(ha["header"][0], Loader=yaml.Loader)
                    self.ha_wvls = self.ha.header["spect_pos"] << u.Angstrom
                    self.px_res = self.ha.header["pixel_scale"] << u.arcsec / u.pixel
                    self.pointing = (self.ha.header["crval"][-2], self.ha.header["crval"][-1]) << u.arcsec
                    self.mid = (self.ha.header["dimensions"][-2] // 2, self.ha.header["dimensions"][-1]) << u.pixel
        else:
            for f in files:
                if ".fits" in f:
                    if "ca" in f:
                        self.ca = fits.open(f)[0]
                        try:
                            self.ca_wvls = self.ca.header["spect_pos"] << u.Angstrom
                        except KeyError:
                            self.ca_wvls = fits.open(f)[1].data << u.Angstrom
                        self.px_res = self.ca.header["CDELT1"] << u.arcsec / u.pixel
                        self.pointing = (self.ca.header["CRVAL2"], self.ca.header["CRVAL1"]) << u.arcsec
                        self.mid = (self.ca.header["NAXIS2"] // 2, self.ca.header["NAXIS1"] // 2) << u.pixel
                    else:
                        self.ha = fits.open(f)[0]
                        try:
                            self.ha_wvls = self.ha.header["spect_pos"] << u.Angstrom
                        except KeyError:
                            self.ha_wvls = fits.open(f)[1].data << u.Angstrom
                elif ".h5" or ".hdf5" in f:
                    if "ca" in f:
                        ca = h5py.File(f, "r")
                        self.ca = ObjDict({})
                        self.ca["data"] = ca["data"]
                        self.ca["header"] = yaml.load(ca["header"][0], Loader=yaml.Loader)
                        self.ca_wvls = self.ca.header["spect_pos"] << u.Angstrom
                        self.px_res = self.ca.header["pixel_scale"] << u.arcsec / u.pixel
                        self.pointing = (self.ca.header["crval"][-2], self.ca.header["crval"][-1]) << u.arcsec
                        self.mid = (self.ca.header["dimensions"][-2] // 2, self.ca.header["dimensions"][-1] // 2) << u.pixel
                    else:
                        ha = h5py.File(f, "r")
                        self.ha = ObjDict({})
                        self.ha["data"] = ha["data"]
                        self.ha["header"] = yaml.load(ha["header"][0], Loader=yaml.Loader)
                        self.ha_wvls = self.ha.header["spect_pos"] << u.Angstrom

    def __str__(self):
        if "ca" and "ha" in self.__dict__:
            try:
                date = self.ca.header["DATE-AVG"][:10]
                time = self.ca.header["DATE-AVG"][11:-4]
                el_1 = self.ca.header["WDESC1"]
                samp_wvl_1 = str(self.ca.header["NAXIS3"])
                el_2 = self.ha.header["WDESC1"]
                samp_wvl_2 = str(self.ha.header["NAXIS3"])
                pointing = (str(round(self.ca.header["CRVAL1"],3)), str(round(self.ca.header["CRVAL2"],3)))
            except KeyError: # must be using hdf5 data files
                date = self.ca.header["date-obs"]
                time = self.ca.header["time-obs"]
                el_1 = self.ca.header["element"]
                samp_wvl_1 = str(self.ca.header["dimensions"][-3])
                el_2 = self.ha.header["element"]
                samp_wvl_2 = str(self.ha.header["dimensions"][-3])
                pointing = (str(self.ca.header["crval"][-1]), str(self.ca.header["crval"][-2]))
            if len(self.ca.data.shape) == 3:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the elements {el_1} angstroms and {el_2} angstroms with {samp_wvl_1} and {samp_wvl_2} wavelengths sampled, respectively. Heliocentric coodinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. Only Stokes I present in these observations."
            elif len(self.ca.data.shape) == 4:
               return f"CRISP observation from {date} {time} UTC with measurements taken in the elements {el_1} angstroms and {el_2} angstroms with {samp_wvl_1} and {samp_wvl_2} wavelengths sampled, respectively. Heliocentric coodinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. All Stokes parameters present in these observations."
        elif "ca" and not "ha" in self.__dict__:
            try:
                date = self.ca.header["DATE-AVG"][:10]
                time = self.ca.header["DATE-AVG"][11:-4]
                el_1 = self.ca.header["WDESC1"]
                samp_wvl_1 = str(self.ca.header["NAXIS3"])
                pointing = (str(round(self.ca.header["CRVAL1"],3)), str(round(self.ca.header["CRVAL2"],3)))
            except KeyError:
                date = self.ca.header["date-obs"]
                time = self.ca.header["time-obs"]
                el_1 = self.ca.header["element"]
                samp_wvl_1 = str(self.ca.header["dimensions"][-3])
                pointing = (str(self.ca.header["crval"][-1]), str(self.ca.header["crval"][-2]))
            if len(self.ca.data.shape) == 3:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the elements {el_1} angstroms with {samp_wvl_1} wavelengths sampled and with heliocentric coordinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. Only Stokes I present in this observation."
            elif len(self.ca.data.shape) == 4:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the elements {el_1} angstroms with {samp_wvl_1} wavelengths sampled and with heliocentric coordinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. All Stokes parameters present in these observations."
        elif not "ca" and "ha" in self.__dict__:
            try:
                date = self.ha.header["DATE-AVG"][:10]
                time = self.ha.header["DATE-AVG"][11:-4]
                el_1 = self.ha.header["WDESC1"]
                samp_wvl_1 = str(self.ha.header["NAXIS3"])
                pointing = (str(round(self.ha.header["CRVAL1"],3)), str(round(self.ha.header["CRVAL2"],3)))
            except KeyError:
                date = self.ha.header["date-obs"]
                time = self.ha.header["time-obs"]
                el_1 = self.ha.header["element"]
                samp_wvl_1 = str(self.ha.header["dimensions"][-3])
                pointing = (str(self.ha.header["crval"][-1]), str(self.ha.header["crval"][-2]))
            if len(self.ha.data.shape) == 3:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the element {el_1} angstroms with {samp_wvl_1} wavelengths sampled and with heliocentric coordinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. Only Stokes I present in this observation."
            elif len(self.ha.data.shape) == 4:
                return f"CRISP observation from {date} {time} UTC with measurements taken in the element {el_1} angstroms with {samp_wvl_1} wavelengths sampled and with heliocentric coordinates at the centre of the image ({pointing[0]},{pointing[1]}) in arcseconds. All Stokes parameters present in these observations."

    def unit_conversion(self, coord, unit_to, centre=False):
        '''
        A method to convert unit coordinates between pixel number, arcsecond (either absolute or relative to pointing) and megametre.

        Parameters
        ----------
        coord : astropy.units.quantity.Quantity
            The coordinate to be transformed.
        unit_to : str
            The coordinate system to convert to.
        centre : bool, optional
            Whether or not to calculate the pixel in arcseconds with respect to the pointing e.g. in the helioprojective frame.
        '''

        if not centre:
            if unit_to == "pix":
                if coord.unit == "pix":
                    return coord
                elif coord.unit == "arcsec":
                    return np.round(coord / self.px_res)
                elif coord.unit == "megameter":
                    return np.round((coord * self.ang_sun / self.mm_sun) / self.px_res) 
            elif unit_to == "arcsec":
                if coord.unit == "pix":
                    return coord * self.px_res
                elif coord.unit == "arcsec":
                    return coord
                elif coord.unit == "megameter":
                    return coord * (self.ang_sun / self.mm_sun)
            elif unit_to == "Mm":
                if coord.unit == "pix":
                    return (coord * self.px_res) * self.mm_sun / self.ang_sun
                elif coord.unit == "arcsec":
                    return coord * self.mm_sun / self.ang_sun
                elif coord.unit == "megameter":
                    return coord
        else:
            # N.B. the conversions which take into account the centre pixel in the helioprojective coordinate frame assume that the given coordinate is in (y, x) format, whereas the other conversions can be done either way round e.g. (y, x) or (x, y)
            if unit_to == "pix":
                if coord.unit == "pix":
                    return coord
                elif coord.unit == "arcsec":
                    px_diff = np.round((coord - self.pointing) / self.px_res)
                    return self.mid + px_diff
                elif coord.unit == "megameter":
                    return self.unit_conversion(coord, unit_to="pix", centre=False)
            elif unit_to == "arcsec":
                if coord.unit == "pix":
                    return ((coord - self.mid) * self.px_res) + self.pointing
                elif coord.unit == "arcsec":
                    # N.B. the assumption here is that you will only use this if you want to convert from arcseconds to helioprojective coordinate frame
                    return (coord - (self.mid * self.px_res) + self.pointing)
                elif coord.unit == "megameter":
                    # this code is really fucking bad but I couldn't be bothered doing it properly so
                    return self.unit_conversion(self.unit_conversion(coord, unit_to="pix", centre=False), unit_to="arcsec", centre=True)
            elif unit_to == "megameter":
                if coord.unit == "pix":
                    return self.unit_conversion(coords, unit_to="pix", centre=False)
                elif coord.unit == "arcsec":
                    # ditto for this one
                    return self.unit_conversion(self.unit_conversion(coord, unit_to="pix", centre=True), unit_to="megameter", centre=False)
                elif coord.unit == "megameter":
                    return coord

    def intensity_vector(self, coord, line, pol=False, centre=False):
        '''
        A class method for returning the intensity vector of a given pixel.

        Parameters
        ----------
        coord : astropy.unit.quantity.Quantity
            The coordinate to give the intensity vector for.
        line : str
            The line to get the intensity vector for. Can be "ca" or "ha".
        pol : bool, optional
            Whether or not to return the polarimetric intensity vector. Default is False.
        centre : bool, optional
            Whether or not to calculate the pixel in arcseconds with respect to the pointing e.g. in the helioprojective frame.
        '''

        if line == "ha" and pol:
            raise WiseGuyError("Tryin' to be a wise guy, eh?")

        coord = self.unit_conversion(coord, unit_to="pix", centre=centre)

        if line.lower() == "ca":
            if not pol:
                if len(self.ca.data.shape) == 4:
                    return self.ca.data[0, :, int(coord[0].value), int(coord[1].value)]
                else:
                    return self.ca.data[:, int(coord[0].value), int(coord[1].value)]
            else:
                if len(self.ca.data.shape) == 4:
                    return self.ca.data[:, :, int(coord[0].value), int(coord[1].value)]
                else:
                    raise WiseGuyError("Tryin' to be a wise guy, eh?")
        elif line.lower() == "ha":
            return self.ha.data[:, int(coord[0].value), int(coord[1].value)]

    def coalign(self):
        '''
        A class method to coalign the calcium and hydrogen images as they are slightly offset with respect to one another. This can be used as an error in the pointing/position information.
        '''
        try:
            start_time = parse_time(self.ha.header["DATE-AVG"])
            end_time = parse_time(self.ca.header["DATE-AVG"])
        except KeyError:
            start_time = parse_time(self.ha.header["date-obs"] + " " + self.ha.header["time-obs"])
            end_time = parse_time(self.ca.header["date-obs"] + " " + self.ca.header["time-obs"])

        c = SkyCoord(self.pointing[1], self.pointing[0], obstime=start_time, observer="earth", frame=Helioprojective)
        new_c = solar_rotate_coordinate(c, time=end_time)

        diff = [(new_c.Ty - c.Ty).value, (new_c.Tx - c.Tx).value] << u.arcsec
        return diff / self.px_res

    def plot_spectrum(self, coord, line, centre=False, d=False):
        # TODO: Add grid to plot and inward ticks.
        '''
        A class method to take a coordinate on the plot and plot the spectral line of that particular location with intensities given in data numbers [DNs] not real units.

        Parameters
        ----------
        coord : astropy.unit.quantity.Quantity
            The coordinates of the point that the user wants the spectral line from.
        line : str
            The line to plot. Can be "both", "ca" or "ha".
        centre : bool, optional
            Whether or not to calculate the pixel in arcseconds with respect to the pointing e.g. in the helioprojective frame. Default is False.
        d : bool, optional
            Whether or not to use Delta lambda for the x-axis. Default is False.
        '''

        coord = self.unit_conversion(coord, unit_to="pix", centre=centre)
        l = html.unescape("&lambda;")
        aa = html.unescape("&#8491;")
        D = html.unescape(("&Delta;"))
        a = html.unescape("&alpha;")

        if line == "both":
            if not d:
                fig = plt.figure()
                ax1 = fig.add_subplot(1,2,1)
                if len(self.ca.data.shape) == 4:
                    ax1.plot(self.ca_wvls, self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                else:
                    ax1.plot(self.ca_wvls, self.ca.data[:, int(coord[0].value), int(coord[1].value)])
                ax1.set_title(f"Ca II {l}8542")
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                
                ax2 = fig.add_subplot(1,2,2)
                ax2.plot(self.ha_wvls, self.ha.data[:, int(coord[0].value), int(coord[1].value)])
                ax2.set_title(f"H{a} {l}6563")
                ax2.set_ylabel("Intensity [DNs]")
                ax2.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(1,2,1)
                if len(self.ca.data.shape) == 4:
                    ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                else:
                    ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, int(coord[0].value), int(coord[1].value)])
                ax1.set_title(f"Ca II {l}8542")
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                
                ax2 = fig.add_subplot(1,2,2)
                ax2.plot(self.ha_wvls - np.median(self.ha_wvls), self.ha.data[:, int(coord[0].value), int(coord[1].value)])
                ax2.set_title(f"H{a} {l}6563")
                ax2.set_ylabel("Intensity [DNs]")
                ax2.set_xlabel(f"{D} {l} [{aa}]")
                fig.show()
        elif line == "ca":
            if not d:
                fig = plt.figure()
                ax1 = fig.gca()
                if len(self.ca.data.shape) == 4:
                    ax1.plot(self.ca_wvls, self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                else:
                    ax1.plot(self.ca_wvls, self.ca.data[:, int(coord[0].value), int(coord[1].value)])
                ax1.set_title(f"Ca II {l}8542")
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.gca()
                if len(self.ca.data.shape) == 4:
                    ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                else:
                    ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, int(coord[0].value), int(coord[1].value)])
                ax1.set_title(f"Ca II {l}8542")
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                fig.show()
        elif line == "ha":
            if not d:                
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ha_wvls, self.ha.data[:, int(coord[0].value), int(coord[1].value)])
                ax1.set_title(f"H{a} {l}6563")
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ha_wvls - np.median(self.ha_wvls), self.ha.data[:, int(coord[0].value), int(coord[1].value)])
                ax1.set_title(f"H{a} {l}6563")
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                fig.show()

    def plot_stokes(self, coord, stokes, centre=False, d=False):
        # TODO: Add grid for plots and ticks facing inwards.
        '''
        A class method to take a coordinate on the plot and plot the Stokes profile of that particular location with intensities given in data numbers [DNs] not real units.

        Parameters
        ----------
        coord : astropy.unit.quantity.Quantity
            The coordinates of the point that the user wants the spectral line from.
        stokes : str
            The stokes profile to plot. Can be "all", "I", "Q", "U", "V" or "IV".
        centre : bool, optional
            Whether or not to calculate the pixel in arcseconds with respect to the pointing e.g. in the helioprojective frame. Default is False.
        d : bool, optional
            Whether or not to use Delta lambda for the x-axis. Default is False.
        '''

        coord = self.unit_conversion(coord, unit_to="pix", centre=centre)
        l = html.unescape("&lambda;")
        aa = html.unescape("&#8491;")
        D = html.unescape(("&Delta;"))

        if stokes == "all":
            if not d:
                fig = plt.figure()
                ax1 = fig.add_subplot(2,2,1)
                ax1.plot(self.ca_wvls, self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("I [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                ax2 = fig.add_subplot(2,2,2)
                ax2.plot(self.ca_wvls, self.ca.data[:, 1, int(coord[0].value), int(coord[1].value)])
                ax2.set_ylabel("Q [DNs]")
                ax2.set_xlabel(f"{l} [{aa}]")
                ax3 = fig.add_subplot(2,2,3)
                ax3.plot(self.ca_wvls, self.ca.data[:, 2, int(coord[0].value), int(coord[1].value)])
                ax3.set_ylabel("U [DNs]")
                ax3.set_xlabel(f"{l} [{aa}]")
                ax4 = fig.add_subplot(2,2,4)
                ax4.plot(self.ca_wvls, self.ca.data[:, 3, int(coord[0].value), int(coord[1].value)])
                ax4.set_ylabel("V [DNs]")
                ax4.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(2,2,1)
                ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("I [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                ax2 = fig.add_subplot(2,2,2)
                ax2.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 1, int(coord[0].value), int(coord[1].value)])
                ax2.set_ylabel("Q [DNs]")
                ax2.set_xlabel(f"{D} {l} [{aa}]")
                ax3 = fig.add_subplot(2,2,3)
                ax3.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 2, int(coord[0].value), int(coord[1].value)])
                ax3.set_ylabel("U [DNs]")
                ax3.set_xlabel(f"{D} {l} [{aa}]")
                ax4 = fig.add_subplot(2,2,4)
                ax4.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 3, int(coord[0].value), int(coord[1].value)])
                ax4.set_ylabel("V [DNs]")
                ax4.set_xlabel(f"{D} {l} [{aa}]")
                fig.show()
        elif stokes == "I":
            if not d:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ca_wvls, self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("I [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("I [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                fig.show()
        elif stokes == "Q":
            if not d:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ca_wvls, self.ca.data[:, 1, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("Q [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 1, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("Q [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                fig.show()
        elif stokes == "U":
            if not d:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ca_wvls, self.ca.data[:, 2, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("U [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 2, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("U [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                fig.show()
        elif stokes == "V":
            if not d:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ca_wvls, self.ca.data[:, 3, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("V [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.gca()
                ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 3, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("V [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                fig.show()
        elif stokes == "IV":
            if not d:
                fig = plt.figure()
                ax1 = fig.add_subplot(1,2,1)
                ax1.plot(self.ca_wvls, self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("I [DNs]")
                ax1.set_xlabel(f"{l} [{aa}]")
                ax2 = fig.add_subplot(1,2,2)
                ax2.plot(self.ca_wvls, self.ca.data[:, 3, int(coord[0].value), int(coord[1].value)])
                ax2.set_ylabel("V [DNs]")
                ax2.set_xlabel(f"{l} [{aa}]")
                fig.show()
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(1,2,1)
                ax1.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 0, int(coord[0].value), int(coord[1].value)])
                ax1.set_ylabel("I [DNs]")
                ax1.set_xlabel(f"{D} {l} [{aa}]")
                ax2 = fig.add_subplot(1,2,2)
                ax2.plot(self.ca_wvls - np.median(self.ca_wvls), self.ca.data[:, 3, int(coord[0].value), int(coord[1].value)])
                fig.show()

    def intensity_map(self, line, frame, ca_wvl_idx=None, ha_wvl_idx=None):
        # TODO: Add colourbars for the plots.
        '''
        A class method to plot the intensity map of a chosen spectral line.

        Parameters
        ----------
        line : str
            Which spectral line to plot the intensity map for. Can be "both", "ca" or "ha".
        frame : str
            Which frame to plot the intensity map in. Can be "pix", "arcsec", "megameter" or "helioprojective".
        ca_wvl_idx : int, optional
            The wavelength to plot for calcium. Default is None.
        ha_wvl_idx : int, optional
            The wavelength to plot for hydrogen. Default is None.
        '''

        l = html.unescape("&lambda;")
        a = html.unescape("&alpha;")
        aa = html.unescape("&#8491;")
        D = html.unescape("&Delta;")
        
        if line == "both":
            assert ca_wvl_idx is not None
            assert ha_wvl_idx is not None
            fig = plt.figure()
            ca_ax = fig.add_subplot(1,2,1)
            ha_ax = fig.add_subplot(1,2,2)
            ca_ax.set_title(f"Ca II {l}8542, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            ha_ax.set_title(f"H{a} {l}6563, {D} {l} = "+str(self.ha_wvls[ha_wvl_idx] - np.median(self.ha_wvls))+f"{aa}")
            ha_ax.tick_params(labelleft=False)
            if frame == "pix":
                if len(self.ca.data.shape) == 4:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx,0], cmap="greys_r", origin="lower")
                else:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx], cmap="Greys_r", origin="lower")
                ha_ax.imshow(self.ha.data[ha_wvl_idx], cmap="Greys_r", origin="lower")
                ca_ax.set_ylabel("y [pixels]")
                ca_ax.set_xlabel("x [pixels]")
                ha_ax.set_xlabel("x [pixels]")
            elif frame == "arcsec":
                ca_tr = self.unit_conversion(self.ca.data.shape[-2:] << u.pix, unit_to="arcsec")
                ha_tr = self.unit_conversion(self.ha.data.shape[-2:] << u.pix, unit_to="arcsec")
                if len(self.ca.data.shape) == 4:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx,0], cmap="greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                else:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                ha_ax.imshow(self.ha.data[ha_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ha_tr[1], 0, ha_tr[0]])
                ca_ax.set_ylabel("y [arcseconds]")
                ca_ax.set_xlabel("x [arcseconds]")
                ha_ax.set_xlabel("x [arcseconds]")
            elif frame == "megameter":
                ca_tr = self.unit_conversion(self.ca.data.shape[-2:] << u.pix, unit_to="megameter")
                ha_tr = self.unit_conversion(self.ha.data.shape[-2:] << u.pix, unit_to="megameter")
                if len(self.ca.data.shape) == 4:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx,0], cmap="greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                else:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                ha_ax.imshow(self.ha.data[ha_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ha_tr[1], 0, ha_tr[0]])
                ca_ax.set_ylabel("y [Mm]")
                ca_ax.set_xlabel("x [Mm]")
                ha_ax.set_xlabel("x [Mm]")
            elif frame == "helioprojective":
                ca_tr = self.unit_conversion(self.ca.data.shape[-2:] << u.pix, unit_to="arcsec", centre=True)
                ha_tr = self.unit_conversion(self.ha.data.shape[-2:] << u.pix, unit_to="arcsec", centre=True)
                if len(self.ca.data.shape) == 4:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx,0], cmap="greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                else:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                ha_ax.imshow(self.ha.data[ha_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ha_tr[1], 0, ha_tr[0]])
                ca_ax.set_ylabel("Helioprojective-y [arcsec]")
                ca_ax.set_xlabel("Helioprojective-x [arcsec]")
                ha_ax.set_xlabel("Helioprojective-x [arcsec]")
        elif line == "ca":
            fig = plt.figure()
            ca_ax = fig.add_subplot(1,1,1)
            ca_ax.set_title(f"Ca II {l}8542, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            if frame == "pix":
                if len(self.ca.data.shape) == 4:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx,0], cmap="greys_r", origin="lower")
                else:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx], cmap="Greys_r", origin="lower")
                ca_ax.set_ylabel("y [pixels]")
                ca_ax.set_xlabel("x [pixels]")
            elif frame == "arcsec":
                ca_tr = self.unit_conversion(self.ca.data.shape[-2:] << u.pix, unit_to="arcsec")
                if len(self.ca.data.shape) == 4:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx,0], cmap="greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                else:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                ca_ax.set_ylabel("y [arcseconds]")
                ca_ax.set_xlabel("x [arcseconds]")
            elif frame == "megameter":
                ca_tr = self.unit_conversion(self.ca.data.shape[-2:] << u.pix, unit_to="megameter")
                if len(self.ca.data.shape) == 4:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx,0], cmap="greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                else:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                ca_ax.set_ylabel("y [Mm]")
                ca_ax.set_xlabel("x [Mm]")
            elif frame == "helioprojective":
                ca_tr = self.unit_conversion(self.ca.data.shape[-2:] << u.pix, unit_to="arcsec", centre=True)
                if len(self.ca.data.shape) == 4:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx,0], cmap="greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                else:
                    ca_ax.imshow(self.ca.data[ca_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ca_tr[1], 0, ca_tr[0]])
                ca_ax.set_ylabel("Helioprojective-y [arcsec]")
                ca_ax.set_xlabel("Helioprojective-x [arcsec]")
        elif line == "ha":
            fig = plt.figure()
            ha_ax = fig.add_subplot(1,1,1)
            ha_ax.set_title(f"H{a} {l}6563, {D} {l} = "+str(self.ha_wvls[ha_wvl_idx] - np.median(self.ha_wvls))+f"{aa}")
            if frame == "pix":
                ha_ax.imshow(self.ha.data[ha_wvl_idx], cmap="Greys_r", origin="lower")
                ha_ax.set_ylabel("y [pixels]")
                ha_ax.set_xlabel("x [pixels]")
            elif frame == "arcsec":
                ha_tr = self.unit_conversion(self.ha.data.shape[-2:] << u.pix, unit_to="arcsec")
                ha_ax.imshow(self.ha.data[ha_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ha_tr[1], 0, ha_tr[0]])
                ha_ax.set_ylabel("y [arcseconds]")
                ha_ax.set_xlabel("x [arcseconds]")
            elif frame == "megameter":
                ha_tr = self.unit_conversion(self.ha.data.shape[-2:] << u.pix, unit_to="megameter")
                ha_ax.imshow(self.ha.data[ha_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ha_tr[1], 0, ha_tr[0]])
                ha_ax.set_ylabel("y [Mm]")
                ha_ax.set_xlabel("x [Mm]")
            elif frame == "helioprojective":
                ha_tr = self.unit_conversion(self.ha.data.shape[-2:] << u.pix, unit_to="arcsec", centre=True)
                ha_ax.imshow(self.ha.data[ha_wvl_idx], cmap="Greys_r", origin="lower", extent=[0, ha_tr[1], 0, ha_tr[0]])
                ha_ax.set_ylabel("Helioprojective-y [arcsec]")
                ha_ax.set_xlabel("Helioprojective-x [arcsec]")
    def stokes_map(self, stokes, frame, wvl_idx):
        '''
        A class method to plot the Stokes maps of the CRISP data.

        Parameters
        ----------
        stokes : str
            Which Stokes to plot maps for. Can be "all", "I", "Q", "U", "V" or "IV".
        frame : str
            Which frame to plot the maps in. Can be "pix", "arcsec", "megameter" or "helioprojective".
        wvl_idx : int
            Which wavelength to plot the maps in.
        '''

        l = html.unescape("&lambda;")
        a = html.unescape("&alpha;")
        aa = html.unescape("&#8491;")
        D = html.unescape("&Delta;")
        
        if stokes == "all":
            fig = plt.figure()
            I_ax = fig.add_subplot(2,2,1)
            Q_ax = fig.add_subplot(2,2,2)
            U_ax = fig.add_subplot(2,2,3)
            V_ax = fig.add_subplot(2,2,4)
            I_ax.set_title(f"I, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            Q_ax.set_title(f"Q, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            U_ax.set_title(f"U, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            V_ax.set_title(f"V, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            I_ax.tick_params(labelbottom=False)
            Q_ax.tick_params(labelbottom=False, labelleft=False)
            V_ax.tick_params(labelleft=False)
            if frame == "pix":
                I_ax.imshow(self.ca.data[wvl_idx,0], cmap="Greys_r", origin="bottom", vmin=0)
                Q_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max())
                U_ax.imshow(self.ca.data[wvl_idx,2], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max())
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max())
                I_ax.set_ylabel("y [pixels]")
                U_ax.set_ylabel("y [pixels]")
                U_ax.set_xlabel("x [pixels]")
                V_ax.set_xlabel("x [pixels]")
            elif frame == "arcsec":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec")
                I_ax.imshow(self.ca.data[wvl_idx,0], cmap="Greys_r", origin="bottom", vmin=0, extent=[0, tr[1], 0, tr[0]])
                Q_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                U_ax.imshow(self.ca.data[wvl_idx,2], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                I_ax.set_ylabel("y [arcsec]")
                U_ax.set_ylabel("y [arcsec]")
                U_ax.set_xlabel("x [arcsec]")
                V_ax.set_xlabel("x [arcsec]")
            elif frame == "megameter":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="megameter")
                I_ax.imshow(self.ca.data[wvl_idx,0], cmap="Greys_r", origin="bottom", vmin=0, extent=[0, tr[1], 0, tr[0]])
                Q_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                U_ax.imshow(self.ca.data[wvl_idx,2], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                I_ax.set_ylabel("y [Mm]")
                U_ax.set_ylabel("y [Mm]")
                U_ax.set_xlabel("x [Mm]")
                V_ax.set_xlabel("x [Mm]")
            elif frame == "helioprojective":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec", centre=True)
                I_ax.imshow(self.ca.data[wvl_idx,0], cmap="Greys_r", origin="bottom", vmin=0, extent=[0, tr[1], 0, tr[0]])
                Q_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                U_ax.imshow(self.ca.data[wvl_idx,2], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                I_ax.set_ylabel("Helioprojective-y [arcsec]")
                U_ax.set_ylabel("Helioprojective-y [arcsec]")
                U_ax.set_xlabel("Helioprojective-x [arcsec]")
                V_ax.set_xlabel("Helioprojective-x [arcsec]")
        elif stokes == "I":
            self.intensity_map(line="ca", frame=frame, ca_wvl_idx=wvl_idx)
        elif stokes == "Q":
            fig = plt.figure()
            Q_ax = fig.add_subplot(1,1,1)
            Q_ax.set_title(f"Q, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            if frame == "pix":
                Q_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max())
                Q_ax.set_ylabel("y [pixels]")
                Q_ax.set_xlabel("x [pixels]")
            elif frame == "arcsec":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec")
                Q_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                Q_ax.set_ylabel("y [arcsec]")
                Q_ax.set_xlabel("x [arcsec]")
            elif frame == "megameter":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="megameter")
                Q_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                Q_ax.set_ylabel("y [Mm]")
                Q_ax.set_xlabel("x [Mm]")
            elif frame == "helioprojective":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec", centre=True)
                Q_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                Q_ax.set_ylabel("Helioprojective-y [arcsec]")
                Q_ax.set_xlabel("Helioprojective-x [arcsec]")
        elif stokes == "U":
            fig = plt.figure()
            U_ax = fig.add_subplot(1,1,1)
            U_ax.set_title(f"U, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            if frame == "pix":
                U_ax.imshow(self.ca.data[wvl_idx,2], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max())
                U_ax.set_ylabel("y [pixels]")
                U_ax.set_xlabel("x [pixels]")
            elif frame == "arcsec":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec")
                U_ax.imshow(self.ca.data[wvl_idx,2], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                U_ax.set_ylabel("y [arcsec]")
                U_ax.set_xlabel("x [arcsec]")
            elif frame == "megameter":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="megameter")
                U_ax.imshow(self.ca.data[wvl_idx,2], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                U_ax.set_ylabel("y [Mm]")
                U_ax.set_xlabel("x [Mm]")
            elif frame == "helioprojective":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec", centre=True)
                U_ax.imshow(self.ca.data[wvl_idx,2], cmap="Greys_r", origin="bottom", vmin=-0.1*self.ca.data[wvl_idx,0].max(), vmax=0.1*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                U_ax.set_ylabel("Helioprojective-y [arcsec]")
                U_ax.set_xlabel("Helioprojective-x [arcsec]")
        elif stokes == "V":
            fig = plt.figure()
            V_ax = fig.add_subplot(1,1,1)
            V_ax.set_title(f"V, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            if frame == "pix":
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max())
                V_ax.set_ylabel("y [pixels]")
                V_ax.set_xlabel("x [pixels]")
            elif frame == "arcsec":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec")
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                V_ax.set_ylabel("y [arcsec]")
                V_ax.set_xlabel("x [arcsec]")
            elif frame == "megameter":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="megameter")
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                V_ax.set_ylabel("y [Mm]")
                V_ax.set_xlabel("x [Mm]")
            elif frame == "helioprojective":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec", centre=True)
                V_ax.imshow(self.ca.data[wvl_idx,1], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                V_ax.set_ylabel("Helioprojective-y [arcsec]")
                V_ax.set_xlabel("Helioprojective-x [arcsec]")
        elif stokes == "IV":
            fig = plt.figure()
            I_ax = fig.add_subplot(1,2,1)
            V_ax = fig.add_subplot(1,2,2)
            I_ax.set_title(f"I, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            V_ax.set_title(f"V, {D} {l} = "+str(self.ca_wvls[ca_wvl_idx] - np.median(self.ca_wvls))+f"{aa}")
            V_ax.tick_params(labelleft=False)
            if frame == "pix":
                I_ax.imshow(self.ca.data[wvl_idx,0], cmap="Greys_r", origin="bottom", vmin=0)
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max())
                I_ax.set_ylabel("y [pixels]")
                I_ax.set_xlabel("x [pixels]")
                V_ax.set_xlabel("x [pixels]")
            elif frame == "arcsec":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec")
                I_ax.imshow(self.ca.data[wvl_idx,0], cmap="Greys_r", origin="bottom", vmin=0, extent=[0, tr[1], 0, tr[0]])
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                I_ax.set_ylabel("y [arcsec]")
                I_ax.set_xlabel("x [arcsec]")
                V_ax.set_xlabel("x [arcsec]")
            elif frame == "megameter":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="megameter")
                I_ax.imshow(self.ca.data[wvl_idx,0], cmap="Greys_r", origin="bottom", vmin=0, extent=[0, tr[1], 0, tr[0]])
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                I_ax.set_ylabel("y [Mm]")
                I_ax.set_xlabel("x [Mm]")
                V_ax.set_xlabel("x [Mm]")
            elif frame == "helioprojective":
                tr = self.unit_conversion(self.ca.data.shape[-2:], unit_to="arcsec", centre=True)
                I_ax.imshow(self.ca.data[wvl_idx,0], cmap="Greys_r", origin="bottom", vmin=0, extent=[0, tr[1], 0, tr[0]])
                V_ax.imshow(self.ca.data[wvl_idx,3], cmap="Greys_r", origin="bottom", vmin=-0.5*self.ca.data[wvl_idx,0].max(), vmax=0.5*self.ca.data[wvl_idx,0].max(), extent=[0, tr[1], 0, tr[0]])
                I_ax.set_ylabel("Helioprojective-y [arcsec]")
                I_ax.set_xlabel("Helioprojective-x [arcsec]")
                V_ax.set_xlabel("Helioprojective-x [arcsec]")