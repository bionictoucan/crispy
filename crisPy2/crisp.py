import numpy as np
import matplotlib.pyplot as plt
import os, html, yaml, h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import astropy.units as u
from astropy.io.fits.header import Header
from specutils.utils.wcs_utils import vac_to_air
from .mixin import CRISPSlicingMixin, CRISPSequenceSlicingMixin
from .utils import ObjDict, pt_bright
from .io import hdf5_header_to_wcs

class CRISP(CRISPSlicingMixin):
    def __init__(self, filename, wcs=None, uncertainty=None, mask=None, nonu=False):
        if type(filename) == str and ".fits" in filename:
            self.file = fits.open(filename)[0]
        elif type(filename) == str and ".h5" or ".hdf5" in filename:
            f = h5py.File(filename, mode="r")
            self.file = ObjDict({})
            self.file["data"] = f["data"]
            self.file["header"] = yaml.load(f["header"][0], Loader=yaml.Loader)
        elif type(filename) == ObjDict:
            self.file = filename
        else:
            raise NotImplementedError("m8 y?")
        if wcs is None and ".fits" in filename:
            self.wcs = WCS(self.file.header)
        elif wcs is None and ".h5" or ".hdf5" in filename:
            self.wcs = hdf5_header_to_wcs(self.file.header, nonu=nonu)
        else:
            self.wcs = wcs
        self.nonu = nonu
        self.uncertainty = uncertainty
        self.mask = mask
        self.aa = html.unescape("&#8491;")
        self.a = html.unescape("&alpha;")
        self.l = html.unescape("&lambda;")
        self.D = html.unescape("&Delta;")
        self.shape = self.file.data.shape

    def __str__(self):
        if type(self.file.header) == Header:
            time = self.file.header.get("DATE-AVG")[-12:]
            date = self.file.header.get("DATE-AVG")[:-13]
            cl = str(np.round(self.file.header.get("TWAVE1"), decimals=2))
            wwidth = self.file.header.get("WWIDTH1")
            shape = str([self.file.header.get(f"NAXIS{j+1}") for j in reversed(range(self.file.data.ndim))])
            el = self.file.header.get("WDESC1")
            pointing_x = str(self.file.header.get("CRVAL1"))
            pointing_y = str(self.file.header.get("CRVAL2"))
        elif type(self.file.header) == dict:
            time = self.file.header["time-obs"]
            date = self.file.header["date-obs"]
            cl = str(self.file.header["crval"][-3])
            wwidth = str(self.file.header["dimensions"][-3])
            shape = str(self.file.header["dimensions"])
            el = self.file.header["element"]
            pointinig_x = str(self.file.header["crval"][-1])
            pointing_y = str(self.file.header["crval"][-2])

        return f"""CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled: {wwidth}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    def plot_spectrum(self, unit=None, air=False, d=False):
        plt.style.use("ggplot")
        if self.file.data.ndim != 1:
            raise IndexError("If you are using Stokes data please use the plot_stokes method.")

        wavelength = self.wcs.array_index_to_world(np.arange(self.file.data.shape[0])) << u.m #This finds the value of the wavlength axis from the WCS in units of m
        if unit is None:
            wavelength <<= u.Angstrom
        else:
            wavelength <<= unit

        if air:
            wavelength = vac_to_air(wavelength)

        if d:
            wavelength = wavelength - np.median(wavelength)
            xlabel = f"{self.D}{self.l} [{self.aa}]"
        else:
            xlabel = f"{self.l} [{self.aa}]"

        point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0].array_index_to_world(*self.ind[-2:])]
        try:
            datetime = self.file.header["DATE-AVG"]
            el = self.file.header["WDESC1"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
            el = self.file.header["element"]

        fig = plt.figure()
        ax1 = fig.gca()
        ax1.plot(wavelength, self.file.data, c=pt_bright["blue"])
        ax1.set_ylabel("Intensity [DNs]")
        ax1.set_xlabel(xlabel)
        ax1.set_title(f"{datetime} {el}{self.aa} ({point[0]},{point[1]})")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_stokes(self, stokes, unit=None, air=False, d=False):

        point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(*self.ind[-2:])]
        try:
            datetime = self.file.header["DATE-AVG"]
            el = self.file.header["WDESC1"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
            el = self.file.header["element"]

        if self.file.data.ndim == 1:
            wavelength = self.wcs.array_index_to_world(np.arange(self.file.data.shape[0])) << u.m

            if unit is None:
                wavelength <<= u.Angstrom
            else:
                wavelength <<= unit

            if air:
                wavelength = vac_to_air(wavelength)

            if d:
                wavelength = wavelength - np.median(wavelength)
                xlabel = f"{self.D}{self.l} [{self.aa}]"
            else:
                xlabel = f"{self.l} [{self.aa}]"

            fig = plt.figure()
            ax1 = fig.gca()
            ax1.plot(wavelength, self.file.data, c=pt_bright["blue"], marker="o")
            if stokes == "I":
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes I ({point[0]},{point[1]})")
            elif stokes == "Q":
                ax1.set_ylabel("Q [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes Q ({point[0]},{point[1]})")
            elif stokes == "U":
                ax1.set_ylabel("U [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes U ({point[0]},{point[1]})")
            elif stokes == "V":
                ax1.set_ylabel("V [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes V ({point[0]},{point[1]})")
            else:
                raise ValueError("This is not a Stokes.")
            ax1.tick_params(direction="in")
            fig.show()
        elif self.file.data.ndim == 2:
            wavelength = self.wcs.array_index_to_world(np.arange(self.file.data.shape[1])) << u.m

            if unit is None:
                wavelength <<= u.Angstrom
            else:
                wavelength <<= unit

            if air:
                wavelength = vac_to_air(wavelength)

            if d:
                wavelength = wavelength - np.median(wavelength)
                xlabel = f"{self.D}{self.l} [{self.aa}]"
            else:
                xlabel = f"{self.l} [{self.aa}]"

            if stokes == "all":
                fig, ax = plt.subplots(nrows=2, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} All  Stokes ({point[0]},{point[1]})")
                ax[0,0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0,0].set_ylabel("I [DNs]")
                ax[0,0].tick_params(labelbottom=False, direction="in")

                ax[0,1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[0,1].set_ylabel("Q [DNs]")
                ax[0,1].yaxis.set_label_position("right")
                ax[0,1].yaxis.tick_right()
                ax[0,1].tick_params(labelbottom=False, direction="in")

                ax[1,0].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[1,0].set_ylabel("U [DNs]")
                ax[1,0].set_xlabel(xlabel)
                ax[1,0].tick_params(direction="in")

                ax[1,1].plot(wavelength, self.file.data[3], c=pt_bright["blue"], marker="o")
                ax[1,1].set_ylabel("V [DNs]")
                ax[1,1].set_xlabel(xlabel)
                ax[1,1].yaxis.set_label_position("right")
                ax[1,1].yaxis.ticks_right()
                ax[1,1].tick_params(direction="in")
            elif stokes == "IQU":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q, U ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("U [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "QUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, U, V ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IQV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q, V ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, U, V ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IQ":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "IU":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, U ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "IV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, V ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "QU":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, U ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "QV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, V ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "UV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes U, V ({point[0]},{point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("U [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

    def intensity_map(self, frame=None):
        plt.style.use("ggplot")
        
        wvl = np.round(self.wave(self.ind) << u.Angstrom, decimals=2).value
        del_wvl = np.round(wvl - (self.wave(self.wcs.low_level_wcs._wcs.array_shape[0]//2) << u.Angstrom).value, decimals=2)
        try:
            datetime = self.file.header["DATE-AVG"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
        
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
            im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0)
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})")
            fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
            fig.show()
        elif frame == "pix":
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0, origin="lower")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})")
            fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
            fig.show()

    def stokes_map(self, stokes, frame=None):
        plt.style.use("ggplot")

        wvl = np.round(self.wcs.low_level_wcs._wcs[0,:,0,0].array_index_to_world(self.ind[1]) << u.Angstrom, decimals=2).value
        del_wvl = np.round(wvl - (self.wcs.low_level_wcs._wcs[0,:,0,0].array_index_to_world(self.wcs.low_level_wcs._wcs.array_shape[1]//2) << u.Angstrom).value, decimals=2)
        try:
            datetime = self.file.header["DATE-AVG"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
        title = f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l}={del_wvl}{self.aa}"

        if frame is None:
            if self.file.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
                if stokes == "I":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0)
                    ax1.set_title("Stokes I "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title("Stokes Q "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title("Stokes U "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-100, vmax=100)
                    ax1.set_title("Stokes V "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                ax1.tick_params(direction="in")
                fig.show()
            elif self.file.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure()
                    fig.suptitle(title)
                    ax1 = fig.add_subplot(2, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.xaxis.set_label_position("top")
                    ax1.xaxis.tick_top()
                    ax1.set_title("Stokes I ")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(2, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.xaxis.set_label_position("top")
                    ax2.xaxis.tick_top()
                    ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()
                    ax2.set_title("Stokes Q ")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(2, 2, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U ")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="U [DNs]")

                    ax4 = fig.add_subplot(2, 2, 4, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,3))
                    im4 = ax4.imshow(self.file.data[3], cmap="greys_r", vmin=-100, vmax=100)
                    ax4.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax4.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax4.yaxis.set_label_position("right")
                    ax4.yaxis.ticks_right()
                    ax4.set_title("Stokes V ")
                    ax4.tick_params(direction="in")
                    fig.colorbar(im4, ax=ax4, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes U")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="U [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
        elif frame == "pix":
            if self.file.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1) 
                if stokes == "I":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_title("Stokes I "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_title("Stokes Q "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_title("Stokes U "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax1.set_title("Stokes V "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("y [pixels]")
                ax1.set_xlabel("x [pixels]")
                ax1.tick_params(direction="in")
                fig.show()
            elif self.file.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure()
                    fig.suptitle(title)
                    ax1 = fig.add_subplot(2, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.xaxis.set_label_position("top")
                    ax1.xaxis.tick_top()
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(2, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.xaxis.set_label_position("top")
                    ax2.xaxis.tick_top()
                    ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(2, 2, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="greys_r", vmin=-10, vmax=10, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="U [DNs]")

                    ax4 = fig.add_subplot(2, 2, 4)
                    im4 = ax4.imshow(self.file.data[3], cmap="greys_r", vmin=-100, vmax=100, origin="lower")
                    ax4.set_ylabel("y [pixels]")
                    ax4.set_xlabel("x [pixels]")
                    ax4.yaxis.set_label_position("right")
                    ax4.yaxis.ticks_right()
                    ax4.set_title("Stokes V")
                    ax4.tick_params(direction="in")
                    fig.colorbar(im4, ax=ax4, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes U")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="U [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V ")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")

    def wave(self, idx):
        if len(self.wcs.low_level_wcs.array_shape) == 4:
            if hasattr(self, "ind") and type(self.ind[1]) == slice:
                return self.wcs.low_level_wcs._wcs[0,self.ind[1],0,0].array_index_to_world(idx) << u.Angstrom
            elif hasattr(self, "ind") and type(self.ind[1]) != slice:
                return self.wcs.low_level_wcs._wcs[0,:,0,0].array_index_to_world(idx) << u.Angstrom
            else:
                return self.wcs[0,:,0,0].array_index_to_world(idx) << u.Angstrom
        elif len(self.wcs.low_level_wcs.array_shape) == 3:
            if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                if type(self.ind[1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,self.ind[1],0,0].array_index_to_world(idx) << u.Angstrom
                else:
                    return self.wcs.low_level_wcs._wcs[0,:,0,0].array_index_to_world(idx) << u.Angstrom
            else:
                if hasattr(self, "ind") and type(self.ind[0]) == slice:
                    return self.wcs.low_level_wcs._wcs[self.ind[0],0,0].array_index_to_world(idx) << u.Angstrom
                elif hasattr(self, "ind") and type(self.ind[0]) != slice:
                    return self.wcs.low_level_wcs._wcs[:,0,0].array_index_to_world(idx) << u.Angstrom
                else:
                    return self.wcs[:,0,0].array_index_to_world(idx) << u.Angstrom
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            if hasattr(self, "ind"):
                if self.wcs.low_level_wcs._wcs.naxis == 4:
                    return self.wcs.low_level_wcs._wcs[0,:,0,0].array_index_to_world(idx) << u.Angstrom
                elif self.wcs.low_level_wcs._wcs.naxis == 3:
                    return self.wcs.low_level_wcs._wcs[:,0,0].array_index_to_world(idx) << u.Angstrom
                else:
                    raise IndexError("There is no spectral component to your data.")
            else:
                raise IndexError("There is no spectral component to your data.")
        elif len(self.wcs.low_level_wcs.array_shape) == 1:
            print("I'm gonna trust you here.")
            return self.wcs.array_index_to_world(idx) << u.Angstrom
        else:
            raise NotImplementedError("This is way too many dimensions for me to handle.")

    def to_lonlat(self, y, x):
        if len(self.wcs.low_level_wcs.array_shape) == 4:
            if hasattr(self, "ind"):
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2]].array_index_to_world(y,x) << u.arcsec
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,:,self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                else:
                    return self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(y,x) << u.arcsec
            else:
                return self.wcs[0,0].array_index_to_world(y,x) << u.arcsec
        elif len(self.wcs.low_level_wcs.array_shape) == 3:
            if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2]].array_index_to_world(y,x) << u.arcsec
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,:,self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                else:
                    return self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(y,x) << u.arcsec
            else:
                if hasattr(self, "ind"):
                    if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                    elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2]].array_index_to_world(y,x) << u.arcsec
                    elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,:,self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                    else:
                        return self.wcs.low_level_wcs._wcs[0].array_index_to_world(y,x) << u.arcsec
                else:
                    return self.wcs[0].array_index_to_world(y,x) << u.arcsec
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            return self.wcs.array_index_to_world(y,x) << u.arcsec
        else:
            raise NotImplementedError("Too many or too little dimensions.")

    def from_lonlat(self,lon,lat):
        lon, lat = lon << u.arcsec, lat << u.arcsec
        if len(self.wcs.low_level_wcs.array_shape) == 4:
            if hasattr(self, "ind"):
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2],self.ind[-1]].world_to_array_index(lon,lat)
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2]].world_to_array_index(lon,lat)
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,:,self.ind[-1]].world_to_array_index(lon,lat)
                else:
                    return self.wcs.low_level_wcs._wcs[0,0].world_to_array_index(lon,lat)
            else:
                return self.wcs.low_level_wcs._wcs[0,0].world_to_array_index(lon,lat)
        elif len(self.wcs.low_level_wcs.array_shape) == 3:
            if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2],self.ind[-1]].world_to_array_index(lon,lat)
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2]].world_to_array_index(lon,lat)
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,:,self.ind[-1]].world_to_array_index(lon,lat)
                else:
                    return self.wcs.low_level_wcs._wcs[0,0].world_to_array_index(lon,lat)
            else:
                if hasattr(self, "ind"):
                    if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2],self.ind[-1]].world_to_array_index(lon,lat)
                    elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2]].world_to_array_index(lon,lat)
                    elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,:,self.ind[-1]].world_to_array_index(lon,lat)
                    else:
                        return self.wcs.low_level_wcs._wcs[0].world_to_array_index(lon,lat)
                else:
                    return self.wcs.low_level_wcs._wcs[0].world_to_array_index(lon,lat)
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            return self.wcs.world_to_array_index(lon,lat)
        else:
            raise NotImplementedError("Too many or too little dimensions.")

class CRISPSequence(CRISPSequenceSlicingMixin):
    def __init__(self, files):
        self.list = [CRISP(**f) for f in files]

    def __str__(self):
        if type(self.list[0].file.header) == Header:
            time = self.list[0].file.header.get("DATE-AVG")[-12:]
            date = self.list[0].file.header.get("DATE-AVG")[:-13]
            cl = [str(np.round(f.file.header.get("TWAVE1"), decimals=2)) for f in self.list]
            wwidth = [f.file.header.get("WWIDTH1") for f in self.list]
            shape = [str([f.file.header.get(f"NAXIS{j+1}") for j in reversed(range(f.file.data.ndim))]) for f in self.list]
            el = [f.file.header.get("WDESC1") for f in self.list]
            pointing_x = str(self.list[0].file.header.get("CRVAL1"))
            pointing_y = str(self.list[0].file.header.get("CRVAL2"))
        elif type(self.list[0].file.header) == dict:
            time = self.list[0].file.header["time-obs"]
            date = self.list[0].file.header["date-obs"]
            cl = [str(f.file.header["crval"][-3]) for f in self.list]
            wwidth = [str(f.file.header["dimensions"][-3]) for f in self.list]
            shape = [str(f.file.header["dimensions"]) for f in self.list]
            el = [f.file.header["element"] for f in self.list]
            pointing_x = str(self.list[0].file.header["crval"][-1])
            pointing_y = str(self.list[0].file.header["crval"][-2])

        return f"""CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled: {wwidth}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    def plot_spectrum(self, idx, unit=None, air=False, d=False):
        if idx != "all":
            self.list[idx].plot_spectrum(unit=unit, air=air, d=d)
        else:
            for f in self.list:
                f.plot_spectrum(unit=unit, air=air, d=d)

    def plot_stokes(self, idx, stokes, unit=None, air=False, d=False):
        if idx != "all":
            self.list[idx].plot_stokes(stokes, unit=unit, air=air, d=d)
        else:
            for f in self.list:
                f.plot_stokes(stokes, unit=unit, air=air, d=d)

    def intensity_map(self, idx, frame=None):
        if idx != "all":
            self.list[idx].intensity_map(frame=frame)
        else:
            for f in self.list:
                f.intensity_map(frame=frame)

    def stokes_map(self, idx, stokes, frame=None):
        if idx != "all":
            self.list[idx].stokes_map(stokes, frame=frame)
        else:
            for f in self.list:
                f.stokes_map(stokes, frame=frame)

class CRISPWideband(CRISP):
    def __str__(self):
        if type(self.file.header) == Header:
            time = self.file.header.get("DATE-AVG")[-12:]
            date = self.file.header.get("DATE-AVG")[:-13]
            shape = str([self.file.header.get(f"NAXIS{j+1}") for j in reversed(range(self.file.data.ndim))])
            el = self.file.header.get("WDESC1")
            pointing_x = str(self.file.header.get("CRVAL1"))
            pointing_y = str(self.file.header.get("CRVAL2"))
        elif type(self.file.header) == dict:
            time = self.file.header["time-obs"]
            date = self.file.header["date-obs"]
            shape = str(self.file.header["dimensions"])
            el = self.file.header["element"]
            pointing_x = str(self.file.header["crval"][-1])
            pointing_y = str(self.file.header["crval"][-2])

        return f"""CRISP Wideband Context Image
        ------------------
        {date} {time}

        Observed: {el}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    def intensity_map(self, frame=None):
        plt.style.use("ggplot")
        try:
            datetime = self.file.header["DATE-AVG"]
            el = self.file.header["WDESC1"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
            el = self.file.header["element"]

        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs)
            im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0)
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"{datetime} {el} {self.aa}")
            fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
            fig.show()
        elif frame == "pix":
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0, origin="lower")
            ax1.set_ylabel("y [arcsec]")
            ax1.set_xlabel("x [arcsec]")
            ax1.set_title(f"{datetime} {el} {self.aa}")
            fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
            fig.show()

class CRISPWidebandSequence(CRISPSequence):
    def __init__(self, files):
        self.list = [CRISPWideband(**f) for f in files]

    def __str__(self):
        if type(self.list[0].file.header) == Header:
            time = [f.file.header.get("DATE-AVG")[-12:] for f in self.list]
            date = self.list[0].file.header.get("DATE-AVG")[:-13]
            shape = [str([f.file.header.get(f"NAXIS{j+1}") for j in reversed(range(f.file.data.ndim))]) for f in self.list]
            el = [f.file.header.get("WDESC1") for f in self.list]
            pointing_x = str(self.list[0].file.header.get("CRVAL1"))
            pointing_y = str(self.list[0].file.header.get("CRVAL2"))
        elif type(self.list[0].file.header) == dict:
            time = [f.file.header["time-obs"] for f in self.list]
            data = self.list[0].file.header["date-obs"]
            shape = [str(f.file.header["dimensions"]) for f in self.list]
            el = [self.list[0].header["element"] for f in self.list]
            pointing_x = str(self.list[0].file.header["crval"][-1])
            pointing_y = str(self.list[0].file.header["crval"][-2])

        return f"""CRISP Wideband Context Image
        ------------------
        {date} {time}

        Observed: {el}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}"""

class CRISPNonU(CRISP):
    def __init__(self, filename, wcs=None, uncertainty=None, mask=None, nonu=True):
        super().__init__(filename=filename, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)

        if ".fits" in filename:
            self.wvls = fits.open(filename)[1].data #This assumes that the true wavelength points are stored in the first HDU of the FITS file as a numpy array
        else:
            self.wvls = self.file.header["spect_pos"]

    def __str__(self):
        if type(self.file.header) == Header:
            time = self.file.header.get("DATE-AVG")[-12:]
            date = self.file.header.get("DATE-AVG")[:-13]
            cl = str(np.round(self.file.header.get("TWAVE1"), decimals=2))
            wwidth = self.file.header.get("WWIDTH1")
            shape = str([self.file.header.get(f"NAXIS{j+1}") for j in reversed(range(self.file.data.ndim))])
            el = self.file.header.get("WDESC1")
            pointing_x = str(self.file.header.get("CRVAL1"))
            pointing_y = str(self.file.header.get("CRVAL2"))
        elif type(self.file.header) == dict:
            time = self.file.header["time-obs"]
            date = self.file.header["date-obs"]
            cl = str(self.file.header["crval"][-3])
            wwidth = self.file.header["dimensions"][-3]
            shape = str(self.file.header["dimensions"])
            el = self.file.header["element"]
            pointing_x = str(self.file.header["crval"][-1])
            pointing_y = str(self.file.header["crval"][-2])
        sampled_wvls = str(self.wvls)

        return f"""CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled: {wwidth}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}
        Wavelengths sampled: {sampled_wvls}"""

    def plot_spectrum(self, unit=None, air=False, d=False):
        plt.style.use("ggplot")
        if self.file.data.ndim != 1:
            raise IndexError("If you are using Stokes data please use the plot_stokes method.")

        wavelength = self.wvls
        if unit is None:
            wavelength <<= u.Angstrom
        else:
            wavelength <<= unit

        if air:
            wavelength = vac_to_air(wavelength)

        if d:
            wavelength = wavelength - np.median(wavelength)
            xlabel = f"{self.D}{self.l} [{self.aa}]"
        else:
            xlabel = f"{self.l} [{self.aa}]"

        point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0].array_index_to_world(*self.ind[-2:])]
        try:
            datetime = self.file.header["DATE-AVG"]
            el = self.file.header["WDESC1"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
            el = self.file.header["element"]

        fig = plt.figure()
        ax1 = fig.gca()
        ax1.plot(wavelength, self.file.data, c=pt_bright["blue"], marker="o")
        ax1.set_ylabel("Intensity [DNs]")
        ax1.set_xlabel(xlabel)
        ax1.set_title(f"{datetime} {el} {self.aa} ({point[0]}, {point[1]})")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_stokes(self, stokes, unit=None, air=False, d=False):

        point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(*self.ind[-2:])]
        try:
            datetime = self.file.header["DATE-AVG"]
            el = self.file.header["WDESC1"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
            el = self.file.header["element"]

        if self.file.data.ndim == 1:
            wavelength = self.wvls

            if unit is None:
                wavelength <<= u.Angstrom
            else:
                wavelength <<= unit

            if air:
                wavelength = vac_to_air(wavelength)

            if d:
                wavelength = wavelength - np.median(wavelength)
                xlabel = f"{self.D}{self.l} [{self.aa}]"
            else:
                xlabel = f"{self.l} [{self.aa}]"

            fig = plt.figure()
            ax1 = fig.gca()
            ax1.plot(wavelength, self.file.data, c=pt_bright["blue"], marker="o")
            if stokes == "I":
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes I ({point[0]}, {point[1]})")
            elif stokes == "Q":
                ax1.set_ylabel("Q [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes Q ({point[0]}, {point[1]})")
            elif stokes == "U":
                ax1.set_ylabel("U [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes U ({point[0]}, {point[1]})")
            elif stokes == "V":
                ax1.set_ylabel("V [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes V ({point[0]}, {point[1]})")
            else:
                raise ValueError("This is not a Stokes.")
            ax1.tick_params(direction="in")
            fig.show()
        elif self.file.data.ndim == 2:
            wavelength = self.wvls

            if unit is None:
                wavelength <<= u.Angstrom
            else:
                wavelength <<= unit

            if air:
                wavelength = vac_to_air(wavelength)

            if d:
                wavelength = wavelength - np.median(wavelength)
                xlabel = f"{self.D}{self.l} [{self.aa}]"
            else:
                xlabel = f"{self.l} [{self.aa}]"

            if stokes == "all":
                fig, ax = plt.subplots(nrows=2, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} All Stokes ({point[0]}, {point[1]})")
                ax[0,0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0,0].set_ylabel("I [DNs]")
                ax[0,0].tick_params(labelbottom=False, direction="in")

                ax[0,1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[0,1].set_ylabel("Q [DNs]")
                ax[0,1].yaxis.set_label_position("right")
                ax[0,1].yaxis.tick_right()
                ax[0,1].tick_params(labelbottom=False, direction="in")

                ax[1,0].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[1,0].set_ylabel("U [DNs]")
                ax[1,0].set_xlabel(xlabel)
                ax[1,0].tick_params(direction="in")

                ax[1,1].plot(wavelength, self.file.data[3], c=pt_bright["blue"], marker="o")
                ax[1,1].set_ylabel("V [DNs]")
                ax[1,1].set_xlabel(xlabel)
                ax[1,1].yaxis.set_label_position("right")
                ax[1,1].yaxis.ticks_right()
                ax[1,1].tick_params(direction="in")
            elif stokes == "IQU":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q, U ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("U [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "QUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, U, V ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IQV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q, V ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, U, V ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IQ":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "IU":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, U ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "IV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, V ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "QU":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, U ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "QV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, V ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "UV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes U, V ({point[0]}, {point[1]})")

                ax[0].plot(wavelength, self.file.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("U [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

    def intensity_map(self, frame=None):
        plt.style.use("ggplot")
        
        wvl = np.round(self.wvls[self.ind], decimals=2)
        del_wvl = np.round(wvl - np.median(self.wvls), decimals=2)
        try:
            datetime = self.file.header["DATE-AVG"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
        
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
            im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0)
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})")
            fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
            fig.show()
        elif frame == "pix":
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0, origin="lower")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})")
            fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
            fig.show()

    def stokes_map(self, stokes, frame=None):
        plt.style.use("ggplot")

        wvl = np.round(self.wvls[self.ind], decimals=2)
        del_wvl = np.round(wvl - np.median(self.wvls), decimals=2)
        try:
            datetime = self.file.header["DATE-AVG"]
        except KeyError:
            datetime = self.file.header["date-obs"] + "T" + self.file.header["time-obs"]
        title = f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l}={del_wvl}{self.aa}"

        if frame is None:
            if self.file.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
                if stokes == "I":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0)
                    ax1.set_title("Stokes I "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title("Stokes Q "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title("Stokes U "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-100, vmax=100)
                    ax1.set_title("Stokes V "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                ax1.tick_params(direction="in")
                fig.show()
            elif self.file.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure()
                    fig.suptitle(title)
                    ax1 = fig.add_subplot(2, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.xaxis.set_label_position("top")
                    ax1.xaxis.tick_top()
                    ax1.set_title("Stokes I ")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(2, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.xaxis.set_label_position("top")
                    ax2.xaxis.tick_top()
                    ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()
                    ax2.set_title("Stokes Q ")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(2, 2, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U ")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="U [DNs]")

                    ax4 = fig.add_subplot(2, 2, 4, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,3))
                    im4 = ax4.imshow(self.file.data[3], cmap="greys_r", vmin=-100, vmax=100)
                    ax4.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax4.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax4.yaxis.set_label_position("right")
                    ax4.yaxis.ticks_right()
                    ax4.set_title("Stokes V ")
                    ax4.tick_params(direction="in")
                    fig.colorbar(im4, ax=ax4, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes U")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="U [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
        elif frame == "pix":
            if self.file.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1) 
                if stokes == "I":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_title("Stokes I "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_title("Stokes Q "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_title("Stokes U "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(self.file.data, cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax1.set_title("Stokes V "+title)
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("y [pixels]")
                ax1.set_xlabel("x [pixels]")
                ax1.tick_params(direction="in")
                fig.show()
            elif self.file.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure()
                    fig.suptitle(title)
                    ax1 = fig.add_subplot(2, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.xaxis.set_label_position("top")
                    ax1.xaxis.tick_top()
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(2, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.xaxis.set_label_position("top")
                    ax2.xaxis.tick_top()
                    ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(2, 2, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="greys_r", vmin=-10, vmax=10, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="U [DNs]")

                    ax4 = fig.add_subplot(2, 2, 4)
                    im4 = ax4.imshow(self.file.data[3], cmap="greys_r", vmin=-100, vmax=100, origin="lower")
                    ax4.set_ylabel("y [pixels]")
                    ax4.set_xlabel("x [pixels]")
                    ax4.yaxis.set_label_position("right")
                    ax4.yaxis.ticks_right()
                    ax4.set_title("Stokes V")
                    ax4.tick_params(direction="in")
                    fig.colorbar(im4, ax=ax4, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                    fig.colorbar(im3, ax=ax3, orientation="horizontal", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes U")
                    ax1.tick_params(direction="in")
                    fig.colorbar(im1, ax=ax1, orientation="horizontal", label="U [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V ")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="horizontal", label="V [DNs]")

    def wave(self, idx):
        return self.wvls[idx]

class CRISPNonUSequence(CRISPSequence):
    def __init__(self, files):
        self.list = [CRISPNonU(**f) for f in files]

    def __str__(self):
        if type(self.list[0].file.header) == Header:
            time = self.list[0].file.header.get("DATE-AVG")[-12:]
            date = self.list[0].file.header.get("DATE-AVG")[:-13]
            cl = [str(np.round(f.file.header.get("TWAVE1"), decimals=2)) for f in self.list]
            wwidth = [f.file.header.get("WWIDTH1") for f in self.list]
            shape = [str([f.file.header.get(f"NAXIS{j+1}") for j in reversed(range(f.file.data.ndim))]) for f in self.list]
            el = [f.file.header.get("WDESC1") for f in self.list]
            pointing_x = str(self.list[0].file.header.get("CRVAL1"))
            pointing_y = str(self.list[0].file.header.get("CRVAL2"))
        elif type(self.list[0].file.header) == dict:
            time = self.list[0].file.header["time-obs"]
            date = self.list[0].file.header["date-obs"]
            cl = [str(f.file.header["crval"][-3]) for f in self.list]
            wwidth = [str(f.file.header["dimensions"][-3]) for f in self.list]
            shape = [str(f.file.header["dimensions"]) for f in self.list]
            el = [f.file.header["element"] for f in self.list]
            pointing_x = str(self.list[0].file.header["crval"][-1])
            pointing_y = str(self.list[0].file.header["crval"][-2])
        sampled_wvls = [f.wvls for f in self.list]

        return f"""CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled: {wwidth}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}
        Sampled wavlengths: {sampled_wvls}"""