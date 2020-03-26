import numpy as np
import matplotlib.pyplot as plt
import os, h5py, yaml, html
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from specutils.utils.wcs_utils import vac_to_air
from .mixin import CRISPSlicingMixin, CRISPSequenceSlicingMixin

class CRISP(CRISPSlicingMixin):
    def __init__(self, file, wcs=None, uncertainty=None, mask=None):
        if type(file) == str:
            self.file = fits.open(file)[0]
        elif type(file) == ObjDict:
            self.file = file
        else:
            raise NotImplementedError("m8 y?")
        if wcs is None:
            self.wcs = WCS(self.file.header)
        else:
            self.wcs = wcs
        self.uncertainty = uncertainty
        self.mask = mask
        self.aa = html.unescape("&#8491;")
        self.a = html.unescape("&alpha;")
        self.l = html.unescape("&lambda;")
        self.D = html.unescape("&Delta;")

    def __str__(self):
        time = self.file.header.get("DATE-AVG")[-12:]
        date = self.file.header.get("DATE-AVG")[:-13]
        cl = str(np.round(self.file.header.get("TWAVE1"), decimals=2))
        wwidth = self.file.header.get("WWIDTH1")
        shape = str([self.file.header.get(f"NAXIS{j+1}") for j in reversed(range(self.file.data.ndim))])
        el = self.file.header.get("WDESC1")
        pointing_x = str(self.file.header.get("CRVAL1"))
        pointing_y = str(self.file.header.get("CRVAL2"))

        return f"""CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled: {wwidth}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    def __repr__(self):
        return self.__str__()

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

        fig = plt.figure()
        ax1 = fig.gca()
        ax1.plot(wavelength, self.file.data)
        ax1.set_ylabel("Intensity [DNs]")
        ax1.set_xlabel(xlabel)
        ax1.set_title(self.file.header.get("WDESC1"))
        ax1.tick_params(direction="in")
        fig.show()

    def plot_stokes(self, stokes, unit=None, air=False, d=False):

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
            ax1.plot(wavelength, self.file.data)
            if stokes == "I":
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes I")
            elif stokes == "Q":
                ax1.set_ylabel("Q [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes Q")
            elif stokes == "U":
                ax1.set_ylabel("U [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes U")
            elif stokes == "V":
                ax1.set_ylabel("V [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes V")
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
                ax[0,0].plot(wavelength, self.file.data[0])
                ax[0,0].set_ylabel("I [DNs]")
                ax[0,0].tick_params(labelbottom=False, direction="in")

                ax[0,1].plot(wavelength, self.file.data[1])
                ax[0,1].set_ylabel("Q [DNs]")
                ax[0,1].yaxis.set_label_position("right")
                ax[0,1].yaxis.tick_right()
                ax[0,1].tick_params(labelbottom=False, direction="in")

                ax[1,0].plot(wavelength, self.file.data[2])
                ax[1,0].set_ylabel("U [DNs]")
                ax[1,0].set_xlabel(xlabel)
                ax[1,0].tick_params(direction="in")

                ax[1,1].plot(wavelength, self.file.data[3])
                ax[1,1].set_ylabel("V [DNs]")
                ax[1,1].set_xlabel(xlabel)
                ax[1,1].yaxis.set_label_position("right")
                ax[1,1].yaxis.ticks_right()
                ax[1,1].tick_params(direction="in")
            elif stokes == "IQU":
                fig, ax = plt.subplots(nrows=1, ncols=3)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2])
                ax[2].set_ylabel("U [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "QUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2])
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IQV":
                fig, ax = plt.subplots(nrows=1, ncols=3)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2])
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.file.data[2])
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                ax[2].tick_params(direction="in")
            elif stokes == "IQ":
                fig, ax = plt.subplots(nrows=1, ncols=2)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "IU":
                fig, ax = plt.subplots(nrows=1, ncols=2)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "IV":
                fig, ax = plt.subplots(nrows=1, ncols=2)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "QU":
                fig, ax = plt.subplots(nrows=1, ncols=2)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "QV":
                fig, ax = plt.subplots(nrows=1, ncols=2)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")
            elif stokes == "UV":
                fig, ax = plt.subplots(nrows=1, ncols=2)

                ax[0].plot(wavelength, self.file.data[0])
                ax[0].set_ylabel("U [DNs]")
                ax[0].set_xlabel(xlabel)
                ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.file.data[1])
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                ax[1].tick_params(direction="in")

    def intensity_map(self, frame=None):
        #TODO: Add title including the wavelength of the observation or deltalambda.
        plt.style.use("ggplot")
        
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
            ax1.imshow(self.file.data, cmap="Greys_r", vmin=0)
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.tick_params(direction="in")
            fig.show()
        elif frame == "pix":
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.imshow(self.file.data, cmap="Greys_r", vmin=0, origin="lower")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.tick_params(direction="in")
            fig.show()

    def stokes_map(self, stokes, frame=None):
        plt.style.use("ggplot")

        if frame is None:
            if self.file.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
                if stokes == "I":
                    ax1.imshow(self.file.data, cmap="Greys_r", vmin=0)
                    ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes I")
                elif stokes == "Q":
                    ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes Q")
                elif stokes == "U":
                    ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes U")
                elif stokes == "V":
                    ax1.imshow(self.file.data, cmap="Greys_r", vmin=-100, vmax=100)
                    ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes V")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                ax1.tick_params(direction="in")
                fig.show()
            elif self.file.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure()
                    ax1 = fig.add_subplot(2, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.xaxis.set_label_position("top")
                    ax1.xaxis.tick_top()
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(2, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.xaxis.set_label_position("top")
                    ax2.xaxis.tick_top()
                    ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(2, 2, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    ax3.imshow(self.file.data[2], cmap="greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")

                    ax4 = fig.add_subplot(2, 2, 4, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,3))
                    ax4.imshow(self.file.data[3], cmap="greys_r", vmin=-100, vmax=100)
                    ax4.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax4.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax4.yaxis.set_label_position("right")
                    ax4.yaxis.ticks_right()
                    ax4.set_title("Stokes V")
                    ax4.tick_params(direction="in")
                elif stokes == "IQU":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")
                elif stokes == "QUV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                elif stokes == "IQV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                elif stokes == "IUV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,2))
                    ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                elif stokes == "IQ":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                elif stokes == "IU":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                elif stokes == "IV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                elif stokes == "QU":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                elif stokes == "QV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                elif stokes == "UV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,0))
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes U")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs,1))
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
        elif frame == "pix":
            if self.file.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1) 
                if stokes == "I":
                    ax1.imshow(self.file.data, cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes I")
                elif stokes == "Q":
                    ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes Q")
                elif stokes == "U":
                    ax1.imshow(self.file.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes U")
                elif stokes == "V":
                    ax1.imshow(self.file.data, cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax1.set_title(self.file.header.get("WDESC1")+f"{self.aa} Stokes V")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("y [pixels]")
                ax1.set_xlabel("x [pixels]")
                ax1.tick_params(direction="in")
                fig.show()
            elif self.file.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure()
                    ax1 = fig.add_subplot(2, 2, 1)
                    ax1.imshow(self.file.data[0], cmap="greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.xaxis.set_label_position("top")
                    ax1.xaxis.tick_top()
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(2, 2, 2)
                    ax2.imshow(self.file.data[1], cmap="greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.xaxis.set_label_position("top")
                    ax2.xaxis.tick_top()
                    ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(2, 2, 3)
                    ax3.imshow(self.file.data[2], cmap="greys_r", vmin=-10, vmax=10, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")

                    ax4 = fig.add_subplot(2, 2, 4)
                    ax4.imshow(self.file.data[3], cmap="greys_r", vmin=-100, vmax=100, origin="lower")
                    ax4.set_ylabel("y [pixels]")
                    ax4.set_xlabel("x [pixels]")
                    ax4.yaxis.set_label_position("right")
                    ax4.yaxis.ticks_right()
                    ax4.set_title("Stokes V")
                    ax4.tick_params(direction="in")
                elif stokes == "IQU":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(direction="in")
                elif stokes == "QUV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                elif stokes == "IQV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                elif stokes == "IUV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")

                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.imshow(self.file.data[2], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(direction="in")
                elif stokes == "IQ":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                elif stokes == "IU":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                elif stokes == "IV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=0, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                elif stokes == "QU":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(direction="in")
                elif stokes == "QV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")
                elif stokes == "UV":
                    fig = plt.figure()

                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(self.file.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes U")
                    ax1.tick_params(direction="in")

                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(self.file.data[1], cmap="Greys_r", vmin=-100, vmax=100, origin="lower")
                    ax2.set_ylabel("y [pixels]")
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(direction="in")

class CRISPSequence(CRISPSequenceSlicingMixin):
    def __init__(self, list):
        self.list = [CRISP(**f) for f in list]

    def __str__(self):
        time = self.list[0].file.header.get("DATE-AVG")[-12:]
        date = self.list[0].file.header.get("DATE-AVG")[:-13]
        cl = [str(np.round(f.file.header.get("TWAVE1"), decimals=2)) for f in self.list]
        wwidth = [f.file.header.get("WWIDTH1") for f in self.list]
        shape = [str([f.file.header.get(f"NAXIS{j+1}") for j in reversed(range(f.file.data.ndim))]) for f in self.list]
        el = [f.file.header.get("WDESC1") for f in self.list]
        pointing_x = str(self.list[0].file.header.get("CRVAL1"))
        pointing_y = str(self.list[0].file.header.get("CRVAL2"))

        return f"""CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled: {wwidth}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    def __repr__(self):
        return self.__str__()

    def plot_spectrum(self, unit=None, air=False, d=False):
        for f in self.list:
            f.plot_spectrum(unit=unit, air=air, d=d)

    def intensity_map(self, frame=None):
        #TODO: Add index argument to choose which list element to use.
        for f in self.list:
            f.intensity_map(frame=frame)

def hdf5_header_to_wcs(header):
    if type(header) != dict:
        header = yaml.load(header, Loader=yaml.Loader)

    wcs = WCS(naxis=len(header["dimensions"]))
    wcs.wcs.crpix = header["crpix"]
    wcs.wcs.naxis = header["dimensions"]