import numpy as np
import matplotlib.pyplot as plt
import html, zarr
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from specutils.utils.wcs_utils import vac_to_air
from sunpy.coordinates import Helioprojective
import matplotlib
from typing import Union, Sequence, List, Dict, Optional, Tuple
from .mixin import CRISPSlicingMixin, CRISPSequenceSlicingMixin
from .utils import (
    ObjDict,
    pt_bright,
    rotate_crop_data,
    rotate_crop_aligned_data,
    reconstruct_full_frame,
    parameter_docstring,
)
from .io import zarr_header_to_wcs

rc_context_dict = {
    # "figure.constrained_layout.use" : True,
    # "figure.autolayout" : True,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "image.origin": "lower",
    "figure.figsize": (10, 6),
    # "image.aspect" : "auto"
    "font.size": 11,
    "font.serif": "New Century Schoolbook",
}


class CRISP(CRISPSlicingMixin):
    """
    Class for a single narrowband CRISP observation. This object is intended to
    be for narrowband observations of a single spectral line. This can be sliced
    directly by virtue of inheriting from `astropy`'s `N-dimensional data
    slicing <https://docs.astropy.org/en/stable/nddata/>`_.

    Parameters
    ----------
    filename : str or ObjDict
        The file to be represented by the class. This can be in the form of a
        fits file or zarr file or an ObjDict object (see ``crispy.utils`` for
        more information on ObjDicts). For fits files, the imaging
        spectroscopy/spectropolarimetry is assumed to be in the PrimaryHDU of
        the fits file. For zarr file it is assumed to have an array called
        "data" in the top path of the zarr directory.
    wcs : astropy.wcs.WCS or None, optional
        Defines the World Coordinate System (WCS) of the observation. If
        ``None``, the WCS is constructed from the header information in the
        file. If a WCS is provided then it will be used by the class instead.
        Default is None.
    uncertainty : numpy.ndarray or None, optional
        The uncertainty in the observable. Default is None.
    mask : numpy.ndarray or None, optional
        The mask to be applied to the data. Default is None.
    nonu : bool, optional
        Whether or not the :math:`\\Delta \\lambda` on the wavelength axis is
        uniform. This is helpful when constructing the WCS but if True, then the
        ``CRISPNonU`` class should be used. Default is False.
    """

    def __init__(
        self,
        filename: Union[str, ObjDict],
        wcs: Optional[WCS] = None,
        uncertainty: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        nonu: bool = False,
    ) -> None:
        if isinstance(filename, str) and ".fits" in filename:
            self.file = fits.open(filename)[0]
        elif isinstance(filename, str) and ".zarr" in filename:
            f = zarr.open(filename, mode="r")
            self.file = ObjDict({})
            self.file["data"] = f["data"]
            self.file["header"] = f["data"].attrs
        elif isinstance(filename, ObjDict):
            self.file = filename
        else:
            raise NotImplementedError("m8 y?")
        if wcs is None and ".fits" in filename:
            self.wcs = WCS(self.file.header)
        elif wcs is None and ".zarr" in filename:
            self.wcs = zarr_header_to_wcs(self.header, nonu=nonu)
        else:
            self.wcs = wcs
        self.nonu = nonu
        self.uncertainty = uncertainty
        self.mask = mask
        self.aa = html.unescape("&#8491;")
        self.a = html.unescape("&alpha;")
        self.l = html.unescape("&lambda;")
        self.D = html.unescape("&Delta;")
        if all(
            x in self.header.keys()
            for x in ["frame_dims", "x_min", "x_max", "y_min", "y_max", "angle"]
        ):
            self.rotate = True
        else:
            self.rotate = False

    def __str__(self) -> str:
        try:
            time = self.header["DATE-AVG"][-12:]
            date = self.header["DATE-AVG"][:-13]
            cl = str(np.round(self.header["TWAVE1"], decimals=2))
            wwidth = self.header["WWIDTH1"]
            shape = str(
                [self.header[f"NAXIS{j+1}"] for j in reversed(range(self.data.ndim))]
            )
            el = self.header["WDESC1"]
            pointing_x = str(self.header["CRVAL1"])
            pointing_y = str(self.header["CRVAL2"])
        except KeyError:
            time = self.header["time_obs"]
            date = self.header["date_obs"]
            cl = str(self.header["crval"][-3])
            wwidth = str(self.header["dimensions"][-3])
            shape = str(self.header["dimensions"])
            el = self.header["element"]
            pointing_x = str(self.header["crval"][-1])
            pointing_y = str(self.header["crval"][-2])

        return f"""
        CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength [{self.aa}]: {cl}
        Wavelengths sampled: {wwidth}
        Pointing [arcsec] (HPLN, HPLT): ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    @property
    def data(self) -> np.ndarray:
        """
        The actual data.
        """
        return self.file.data

    @property
    def header(self) -> Dict:
        """
        The metainformation about the observations.
        """
        return dict(self.file.header)

    @property
    def shape(self) -> Tuple:
        """
        The dimensions of the data.
        """
        return self.data.shape

    @property
    def wvls(self) -> np.ndarray:
        """
        The wavelengths sampled in the observation.
        """
        return self.wave(np.arange(self.shape[-3]))

    @property
    def info(self) -> str:
        """
        Information about the observation.
        """
        return self.__str__()

    @property
    def time(self) -> str:
        """
        The time of the observation in UTC.
        """
        try:
            return self.header["DATE-AVG"][-12:]
        except KeyError:
            return self.header["time_obs"]

    @property
    def date(self) -> str:
        """
        The date of the observation.
        """
        try:
            return self.header["DATE-AVG"][:-13]
        except KeyError:
            return self.header["date_obs"]

    def rotate_crop(self, sep: bool = False) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        For an image containing the data as a rotated subframe this method
        returns the data after rotation and cropping in addition to the
        metadata required to reconstruct the full frame (excluding a small
        border that is removed during refinement of the data corners).

        Parameters
        ----------
        sep : bool, optional
            Whether or not to return just the rotated array i.e. if False then
            ```self.data```is replaced with the rotated object and the full
            frame is moved to ```self.full_frame``` else the rotated data is
            returned as a numpy array. Default is False.

        Returns
        -------
        crop : numpy.ndarray
            3 or 4D array containing the rotated and cropped data from the image.
        cropData : dict
            Dictionary containing the metadata necessary to reconstruct these
            cropped images into their full-frame input using
            utils.reconstruct_full_frame (excluding the border lost to the
            crop).
        """

        if sep:
            return rotate_crop_data(self.data)

        self.full_frame = self.data
        crop, crop_dict = rotate_crop_data(self.data)
        self.file.header["frame_dims"] = crop_dict["frameDims"]
        self.file.header["x_min"] = crop_dict["xMin"]
        self.file.header["x_max"] = crop_dict["xMax"]
        self.file.header["y_min"] = crop_dict["yMin"]
        self.file.header["y_max"] = crop_dict["yMax"]
        self.file.header["angle"] = crop_dict["angle"]
        self.file.data = crop
        self.rotate = True
        return None

    def reconstruct_full_frame(self, sep: bool = False) -> Optional[np.ndarray]:
        """
        If the image has been rotated (which would take it out of its WCS) then
        this method can be used to reconstruct the image in its original WCS
        frame.

        Parameters
        ----------
        sep : bool, optional
            Whether or not to return just the array of the full frame i.e. if
            False, then ```self.data``` is replaced with the full frame object
            and the rotated is moved to ```self.rot_data``` else the full frame
            is return as a numpy array. Default is False.

        Returns
        -------
        rotatedIm : numpy.ndarray
            A derotated, full frame, copy of the input image cube.
        """

        assert "frame_dims" in self.header
        crop_dict = {
            "frameDims": self.header["frame_dims"],
            "xMin": self.header["x_min"],
            "xMax": self.header["x_max"],
            "yMin": self.header["y_min"],
            "yMax": self.header["y_max"],
            "angle": self.header["angle"],
        }

        if sep:
            return reconstruct_full_frame(crop_dict, self.data)

        self.rot_data = self.data
        self.file.data = reconstruct_full_frame(crop_dict, self.data)
        self.rotate = False
        return None

    @plt.rc_context(rc_context_dict)
    def plot_spectrum(
        self, unit: Optional[u.Unit] = None, air: bool = False, d: bool = False
    ) -> None:
        """
        Plots the intensity spectrum for a specified coordinate by slicing.

        Parameters
        ----------
        unit : astropy.units.Unit or None, optional
            The unit to have the wavelength axis in. Default is None which changes the units to Angstrom.
        air : bool, optional
            Whether or not to convert the wavelength axis to air wavelength (if
            it is not already been converted). e.g. for the Ca II 8542  spectral
            line, 8542 is the rest wavelength of the spectral line measured in
            air. It is possible that the header data (and by proxy the WCS) will
            have the value of the rest wavelength in vacuum (which in this case
            is 8544). Default is False.
        d : bool, optional
            Converts the wavelength axis to :math:`\\Delta \\lambda`. Default is False.
        """
        if self.data.ndim != 1:
            raise IndexError(
                "If you are using Stokes data please use the plot_stokes method."
            )

        wavelength = self.wave(
            np.arange(self.data.shape[0])
        )  # This finds the value of the wavlength axis from the WCS in units of m
        if unit != None:
            wavelength <<= unit

        if air:
            wavelength = vac_to_air(wavelength)

        if d:
            wavelength = wavelength - np.median(wavelength)
            xlabel = f"{self.D}{self.l} [{self.aa}]"
        else:
            xlabel = f"{self.l} [{self.aa}]"

        # coord = self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(*self.ind[-2:])
        # lon, lat = np.round(coord.Tx, decimals=2), np.round(coord.Ty, decimals=2)
        try:
            datetime = self.header["DATE-AVG"]
            el = self.header["WDESC1"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
            el = self.header["element"]

        fig = plt.figure()
        ax1 = fig.gca()
        ax1.plot(wavelength, self.data, c=pt_bright["blue"])
        ax1.set_ylabel("Intensity [DNs]")
        ax1.set_xlabel(xlabel)
        fig.suptitle(f"{datetime} {el}{self.aa}")
        # ax1.set_title(f"({lon}, {lat})")
        # ax1.tick_params(direction="in")
        fig.show()

    @plt.rc_context(rc_context_dict)
    def plot_stokes(
        self,
        stokes: str,
        unit: Optional[u.Unit] = None,
        air: bool = False,
        d: bool = False,
    ) -> None:
        """
        Plots the Stokes profiles for a given slice of the data.

        Parameters
        ----------
        stokes : str
            This is to ensure the plots are labelled correctly. Choose "all" to
            plot the 4 Stokes profiles or a combination e.g. "IQU", "QV" or
            single letter to plot just one of the Stokes parameters e.g. "U".
        unit : astropy.units.Unit or None, optional
            The unit to have the wavelength axis in. Default is None which
            changes the units to Angstrom.
        air : bool, optional
            Whether or not to convert the wavelength axis to air wavelength (if
            it is not already been converted). e.g. for the Ca II 8542  spectral
            line, 8542 is the rest wavelength of the spectral line measured in
            air. It is possible that the header data (and by proxy the WCS) will
            have the value of the rest wavelength in vacuum (which in this case
            is 8544). Default is False.
        d : bool, optional
            Converts the wavelength axis to :math:`\\Delta \\lambda`. Default is False.
        """

        # coord = self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(*self.ind[-2:])
        # lon, lat = np.round(coord.Tx, decimals=2), np.round(coord.Ty, decimals=2)
        try:
            datetime = self.header["DATE-AVG"]
            el = self.header["WDESC1"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
            el = self.header["element"]

        if self.data.ndim == 1:
            wavelength = self.wave(np.arange(self.data.shape[0]))

            if unit != None:
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
            ax1.plot(wavelength, self.data, c=pt_bright["blue"], marker="o")
            if stokes == "I":
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(xlabel)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I")
            elif stokes == "Q":
                ax1.set_ylabel("Q [DNs]")
                ax1.set_xlabel(xlabel)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q")
            elif stokes == "U":
                ax1.set_ylabel("U [DNs]")
                ax1.set_xlabel(xlabel)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes U")
            elif stokes == "V":
                ax1.set_ylabel("V [DNs]")
                ax1.set_xlabel(xlabel)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes V")
            else:
                raise ValueError("This is not a Stokes.")
            # ax1.tick_params(direction="in")
            fig.show()
        elif self.data.ndim == 2:
            wavelength = self.wave(np.arange(self.data.shape[0])) << u.m

            if unit != None:
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
                fig.suptitle(f"{datetime} {el} {self.aa} All  Stokes")
                ax[0, 0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0, 0].set_ylabel("I [DNs]")
                ax[0, 0].tick_params(labelbottom=False)
                ax[0, 1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[0, 1].set_ylabel("Q [DNs]")
                ax[0, 1].yaxis.set_label_position("right")
                ax[0, 1].yaxis.tick_right()
                ax[0, 1].tick_params(labelbottom=False)

                ax[1, 0].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[1, 0].set_ylabel("U [DNs]")
                ax[1, 0].set_xlabel(xlabel)
                # ax[1,0].tick_params(direction="in")

                ax[1, 1].plot(wavelength, self.data[3], c=pt_bright["blue"], marker="o")
                ax[1, 1].set_ylabel("V [DNs]")
                ax[1, 1].set_xlabel(xlabel)
                ax[1, 1].yaxis.set_label_position("right")
                ax[1, 1].yaxis.ticks_right()
                # ax[1,1].tick_params(direction="in")
            elif stokes == "IQU":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q, U")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("U [DNs]")
                ax[2].set_xlabel(xlabel)
                # ax[2].tick_params(direction="in")
            elif stokes == "QUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, U, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                # ax[2].tick_params(direction="in")
            elif stokes == "IQV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                # ax[2].tick_params(direction="in")
            elif stokes == "IUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, U, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                # ax[2].tick_params(direction="in")
            elif stokes == "IQ":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "IU":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, U")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "IV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "QU":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, U")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "QV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "UV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes U, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("U [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

        fig.show()

    @plt.rc_context(rc_context_dict)
    def intensity_map(
        self,
        frame: Optional[str] = None,
        norm: Optional[matplotlib.colors.Normalize] = None,
    ) -> None:
        """
        This plots the image for a certain wavelength depending on a specific slice.

        Parameters
        ----------
        frame : str or None, optional
            The units to use on the axes. Default is None so the WCS is used.
            Other option is "pix" for pixel frame.
        norm : matplotlib.colors.Normalize or None, optional
            The normalisation to use in the colourmap.
        """

        if isinstance(self.ind, int):
            idx = self.ind
        elif self.wcs.low_level_wcs._wcs.naxis == 4:
            idx = self.ind[1]
        else:
            idx = self.ind[0]
        wvl = np.round(self.wave(idx) << u.Angstrom, decimals=2).value
        del_wvl = np.round(
            wvl
            - (
                self.wave(self.wcs.low_level_wcs._wcs.array_shape[0] // 2) << u.Angstrom
            ).value,
            decimals=2,
        )
        try:
            datetime = self.header["DATE-AVG"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]

        if self.data.min() < 0:
            vmin = 0
        else:
            vmin = self.data.min()

        if frame is None and not self.rotate:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
            im1 = ax1.imshow(self.data, cmap="Greys_r", vmin=vmin, norm=norm)
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(
                f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})"
            )
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()
        elif frame == "pix":
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(
                self.data, cmap="Greys_r", vmin=vmin, origin="lower", norm=norm
            )
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(
                f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})"
            )
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()
        elif (frame == "arcsec") or (frame is None and self.rotate):
            try:
                xmax = self.header["CDELT1"] * self.shape[-1]
                ymax = self.header["CDELT2"] * self.shape[-2]
            except KeyError:
                xmax = self.header["pixel_scale"] * self.shape[-1]
                ymax = self.header["pixel_scale"] * self.shape[-2]
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(
                self.data,
                cmap="Greys_r",
                vmin=vmin,
                origin="lower",
                norm=norm,
                extent=[0, xmax, 0, ymax],
            )
            ax1.set_ylabel("y [arcsec]")
            ax1.set_xlabel("x [arcsec]")
            ax1.set_title(
                f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})"
            )
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()

    @plt.rc_context(rc_context_dict)
    def stokes_map(self, stokes: str, frame: Optional[str] = None) -> None:
        """
        This plots the Stokes images for certain wavelength.

        Parameters
        ----------
        stokes : str
            This is to ensure the plots are labelled correctly. Choose "all" to
            plot the 4 Stokes profiles or a combination e.g. "IQU", "QV" or
            single letter to plot just one of the Stokes parameters e.g. "U".
        frame : str or None, optional
            The units to use on the axes. Default is None so the WCS is used.
            Other option is "pix" for pixel frame.
        """

        wvl = np.round(
            self.wcs.low_level_wcs._wcs[0, :, 0, 0].array_index_to_world(self.ind[1])
            << u.Angstrom,
            decimals=2,
        ).value
        del_wvl = np.round(
            wvl
            - (
                self.wcs.low_level_wcs._wcs[0, :, 0, 0].array_index_to_world(
                    self.wcs.low_level_wcs._wcs.array_shape[1] // 2
                )
                << u.Angstrom
            ).value,
            decimals=2,
        )
        try:
            datetime = self.header["DATE-AVG"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
        title = (
            f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l}={del_wvl}{self.aa})"
        )

        if frame is None and not self.rotate:
            if self.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
                if stokes == "I":
                    data = self.data
                    data[data < 0] = np.nan
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_title("Stokes I " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(self.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title("Stokes Q " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(self.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title("Stokes U " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(self.data, cmap="Greys_r", vmin=-100, vmax=100)
                    ax1.set_title("Stokes V " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                fig.show()
            elif self.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure(constrained_layout=True)
                    fig.suptitle(title)
                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        2, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel(" ")
                    # ax1.xaxis.set_label_position("top")
                    ax1.xaxis.tick_top()
                    ax1.set_title("Stokes I ")
                    ax1.tick_params(axis="x", labelbottom=False)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        2, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel(" ")
                    # ax2.xaxis.set_label_position("top")
                    ax2.xaxis.tick_top()
                    # ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()
                    ax2.set_title("Stokes Q ")
                    ax2.tick_params(axis="x", labelbottom=False)
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(
                        2, 2, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U ")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")

                    ax4 = fig.add_subplot(
                        2, 2, 4, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 3)
                    )
                    im4 = ax4.imshow(self.data[3], cmap="Greys_r", vmin=-100, vmax=100)
                    ax4.set_ylabel(" ")
                    ax4.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax4.yaxis.set_label_position("right")
                    ax4.yaxis.ticks_right()
                    ax4.set_title("Stokes V ")
                    ax4.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im4, ax=ax4, orientation="vertical", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(
                        1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel(" ")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(
                        1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(self.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(
                        1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(
                        1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel(" ")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(
                        1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel(" ")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(
                        1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel(" ")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel("")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(self.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(self.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(self.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes U")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
        elif frame == "pix":
            if self.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                if stokes == "I":
                    data = self.data
                    data[data < 0] = np.nan
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_title("Stokes I " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(
                        self.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_title("Stokes Q " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(
                        self.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_title("Stokes U " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(
                        self.data, cmap="Greys_r", vmin=-100, vmax=100, origin="lower"
                    )
                    ax1.set_title("Stokes V " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("y [pixels]")
                ax1.set_xlabel("x [pixels]")
                fig.show()
            elif self.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure(constrained_layout=True)
                    fig.suptitle(title)
                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(2, 2, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xticks([])
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(2, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xticks([])
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(2, 2, 3)
                    im3 = ax3.imshow(
                        self.data[2], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")

                    ax4 = fig.add_subplot(2, 2, 4)
                    im4 = ax4.imshow(
                        self.data[3],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax4.set_yticks([])
                    ax4.set_xlabel("x [pixels]")
                    ax4.set_title("Stokes V")
                    fig.colorbar(im4, ax=ax4, orientation="vertical", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax3.set_yticks([])
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        self.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax3.set_yticks([])
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax3.set_yticks([])
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax3.set_yticks([])
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes U")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_title("Stokes V ")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
        elif (frame == "arcsec") or (frame is None and self.rotate):
            try:
                xmax = self.header["CDELT1"] * self.shape[-1]
                ymax = self.header["CDELT2"] * self.shape[-2]
            except KeyError:
                xmax = self.header["pixel_scale"] * self.shape[-1]
                ymax = self.header["pixel_scale"] * self.shape[-2]
            if self.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                if stokes == "I":
                    data = self.data
                    data[data < 0] = np.nan
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_title("Stokes I " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(
                        self.data,
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_title("Stokes Q " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(
                        self.data,
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_title("Stokes U " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(
                        self.data,
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_title("Stokes V " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("y [arcsec]")
                ax1.set_xlabel("x [arcsec]")
                fig.show()
            elif self.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure(constrained_layout=True)
                    fig.suptitle(title)
                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(2, 2, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xticks([])
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(2, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xticks([])
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(2, 2, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_ylabel("y [arcsec]")
                    ax3.set_xlabel("x [arcsed]")
                    ax3.set_title("Stokes U")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")

                    ax4 = fig.add_subplot(2, 2, 4)
                    im4 = ax4.imshow(
                        self.data[3],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax4.set_yticks([])
                    ax4.set_xlabel("x [arcsed]")
                    ax4.set_title("Stokes V")
                    fig.colorbar(im4, ax=ax4, orientation="vertical", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_yticks([])
                    ax3.set_xlabel("x [arcsec]")
                    ax3.set_title("Stokes U")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        self.data[0],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_yticks([])
                    ax3.set_xlabel("x [arcsec]")
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_yticks([])
                    ax3.set_xlabel("x [arcsec]")
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_yticks([])
                    ax3.set_xlabel("x [arcsec]")
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes V")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes V")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes U")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_title("Stokes V ")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")

        fig.show()

    def wave(self, idx: Union[int, Sequence[int]]) -> np.ndarray:
        """
        This function will take an index number or range and return the wavelength in Angstroms.

        Parameters
        ----------
        idx : int or numpy.ndarray of ints
            The index or indices along the wavelength axis to be converted to
            physical units.

        Returns
        -------
        float or numpy.ndarray of floats
            The wavelength or wavelengths indicated by the index/indices passed
            to the function.
        """
        if len(self.wcs.low_level_wcs.array_shape) == 4:
            if hasattr(self, "ind") and isinstance(self.ind[1], slice):
                return (
                    self.wcs.low_level_wcs._wcs[
                        0, self.ind[1], 0, 0
                    ].array_index_to_world(idx)
                    << u.Angstrom
                )
            elif hasattr(self, "ind") and not isinstance(self.ind[1], slice):
                return (
                    self.wcs.low_level_wcs._wcs[0, :, 0, 0].array_index_to_world(idx)
                    << u.Angstrom
                )
            else:
                return self.wcs[0, :, 0, 0].array_index_to_world(idx) << u.Angstrom
        elif len(self.wcs.low_level_wcs.array_shape) == 3:
            if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                if isinstance(self.ind[1], slice):
                    return (
                        self.wcs.low_level_wcs._wcs[
                            0, self.ind[1], 0, 0
                        ].array_index_to_world(idx)
                        << u.Angstrom
                    )
                else:
                    return (
                        self.wcs.low_level_wcs._wcs[0, :, 0, 0].array_index_to_world(
                            idx
                        )
                        << u.Angstrom
                    )
            else:
                if hasattr(self, "ind") and isinstance(self.ind[0], slice):
                    return (
                        self.wcs.low_level_wcs._wcs[
                            self.ind[0], 0, 0
                        ].array_index_to_world(idx)
                        << u.Angstrom
                    )
                elif hasattr(self, "ind") and not isinstance(self.ind[0], slice):
                    return (
                        self.wcs.low_level_wcs._wcs[:, 0, 0].array_index_to_world(idx)
                        << u.Angstrom
                    )
                else:
                    return self.wcs.array_index_to_world(idx, 0, 0)[1] << u.Angstrom
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            if hasattr(self, "ind"):
                if self.wcs.low_level_wcs._wcs.naxis == 4:
                    return (
                        self.wcs.low_level_wcs._wcs[0, :, 0, 0].array_index_to_world(
                            idx
                        )
                        << u.Angstrom
                    )
                elif self.wcs.low_level_wcs._wcs.naxis == 3:
                    return (
                        self.wcs.low_level_wcs._wcs[:, 0, 0].array_index_to_world(idx)
                        << u.Angstrom
                    )
                else:
                    raise IndexError("There is no spectral component to your data.")
            else:
                raise IndexError("There is no spectral component to your data.")
        elif len(self.wcs.low_level_wcs.array_shape) == 1:
            return self.wcs.array_index_to_world(idx) << u.Angstrom
        else:
            raise NotImplementedError(
                "This is way too many dimensions for me to handle."
            )

    def to_lonlat(
        self, y: int, x: int, coord: bool = False, unit: bool = False
    ) -> Tuple[float, float]:
        """
        This function will take a y, x coordinate in pixel space and map it to
        Helioprojective Longitude, Helioprojective Latitude according to the
        transform in the WCS. This will return the Helioprojective coordinates
        in units of arcseconds. Note this function takes arguments in the order
        of numpy indexing (y,x) but returns a pair longitude/latitude which is
        Solar-X, Solar-Y.

        Parameters
        ----------
        y : int
            The y-index to be converted to Helioprojective Latitude.
        x : int
            The x-index to be converted to Helioprojective Longitude.
        coord : bool, optional
            Whether or not to return an ```astropy.coordinates.SkyCoord```. Default is False.
        unit : bool, optional
            Whether or not to return the values with associated
            ```astropy.units```. Default is False.

        Returns
        -------
        tuple[float]
            A tuple containing the Helioprojective Longitude and Helioprojective
            Latitude of the indexed point.
        """
        if coord:
            if len(self.wcs.low_level_wcs.array_shape) == 4:
                if hasattr(self, "ind"):
                    if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, 0, self.ind[-2], self.ind[-1]
                        ].array_index_to_world(y, x)
                    elif isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, 0, self.ind[-2]
                        ].array_index_to_world(y, x)
                    elif not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, 0, :, self.ind[-1]
                        ].array_index_to_world(y, x)
                    else:
                        return self.wcs.low_level_wcs._wcs[0, 0].array_index_to_world(
                            y, x
                        )
                else:
                    return self.wcs[0, 0].array_index_to_world(y, x)
            elif len(self.wcs.low_level_wcs.array_shape) == 3:
                if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                    if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, 0, self.ind[-2], self.ind[-1]
                        ].array_index_to_world(y, x)
                    elif isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, 0, self.ind[-2]
                        ].array_index_to_world(y, x)
                    elif not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, 0, :, self.ind[-1]
                        ].array_index_to_world(y, x)
                    else:
                        return self.wcs.low_level_wcs._wcs[0, 0].array_index_to_world(
                            y, x
                        )
                else:
                    if hasattr(self, "ind"):
                        if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                            return self.wcs.low_level_wcs._wcs[
                                0, self.ind[-2], self.ind[-1]
                            ].array_index_to_world(y, x)
                        elif (
                            isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice)
                        ):
                            return self.wcs.low_level_wcs._wcs[
                                0, self.ind[-2]
                            ].array_index_to_world(y, x)
                        elif (
                            not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice)
                        ):
                            return self.wcs.low_level_wcs._wcs[
                                0, :, self.ind[-1]
                            ].array_index_to_world(y, x)
                        else:
                            return self.wcs.low_level_wcs._wcs[0].array_index_to_world(
                                y, x
                            )
                    else:
                        return self.wcs[0].array_index_to_world(y, x)
            elif len(self.wcs.low_level_wcs.array_shape) == 2:
                return self.wcs.array_index_to_world(y, x)
            else:
                raise NotImplementedError("Too many or too little dimensions.")
        else:
            if unit:
                if len(self.wcs.low_level_wcs.array_shape) == 4:
                    if hasattr(self, "ind"):
                        if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, self.ind[-2], self.ind[-1]
                            ].array_index_to_world(y, x)
                            return sc.Tx, sc.Ty
                        elif (
                            isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice)
                        ):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, self.ind[-2]
                            ].array_index_to_world(y, x)
                            return sc.Tx, sc.Ty
                        elif (
                            not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice)
                        ):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, :, self.ind[-1]
                            ].array_index_to_world(y, x)
                            return sc.Tx, sc.Ty
                        else:
                            sc = self.wcs.low_level_wcs._wcs[0, 0].array_index_to_world(
                                y, x
                            )
                            return sc.Tx, sc.Ty
                    else:
                        sc = self.wcs[0, 0].array_index_to_world(y, x)
                        return sc.Tx, sc.Ty
                elif len(self.wcs.low_level_wcs.array_shape) == 3:
                    if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                        if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, self.ind[-2], self.ind[-1]
                            ].array_index_to_world(y, x)
                            return sc.Tx, sc.Ty
                        elif (
                            isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice)
                        ):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, self.ind[-2]
                            ].array_index_to_world(y, x)
                            return sc.Tx, sc.Ty
                        elif (
                            not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice)
                        ):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, :, self.ind[-1]
                            ].array_index_to_world(y, x)
                            return sc.Tx, sc.Ty
                        else:
                            sc = self.wcs.low_level_wcs._wcs[0, 0].array_index_to_world(
                                y, x
                            )
                            return sc.Tx, sc.Ty
                    else:
                        if hasattr(self, "ind"):
                            if (
                                isinstance(self.ind[-2], slice)
                                and isinstance(self.ind[-1], slice)
                            ):
                                sc = self.wcs.low_level_wcs._wcs[
                                    0, self.ind[-2], self.ind[-1]
                                ].array_index_to_world(y, x)
                                return sc.Tx, sc.Ty
                            elif (
                                isinstance(self.ind[-2], slice)
                                and not isinstance(self.ind[-1], slice)
                            ):
                                sc = self.wcs.low_level_wcs._wcs[
                                    0, self.ind[-2]
                                ].array_index_to_world(y, x)
                                return sc.Tx, sc.Ty
                            elif (
                                not isinstance(self.ind[-2], slice)
                                and isinstance(self.ind[-1], slice)
                            ):
                                sc = self.wcs.low_level_wcs._wcs[
                                    0, :, self.ind[-1]
                                ].array_index_to_world(y, x)
                                return sc.Tx, sc.Ty
                            else:
                                sc = self.wcs.low_level_wcs._wcs[
                                    0
                                ].array_index_to_world(y, x)
                                return sc.Tx, sc.Ty
                        else:
                            sc = self.wcs[0].array_index_to_world(y, x)
                            return sc.Tx, sc.Ty
                elif len(self.wcs.low_level_wcs.array_shape) == 2:
                    sc = self.wcs.array_index_to_world(y, x)
                    return sc.Tx, sc.Ty
                else:
                    raise NotImplementedError("Too many or too little dimensions.")
            else:
                if len(self.wcs.low_level_wcs.array_shape) == 4:
                    if hasattr(self, "ind"):
                        if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, self.ind[-2], self.ind[-1]
                            ].array_index_to_world(y, x)
                            return sc.Tx.value, sc.Ty.value
                        elif (
                            isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice)
                        ):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, self.ind[-2]
                            ].array_index_to_world(y, x)
                            return sc.Tx.value, sc.Ty.value
                        elif (
                            not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice)
                        ):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, :, self.ind[-1]
                            ].array_index_to_world(y, x)
                            return sc.Tx.value, sc.Ty.value
                        else:
                            sc = self.wcs.low_level_wcs._wcs[0, 0].array_index_to_world(
                                y, x
                            )
                            return sc.Tx.value, sc.Ty.value
                    else:
                        sc = self.wcs[0, 0].array_index_to_world(y, x)
                        return sc.Tx.value, sc.Ty.value
                elif len(self.wcs.low_level_wcs.array_shape) == 3:
                    if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                        if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, self.ind[-2], self.ind[-1]
                            ].array_index_to_world(y, x)
                            return sc.Tx.value, sc.Ty.value
                        elif (
                            isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice)
                        ):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, self.ind[-2]
                            ].array_index_to_world(y, x)
                            return sc.Tx.value, sc.Ty.value
                        elif (
                            not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice)
                        ):
                            sc = self.wcs.low_level_wcs._wcs[
                                0, 0, :, self.ind[-1]
                            ].array_index_to_world(y, x)
                            return sc.Tx.value, sc.Ty.value
                        else:
                            sc = self.wcs.low_level_wcs._wcs[0, 0].array_index_to_world(
                                y, x
                            )
                            return sc.Tx.value, sc.Ty.value
                    else:
                        if hasattr(self, "ind"):
                            if (
                                isinstance(self.ind[-2], slice)
                                and isinstance(self.ind[-1], slice)
                            ):
                                sc = self.wcs.low_level_wcs._wcs[
                                    0, self.ind[-2], self.ind[-1]
                                ].array_index_to_world(y, x)
                                return sc.Tx.value, sc.Ty.value
                            elif (
                                isinstance(self.ind[-2], slice)
                                and not isinstance(self.ind[-1], slice)
                            ):
                                sc = self.wcs.low_level_wcs._wcs[
                                    0, self.ind[-2]
                                ].array_index_to_world(y, x)
                                return sc.Tx.value, sc.Ty.value
                            elif (
                                not isinstance(self.ind[-2], slice)
                                and isinstance(self.ind[-1], slice)
                            ):
                                sc = self.wcs.low_level_wcs._wcs[
                                    0, :, self.ind[-1]
                                ].array_index_to_world(y, x)
                                return sc.Tx.value, sc.Ty.value
                            else:
                                sc = self.wcs.low_level_wcs._wcs[
                                    0
                                ].array_index_to_world(y, x)
                                return sc.Tx.value, sc.Ty.value
                        else:
                            sc = self.wcs[0].array_index_to_world(y, x)
                            return sc.Tx.value, sc.Ty.value
                elif len(self.wcs.low_level_wcs.array_shape) == 2:
                    sc = self.wcs.array_index_to_world(y, x)
                    return sc.Tx.value, sc.Ty.value
                else:
                    raise NotImplementedError("Too many or too little dimensions.")

    def from_lonlat(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        This function takes a Helioprojective Longitude, Helioprojective
        Latitude pair and converts them to the y, x indices to index the object
        correctly. The function takes its arguments in the order Helioprojective
        Longitude, Helioprojective Latitude but returns the indices in the (y,x)
        format so that the output of this function can be used to directly index
        the object.

        Parameters
        ----------
        lon : float
            The Helioprojective Longitude in arcseconds.
        lat : float
            The Helioprojective Latitude in arcseconds.

        Returns
        -------
        tuple[float]
            A tuple of the index needed to retrieve the point for a specific
            Helioprojective Longitude and Helioprojective Latitude.
        """
        lon, lat = lon << u.arcsec, lat << u.arcsec
        sc = SkyCoord(lon, lat, frame=Helioprojective)
        if len(self.wcs.low_level_wcs.array_shape) == 4:
            if hasattr(self, "ind"):
                if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                    return self.wcs.low_level_wcs._wcs[
                        0, 0, self.ind[-2], self.ind[-1]
                    ].world_to_array_index(sc)
                elif isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice):
                    return self.wcs.low_level_wcs._wcs[
                        0, 0, self.ind[-2]
                    ].world_to_array_index(sc)
                elif not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                    return self.wcs.low_level_wcs._wcs[
                        0, 0, :, self.ind[-1]
                    ].world_to_array_index(sc)
                else:
                    return self.wcs.low_level_wcs._wcs[0, 0].world_to_array_index(sc)
            else:
                return self.wcs[0, 0].world_to_array_index(sc)
        elif len(self.wcs.low_level_wcs.array_shape) == 3:
            if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                    return self.wcs.low_level_wcs._wcs[
                        0, 0, self.ind[-2], self.ind[-1]
                    ].world_to_array_index(sc)
                elif isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice):
                    return self.wcs.low_level_wcs._wcs[
                        0, 0, self.ind[-2]
                    ].world_to_array_index(sc)
                elif not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                    return self.wcs.low_level_wcs._wcs[
                        0, 0, :, self.ind[-1]
                    ].world_to_array_index(sc)
                else:
                    return self.wcs.low_level_wcs._wcs[0, 0].world_to_array_index(sc)
            else:
                if hasattr(self, "ind"):
                    if isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, self.ind[-2], self.ind[-1]
                        ].world_to_array_index(sc)
                    elif isinstance(self.ind[-2], slice) and not isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, self.ind[-2]
                        ].world_to_array_index(sc)
                    elif not isinstance(self.ind[-2], slice) and isinstance(self.ind[-1], slice):
                        return self.wcs.low_level_wcs._wcs[
                            0, :, self.ind[-1]
                        ].world_to_array_index(sc)
                    else:
                        return self.wcs.low_level_wcs._wcs[0].world_to_array_index(sc)
                else:
                    return self.wcs[0].world_to_array_index(sc)
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            return self.wcs.world_to_array_index(sc)
        else:
            raise NotImplementedError("Too many or too little dimensions.")


class CRISPSequence(CRISPSequenceSlicingMixin):
    """
    Class for multiple narrowband CRISP observations.

    Parameters
    ----------
    files : list[dict]
        A list of dictionaries containing the parameters for individual
        ``CRISP`` instances. The function
        ``crispy.utils.CRISP_sequence_generator`` can be used to generate this
        list.
    """

    def __init__(self, files: List[Dict]) -> None:
        self.list = [CRISP(**f) for f in files]
        self.aa = html.unescape("#&8491;")

    def __str__(self) -> str:
        try:
            time = [f.file.header["DATE-AVG"][-12:] for f in self.list]
            date = self.list[0].file.header["DATE-AVG"][:-13]
            cl = [str(np.round(f.file.header["TWAVE1"], decimals=2)) for f in self.list]
            wwidth = [f.file.header["WWIDTH1"] for f in self.list]
            shape = [
                str(
                    [
                        f.file.header[f"NAXIS{j+1}"]
                        for j in reversed(range(f.file.data.ndim))
                    ]
                )
                for f in self.list
            ]
            el = [f.file.header["WDESC1"] for f in self.list]
            pointing_x = str(self.list[0].file.header["CRVAL1"])
            pointing_y = str(self.list[0].file.header["CRVAL2"])
        except KeyError:
            time = [f.file.header["time_obs"] for f in self.list]
            date = self.list[0].file.header["date_obs"]
            cl = [str(f.file.header["crval"][-3]) for f in self.list]
            wwidth = [str(f.file.header["dimensions"][-3]) for f in self.list]
            shape = [str(f.file.header["dimensions"]) for f in self.list]
            el = [f.file.header["element"] for f in self.list]
            pointing_x = str(self.list[0].file.header["crval"][-1])
            pointing_y = str(self.list[0].file.header["crval"][-2])

        return f"""
        CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled [{self.aa}]: {wwidth}
        Pointing [arcsec] (HPLN, HPLT): ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    @property
    def data(self) -> List[np.ndarray]:
        """
        Returns a list of the data arrays.
        """
        return [f.data for f in self.list]

    @property
    def header(self) -> List[Dict]:
        """
        Returns a list of the metainformation of the observations.
        """
        return [f.header for f in self.list]

    @property
    def wvls(self) -> List[np.ndarray]:
        """
        Returns a list of the wavelengths sampled in the observations.
        """
        return [f.wave(np.arange(f.shape[-3])) for f in self.list]

    @property
    def shape(self) -> List[Tuple]:
        """
        Returns a list of the shapes of the data.
        """
        return [f.shape for f in self.list]

    @property
    def info(self) -> str:
        """
        Returns information about the observations.
        """
        return self.__str__()

    @property
    def time(self) -> List[str]:
        """
        The times of the observations.
        """
        return [f.time for f in self.list]

    @property
    def date(self) -> List[str]:
        """
        The dates of the observations.
        """
        return [f.date for f in self.list]

    def rotate_crop(
        self, sep: bool = False, diff_t: bool = False
    ) -> Optional[List[np.ndarray]]:
        """
        Instance method for rotating and cropping data if there is a rotation
        with respect to the image plane.

        Parameters
        ----------
        sep : bool, optional
            Whether or not to return the rotated arrays and not alter the
            ``CRISPSequence`` object. Default is False, the object will be
            changed in place with the original data being stored in the
            respective ``CRISP`` instances' ``full_frame`` attribute.
        diff_t : bool, optional
            Whether or not the sequence of observations are taken at different
            times. Default is False.
        """
        if diff_t:
            if sep:
                return [f.rotate_crop() for f in self.list] # type: ignore

            self.full_frame = [*self.data]
            for f in self.list:
                crop, crop_dict = f.rotate_crop(sep=True) # type: ignore
                f.file.data = crop
                f.file.header["frame_dims"] = crop_dict["frameDims"]
                f.file.header["x_min"] = crop_dict["xMin"]
                f.file.header["x_max"] = crop_dict["xMax"]
                f.file.header["y_min"] = crop_dict["yMin"]
                f.file.header["y_max"] = crop_dict["yMax"]
                f.file.header["angle"] = crop_dict["angle"]
                f.rotate = True
        else:
            if sep:
                return rotate_crop_aligned_data(self.list[0].data, self.list[1].data)

            self.full_frame = [self.list[0].data, self.list[1].data]
            crop_a, crop_b, crop_dict = rotate_crop_aligned_data(
                self.list[0].data, self.list[1].data
            )
            self.list[0].file.data = crop_a
            self.list[1].file.data = crop_b
            self.list[0].file.header["frame_dims"] = crop_dict["frameDims"]
            self.list[0].file.header["x_min"] = crop_dict["xMin"]
            self.list[0].file.header["x_max"] = crop_dict["xMax"]
            self.list[0].file.header["y_min"] = crop_dict["yMin"]
            self.list[0].file.header["y_max"] = crop_dict["yMax"]
            self.list[0].file.header["angle"] = crop_dict["angle"]
            self.list[1].file.header["frame_dims"] = crop_dict["frameDims"]
            self.list[1].file.header["x_min"] = crop_dict["xMin"]
            self.list[1].file.header["x_max"] = crop_dict["xMax"]
            self.list[1].file.header["y_min"] = crop_dict["yMin"]
            self.list[1].file.header["y_max"] = crop_dict["yMax"]
            self.list[1].file.header["angle"] = crop_dict["angle"]
            for f in self.list:
                f.rotate = True
        return None

    def reconstruct_full_frame(self, sep: bool = False) -> Optional[List[np.ndarray]]:
        """
        Instance method to derotate data back into the Helioprojective frame.

        Parameters
        ----------
        sep : bool, optional
            Whether or not to return the derotated arrays and not alter the
            ``CRISPSequence`` object. Default is False, the object will be
            changed in place with the original data being stored in the
            respective ``CRISP`` instances' ``rot_data`` attribute.
        """
        if sep:
            return [f.reconstruct_full_frame(sep=True) for f in self.list] # type: ignore

        for f in self.list:
            f.reconstruct_full_frame(sep=False)
            f.rotate = False
        return None

    @plt.rc_context(rc_context_dict)
    def plot_spectrum(
        self,
        idx: Union[str, int],
        unit: Optional[u.Unit] = None,
        air: bool = False,
        d: bool = False,
    ) -> None:
        """
        Function for plotting the intensity spectrum for a given slice. Can be
        done either for all of the instances or for a single instance.

        Parameters
        ----------
        idx : str or int
            If "all" then the spectrum for a specific slice is plotted for all
            instances. If an int, then the spectrum for a specific slice for a
            specific instance is plotted.
        unit : astropy.units.Unit or None, optional
            The unit to have the wavelength axis in. Default is None which changes the units to Angstrom.
        air : bool, optional
            Whether or not to convert the wavelength axis to air wavelength (if
            it is not already been converted). e.g. for the Ca II 8542  spectral
            line, 8542 is the rest wavelength of the spectral line measured in
            air. It is possible that the header data (and by proxy the WCS) will
            have the value of the rest wavelength in vacuum (which in this case
            is 8544). Default is False.
        d : bool, optional
            Converts the wavelength axis to :math:`\\Delta \\lambda`. Default is False.
        """
        if not isinstance(idx, str):
            self.list[idx].plot_spectrum(unit=unit, air=air, d=d)
        elif idx == "all":
            for f in self.list:
                f.plot_spectrum(unit=unit, air=air, d=d)
        else:
            raise ValueError(f"Unexpected index `{idx}`, expected int or \"all\"")

    @plt.rc_context(rc_context_dict)
    def plot_stokes(
        self,
        idx: Union[str, int],
        stokes: str,
        unit: Optional[u.Unit] = None,
        air: bool = False,
        d: bool = False,
    ) -> None:
        """
        Function for plotting the Stokes profiles for a given slice. Can be done
        either for all of the instances or for a single instance.

        Parameters
        ----------
        idx : str or int
            If "all" then the spectrum for a specific slice is plotted for all
            instances. If an int, then the spectrum for a specific slice for a
            specific instance is plotted.
        stokes : str
            This is to ensure the plots are labelled correctly. Choose "all" to
            plot the 4 Stokes profiles or a combination e.g. "IQU", "QV" or
            single letter to plot just one of the Stokes parameters e.g. "U".
        unit : astropy.units.Unit or None, optional
            The unit to have the wavelength axis in. Default is None which
            changes the units to Angstrom.
        air : bool, optional
            Whether or not to convert the wavelength axis to air wavelength (if
            it is not already been converted). e.g. for the Ca II 8542  spectral
            line, 8542 is the rest wavelength of the spectral line measured in
            air. It is possible that the header data (and by proxy the WCS) will
            have the value of the rest wavelength in vacuum (which in this case
            is 8544). Default is False.
        d : bool, optional
            Converts the wavelength axis to :math:`\\Delta \\lambda`. Default is False.
        """
        if not isinstance(idx, str):
            self.list[idx].plot_stokes(stokes, unit=unit, air=air, d=d)
        elif idx == "all":
            for f in self.list:
                f.plot_stokes(stokes, unit=unit, air=air, d=d)
        else:
            raise ValueError(f"Unexpected index `{idx}`, expected int or \"all\"")

    @plt.rc_context(rc_context_dict)
    def intensity_map(
        self,
        idx: Union[str, int],
        frame: Optional[str] = None,
        norm: Optional[matplotlib.colors.Normalize] = None,
    ) -> None:
        """
        Function for plotting the intensity image for a given wavelength. Can be
        done either for all of the instances or for a single instance.

        Parameters
        ----------
        idx : str or int
            If "all" then the spectrum for a specific slice is plotted for all
            instances. If an int, then the spectrum for a specific slice for a
            specific instance is plotted.
        frame : str or None, optional
            The units to use on the axes. Default is None so the WCS is used.
            Other option is "pix" for pixel frame.
        norm : matplotlib.colors.Normalize or None, optional
            The normalisation to use in the colourmap.
        """
        if not isinstance(idx, str):
            self.list[idx].intensity_map(frame=frame, norm=norm)
        elif idx == "all":
            for f in self.list:
                f.intensity_map(frame=frame, norm=norm)
        else:
            raise ValueError(f"Unexpected index `{idx}`, expected int or \"all\"")

    @plt.rc_context(rc_context_dict)
    def stokes_map(
        self, idx: Union[str, int], stokes: str, frame: Optional[str] = None
    ) -> None:
        """
        Function to plot the Stokes maps for a given wavelength. Can be done
        either for all of the instances or for a single instance.

        Parameters
        ----------
        idx : str or int
            If "all" then the spectrum for a specific slice is plotted for all
            instances. If an int, then the spectrum for a specific slice for a
            specific instance is plotted.
        stokes : str
            This is to ensure the plots are labelled correctly. Choose "all" to
            plot the 4 Stokes profiles or a combination e.g. "IQU", "QV" or
            single letter to plot just one of the Stokes parameters e.g. "U".
        frame : str or None, optional
            The units to use on the axes. Default is None so the WCS is used.
            Other option is "pix" for pixel frame.
        """
        if not isinstance(idx, str):
            self.list[idx].stokes_map(stokes, frame=frame)
        elif idx == "all":
            for f in self.list:
                f.stokes_map(stokes, frame=frame)
        else:
            raise ValueError(f"Unexpected index `{idx}`, expected int or \"all\"")

    def from_lonlat(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        This function takes a Helioprojective Longitude, Helioprojective
        Latitude pair and converts them to the y, x indices to index the object
        correctly. The function takes its arguments in the order Helioprojective
        Longitude, Helioprojective Latitude but returns the indices in the (y,x)
        format so that the output of this function can be used to directly index
        the object.

        Parameters
        ----------
        lon : float
            The Helioprojective Longitude in arcseconds.
        lat : float
            The Helioprojective Latitude in arcseconds.

        Returns
        -------
        tuple[float]
            A tuple of the index needed to retrieve the point for a specific
            Helioprojective Longitude and Helioprojective Latitude.
        """
        return self.list[0].from_lonlat(lon, lat)

    def to_lonlat(self, y: int, x: int) -> Tuple[float, float]:
        """
        This function will take a y, x coordinate in pixel space and map it to
        Helioprojective Longitude, Helioprojective Latitude according to the
        transform in the WCS. This will return the Helioprojective coordinates
        in units of arcseconds. Note this function takes arguments in the order
        of numpy indexing (y,x) but returns a pair longitude/latitude which is
        Solar-X, Solar-Y.

        Parameters
        ----------
        y : int
            The y-index to be converted to Helioprojective Latitude.
        x : int
            The x-index to be converted to Helioprojective Longitude.
        coord : bool, optional
            Whether or not to return an ```astropy.coordinates.SkyCoord```. Default is False.
        unit : bool, optional
            Whether or not to return the values with associated ```astropy.units```. Default is False.

        Returns
        -------
        tuple[float]
            A tuple containing the Helioprojective Longitude and Helioprojective
            Latitude of the indexed point.
        """
        return self.list[0].to_lonlat(y, x)


class CRISPWideband(CRISP):
    """
    Class for wideband or single wavelength CRISP images. This class expects the
    data to be two-dimensional.
    """

    __doc__ += parameter_docstring(CRISP)

    def __str__(self) -> str:
        try:
            time = self.header["DATE-AVG"][-12:]
            date = self.header["DATE-AVG"][:-13]
            shape = str(
                [self.header[f"NAXIS{j+1}"] for j in reversed(range(self.data.ndim))]
            )
            el = self.header["WDESC1"]
            pointing_x = str(self.header["CRVAL1"])
            pointing_y = str(self.header["CRVAL2"])
        except KeyError:
            time = self.header["time_obs"]
            date = self.header["date_obs"]
            shape = str(self.header["dimensions"])
            el = self.header["element"]
            pointing_x = str(self.header["crval"][-1])
            pointing_y = str(self.header["crval"][-2])

        return f"""
        CRISP Wideband Context Image
        ------------------
        {date} {time}

        Observed: {el}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    @plt.rc_context(rc_context_dict)
    def intensity_map(
        self, frame: Optional[str] = None, norm: Optional[str] = None
    ) -> None:
        """
        This function plots the image in the same manner as the
        ``crispy.crisp.CRISP.intensity_map`` method.

        Parameters
        ----------
        frame : str or None, optional
            The frame to plot the data in. Default is None, meaning the WCS
            frame is used. The other option is "pix" to plot in the pixel plane.
        norm : matplotlib.colors.Normalize or None, optional
            The normalisation to use in the colourmap.
        """
        try:
            datetime = self.header["DATE-AVG"]
            el = self.header["WDESC1"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
            el = self.header["element"]

        if frame is None:
            fig = plt.figure()
            data = self.data[...].astype(np.float32)
            data[data < 0] = np.nan
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs)
            im1 = ax1.imshow(data, cmap="Greys_r", norm=norm)
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"{datetime} {el} {self.aa}")
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()
        elif frame == "pix":
            fig = plt.figure()
            data = self.data[...].astype(np.float32)
            data[data < 0] = np.nan
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(data, cmap="Greys_r", origin="lower", norm=norm)
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"{datetime} {el} {self.aa}")
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()
        elif frame == "arcsec":
            try:
                xmax = self.header["CDELT1"] * self.shape[-1]
                ymax = self.header["CDELT2"] * self.shape[-2]
            except KeyError:
                xmax = self.header["pixel_scale"] * self.shape[-1]
                ymax = self.header["pixel_scale"] * self.shape[-2]

            if self.data.min() < 0:
                vmin = 0
            else:
                vmin = self.data.min()

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(
                self.data,
                cmap="Greys_r",
                vmin=vmin,
                origin="lower",
                norm=norm,
                extent=[0, xmax, 0, ymax],
            )
            ax1.set_ylabel("y [arcsec]")
            ax1.set_xlabel("x [arcsec]")
            ax1.set_title(
                f"{datetime} (wideband)"
            )
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()


class CRISPWidebandSequence(CRISPSequence):
    """
    This class is for having a sequence of wideband or single wavelength images
    (preferrably chronologically but no limit is placed on this so y'know be
    careful).

    Parameters
    ----------
    files : list[dict]
        A list of dictionaries containing the parameters for individual
        ``CRISPWideband`` instances. The function
        ``crispy.utils.CRISP_sequence_generator`` can be used to generate this list.
    """

    def __init__(self, files: List[Dict]) -> None:
        self.list = [CRISPWideband(**f) for f in files]

    def __str__(self) -> str:
        try:
            time = [f.file.header["DATE-AVG"][-12:] for f in self.list]
            date = self.list[0].file.header["DATE-AVG"][:-13]
            shape = [
                str(
                    [
                        f.file.header[f"NAXIS{j+1}"]
                        for j in reversed(range(f.file.data.ndim))
                    ]
                )
                for f in self.list
            ]
            el = [f.file.header["WDESC1"] for f in self.list]
            pointing_x = str(self.list[0].file.header["CRVAL1"])
            pointing_y = str(self.list[0].file.header["CRVAL2"])
        except KeyError:
            time = [f.file.header["time_obs"] for f in self.list]
            date = self.list[0].file.header["date_obs"]
            shape = [str(f.file.header["dimensions"]) for f in self.list]
            el = [self.list[0].file.header["element"] for f in self.list]
            pointing_x = str(self.list[0].file.header["crval"][-1])
            pointing_y = str(self.list[0].file.header["crval"][-2])

        return f"""
        CRISP Wideband Context Image
        ------------------
        {date} {time}

        Observed: {el}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}"""


class CRISPNonU(CRISP):
    """
    This is a class for narrowband CRISP observations whose wavelength axis is
    sampled non-uniformly. What this means is that each pair of sampled
    wavelengths is not necessarily separated by the same :math:`\\Delta
    \\lambda` and thus the ``CDELT3`` fits keyword becomes meaningless as this
    can only comprehend constant changes in the third axis. This also means that
    the WCS does not work for the wavelength axis but is still constructed as it
    holds true in the y,x spatial plane. This class assumes that if the sampling
    is non-uniform then the true wavelengths that are sampled are stored in the
    first non-PrimaryHDU in the fits file.
    """

    __doc__ += parameter_docstring(CRISP)

    def __init__(
        self,
        filename: str,
        wcs: Optional[WCS] = None,
        uncertainty: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        nonu: bool = True,
    ) -> None:
        super().__init__(
            filename=filename, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu
        )

        if ".fits" in filename:
            self.wvl = fits.open(filename)[
                1
            ].data  # This assumes that the true wavelength points are stored in the first HDU of the FITS file as a numpy array
        else:
            self.wvl = self.header["wavels"]

    def __str__(self) -> str:
        try:
            time = self.header["DATE-AVG"][-12:]
            date = self.header["DATE-AVG"][:-13]
            cl = str(np.round(self.header["TWAVE1"], decimals=2))
            wwidth = self.header["WWIDTH1"]
            shape = str(
                [self.header[f"NAXIS{j+1}"] for j in reversed(range(self.data.ndim))]
            )
            el = self.header["WDESC1"]
            pointing_x = str(self.header["CRVAL1"])
            pointing_y = str(self.header["CRVAL2"])
        except KeyError:
            time = self.header["time_obs"]
            date = self.header["date_obs"]
            cl = str(self.header["crval"][-3])
            wwidth = self.header["dimensions"][-3]
            shape = str(self.header["dimensions"])
            el = self.header["element"]
            pointing_x = str(self.header["crval"][-1])
            pointing_y = str(self.header["crval"][-2])
        sampled_wvls = str(self.wvls)

        return f"""
        CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled: {wwidth}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}
        Wavelengths sampled: {sampled_wvls}"""

    @property
    def wvls(self) -> np.ndarray:
        """
        The wavelengths sampled in the observation.
        """
        return self.wvl

    @plt.rc_context(rc_context_dict)
    def plot_spectrum(
        self, unit: Optional[u.Unit] = None, air: bool = False, d: bool = False
    ) -> None:
        """
        Plots the intensity spectrum for a specified coordinate by slicing.

        Parameters
        ----------
        unit : astropy.units.Unit or None, optional
            The unit to have the wavelength axis in. Default is None which
            changes the units to Angstrom.
        air : bool, optional
            Whether or not to convert the wavelength axis to air wavelength (if
            it is not already been converted). e.g. for the Ca II 8542  spectral
            line, 8542 is the rest wavelength of the spectral line measured in
            air. It is possible that the header data (and by proxy the WCS) will
            have the value of the rest wavelength in vacuum (which in this case
            is 8544). Default is False.
        d : bool, optional
            Converts the wavelength axis to :math:`\\Delta \\lambda`. Default is False.
        """
        if self.data.ndim != 1:
            raise IndexError(
                "If you are using Stokes data please use the plot_stokes method."
            )

        wavelength = self.wvls
        if unit != None:
            wavelength <<= unit

        if air:
            wavelength = vac_to_air(wavelength)

        if d:
            wavelength = wavelength - np.median(wavelength)
            xlabel = f"{self.D}{self.l} [{self.aa}]"
        else:
            xlabel = f"{self.l} [{self.aa}]"

        # point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0].array_index_to_world(*self.ind[-2:])]
        try:
            datetime = self.header["DATE-AVG"]
            el = self.header["WDESC1"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
            el = self.header["element"]

        fig = plt.figure()
        ax1 = fig.gca()
        ax1.plot(wavelength, self.data, c=pt_bright["blue"], marker="o")
        ax1.set_ylabel("Intensity [DNs]")
        ax1.set_xlabel(xlabel)
        ax1.set_title(f"{datetime} {el} {self.aa}")
        # ax1.tick_params(direction="in")
        fig.show()

    @plt.rc_context(rc_context_dict)
    def plot_stokes(
        self,
        stokes: str,
        unit: Optional[u.Unit] = None,
        air: bool = False,
        d: bool = False,
    ) -> None:
        """
        Plots the Stokes profiles for a given slice of the data.

        Parameters
        ----------
        stokes : str
            This is to ensure the plots are labelled correctly. Choose "all" to
            plot the 4 Stokes profiles or a combination e.g. "IQU", "QV" or
            single letter to plot just one of the Stokes parameters e.g. "U".
        unit : astropy.units.Unit or None, optional
            The unit to have the wavelength axis in. Default is None which
            changes the units to Angstrom.
        air : bool, optional
            Whether or not to convert the wavelength axis to air wavelength (if
            it is not already been converted). e.g. for the Ca II 8542  spectral
            line, 8542 is the rest wavelength of the spectral line measured in
            air. It is possible that the header data (and by proxy the WCS) will
            have the value of the rest wavelength in vacuum (which in this case
            is 8544). Default is False.
        d : bool, optional
            Converts the wavelength axis to :math:`\\Delta \\lambda`. Default is False.
        """

        # point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(*self.ind[-2:])]
        try:
            datetime = self.header["DATE-AVG"]
            el = self.header["WDESC1"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
            el = self.header["element"]

        if self.data.ndim == 1:
            wavelength = self.wvls

            if unit != None:
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
            ax1.plot(wavelength, self.data, c=pt_bright["blue"], marker="o")
            if stokes == "I":
                ax1.set_ylabel("Intensity [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes I")
            elif stokes == "Q":
                ax1.set_ylabel("Q [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes Q")
            elif stokes == "U":
                ax1.set_ylabel("U [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes U")
            elif stokes == "V":
                ax1.set_ylabel("V [DNs]")
                ax1.set_xlabel(xlabel)
                ax1.set_title(f"{datetime} {el} {self.aa} Stokes V")
            else:
                raise ValueError("This is not a Stokes.")
            # ax1.tick_params(direction="in")
            fig.show()
        elif self.data.ndim == 2:
            wavelength = self.wvls

            if unit != None:
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
                fig.suptitle(f"{datetime} {el} {self.aa} All Stokes")
                ax[0, 0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0, 0].set_ylabel("I [DNs]")
                ax[0, 0].tick_params(labelbottom=False)

                ax[0, 1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[0, 1].set_ylabel("Q [DNs]")
                ax[0, 1].yaxis.set_label_position("right")
                ax[0, 1].yaxis.tick_right()
                ax[0, 1].tick_params(labelbottom=False)

                ax[1, 0].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[1, 0].set_ylabel("U [DNs]")
                ax[1, 0].set_xlabel(xlabel)
                # ax[1,0].tick_params(direction="in")

                ax[1, 1].plot(wavelength, self.data[3], c=pt_bright["blue"], marker="o")
                ax[1, 1].set_ylabel("V [DNs]")
                ax[1, 1].set_xlabel(xlabel)
                ax[1, 1].yaxis.set_label_position("right")
                ax[1, 1].yaxis.tick_right()
                # ax[1,1].tick_params(direction="in")
            elif stokes == "IQU":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q, U")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("U [DNs]")
                ax[2].set_xlabel(xlabel)
                # ax[2].tick_params(direction="in")
            elif stokes == "QUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, U, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                # ax[2].tick_params(direction="in")
            elif stokes == "IQV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                # ax[2].tick_params(direction="in")
            elif stokes == "IUV":
                fig, ax = plt.subplots(nrows=1, ncols=3)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, U, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

                ax[2].plot(wavelength, self.data[2], c=pt_bright["blue"], marker="o")
                ax[2].set_ylabel("V [DNs]")
                ax[2].set_xlabel(xlabel)
                # ax[2].tick_params(direction="in")
            elif stokes == "IQ":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, Q")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("Q [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "IU":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, U")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "IV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes I, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("I [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "QU":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, U")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("U [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "QV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes Q, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("Q [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")
            elif stokes == "UV":
                fig, ax = plt.subplots(nrows=1, ncols=2)
                fig.suptitle(f"{datetime} {el} {self.aa} Stokes U, V")

                ax[0].plot(wavelength, self.data[0], c=pt_bright["blue"], marker="o")
                ax[0].set_ylabel("U [DNs]")
                ax[0].set_xlabel(xlabel)
                # ax[0].tick_params(direction="in")

                ax[1].plot(wavelength, self.data[1], c=pt_bright["blue"], marker="o")
                ax[1].set_ylabel("V [DNs]")
                ax[1].set_xlabel(xlabel)
                # ax[1].tick_params(direction="in")

        fig.show()

    @plt.rc_context(rc_context_dict)
    def intensity_map(
        self,
        frame: Optional[str] = None,
        norm: Optional[matplotlib.colors.Normalize] = None,
    ) -> None:
        """
        This plots the image for a certain wavelength depending on a specific slice.

        Parameters
        ----------
        frame : str or None, optional
            The units to use on the axes. Default is None so the WCS is used.
            Other option is "pix" for pixel frame.
        norm : matplotlib.colors.Normalize or None, optional
            The normalisation to use in the colourmap.
        """

        if isinstance(self.ind, int):
            idx = self.ind
        elif self.wcs.low_level_wcs._wcs.naxis == 4:
            idx = self.ind[1]
        else:
            idx = self.ind[0]
        wvl = np.round(self.wave(idx) << u.Angstrom, decimals=2).value
        del_wvl = np.round(
            wvl
            - (
                self.wave(self.wcs.low_level_wcs._wcs.array_shape[0] // 2) << u.Angstrom
            ).value,
            decimals=2,
        )
        try:
            datetime = self.header["DATE-AVG"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]

        if frame is None and not self.rotate:
            fig = plt.figure()
            data = self.data
            data[data < 0] = np.nan
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
            im1 = ax1.imshow(data, cmap="Greys_r", norm=norm)
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(
                f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})"
            )
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()
        elif frame == "pix":
            fig = plt.figure()
            data = self.data
            data[data < 0] = np.nan
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(data, cmap="Greys_r", origin="lower", norm=norm)
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(
                f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})"
            )
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()
        elif (frame == "arcsec") or (frame is None and self.rotate):
            try:
                xmax = self.header["CDELT1"] * self.shape[-1]
                ymax = self.header["CDELT2"] * self.shape[-2]
            except KeyError:
                xmax = self.header["pixel_scale"] * self.shape[-1]
                ymax = self.header["pixel_scale"] * self.shape[-2]

            if self.data.min() < 0:
                vmin = 0
            else:
                vmin = self.data.min()

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            im1 = ax1.imshow(
                self.data,
                cmap="Greys_r",
                vmin=vmin,
                origin="lower",
                norm=norm,
                extent=[0, xmax, 0, ymax],
            )
            ax1.set_ylabel("y [arcsec]")
            ax1.set_xlabel("x [arcsec]")
            ax1.set_title(
                f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})"
            )
            fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
            fig.show()

    @plt.rc_context(rc_context_dict)
    def stokes_map(self, stokes: str, frame: Optional[str] = None) -> None:
        """
        This plots the Stokes images for certain wavelength.

        Parameters
        ----------
        stokes : str
            This is to ensure the plots are labelled correctly. Choose "all" to
            plot the 4 Stokes profiles or a combination e.g. "IQU", "QV" or
            single letter to plot just one of the Stokes parameters e.g. "U".
        frame : str or None, optional
            The units to use on the axes. Default is None so the WCS is used.
            Other option is "pix" for pixel frame.
        """

        wvl = np.round(
            self.wcs.low_level_wcs._wcs[0, :, 0, 0].array_index_to_world(self.ind[1])
            << u.Angstrom,
            decimals=2,
        ).value
        del_wvl = np.round(
            wvl
            - (
                self.wcs.low_level_wcs._wcs[0, :, 0, 0].array_index_to_world(
                    self.wcs.low_level_wcs._wcs.array_shape[1] // 2
                )
                << u.Angstrom
            ).value,
            decimals=2,
        )
        try:
            datetime = self.header["DATE-AVG"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
        title = (
            f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l}={del_wvl}{self.aa})"
        )

        if (frame is None) and (not self.rotate):
            if self.data.ndim == 2:
                fig = plt.figure(constrained_layout=True)
                ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
                if stokes == "I":
                    data = self.data
                    data[data < 0] = np.nan
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_title("Stokes I " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(self.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title("Stokes Q " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(self.data, cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_title("Stokes U " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(self.data, cmap="Greys_r", vmin=-100, vmax=100)
                    ax1.set_title("Stokes V " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                fig.show()
            elif self.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure()
                    fig.suptitle(title)
                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        2, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel(" ")
                    ax1.set_title("Stokes I ")
                    ax1.tick_params(axis="x", labelbottom=False)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        2, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel(" ")
                    ax2.set_title("Stokes Q ")
                    ax2.tick_params(axis="x", labelbottom=False)
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(
                        2, 2, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U ")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")

                    ax4 = fig.add_subplot(
                        2, 2, 4, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 3)
                    )
                    im4 = ax4.imshow(self.data[3], cmap="Greys_r", vmin=-100, vmax=100)
                    ax4.set_ylabel(" ")
                    ax4.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax4.set_title("Stokes V ")
                    ax4.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im4, ax=ax4, orientation="vertical", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(
                        1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-10, vmax=10)
                    ax3.set_ylabel(" ")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes U")
                    ax3.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(
                        1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(self.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(
                        1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(
                        1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel(" ")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(
                        1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel(" ")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 3, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 3, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(
                        1, 3, 3, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 2)
                    )
                    im3 = ax3.imshow(self.data[2], cmap="Greys_r", vmin=-100, vmax=100)
                    ax3.set_ylabel(" ")
                    ax3.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax3.set_title("Stokes V")
                    ax3.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(data, cmap="Greys_r")
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(self.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-10, vmax=10)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes U")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(self.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(
                        1, 2, 1, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 0)
                    )
                    im1 = ax1.imshow(self.data[0], cmap="Greys_r", vmin=-10, vmax=10)
                    ax1.set_ylabel("Helioprojective Latitude [arcsec]")
                    ax1.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax1.set_title("Stokes U")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")

                    ax2 = fig.add_subplot(
                        1, 2, 2, projection=SlicedLowLevelWCS(self.wcs.low_level_wcs, 1)
                    )
                    im2 = ax2.imshow(self.data[1], cmap="Greys_r", vmin=-100, vmax=100)
                    ax2.set_ylabel(" ")
                    ax2.set_xlabel("Helioprojective Longitude [arcsec]")
                    ax2.set_title("Stokes V")
                    ax2.tick_params(axis="y", labelleft=False)
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
        elif frame == "pix":
            if self.data.ndim == 2:
                fig = plt.figure(constrained_layout=True)
                ax1 = fig.add_subplot(1, 1, 1)
                if stokes == "I":
                    data = self.data
                    data[data < 0] = np.nan
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_title("Stokes I " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(
                        self.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_title("Stokes Q " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(
                        self.data, cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_title("Stokes U " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(
                        self.data, cmap="Greys_r", vmin=-100, vmax=100, origin="lower"
                    )
                    ax1.set_title("Stokes V " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("y [pixels]")
                ax1.set_xlabel("x [pixels]")
                fig.show()
            elif self.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure()
                    fig.suptitle(title)
                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(2, 2, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xticks([])
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(2, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_yticks([])
                    ax2.set_xticks([])
                    ax2.set_title("Stokes Q")
                    ax2.tick_params(direction="in")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(2, 2, 3)
                    im3 = ax3.imshow(
                        self.data[2], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax3.set_ylabel("y [pixels]")
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_title("Stokes U")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")

                    ax4 = fig.add_subplot(2, 2, 4)
                    im4 = ax4.imshow(
                        self.data[3],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax4.set_xlabel("x [pixels]")
                    ax4.set_yticks([])
                    ax4.set_title("Stokes V")
                    fig.colorbar(im4, ax=ax4, orientation="vertical", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_yticks([])
                    ax3.set_title("Stokes U")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        self.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_yticks([])
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_yticks([])
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax3.set_xlabel("x [pixels]")
                    ax3.set_yticks([])
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(data, cmap="Greys_r", origin="lower")
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes V")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes V")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0], cmap="Greys_r", vmin=-10, vmax=10, origin="lower"
                    )
                    ax1.set_ylabel("y [pixels]")
                    ax1.set_xlabel("x [pixels]")
                    ax1.set_title("Stokes U")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                    )
                    ax2.set_xlabel("x [pixels]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes V ")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
        elif (frame == "arcsec") or (frame is None and self.rotate):
            try:
                xmax = self.header["CDELT1"] * self.shape[-1]
                ymax = self.header["CDELT2"] * self.shape[-2]
            except KeyError:
                xmax = self.header["pixel_scale"] * self.shape[-1]
                ymax = self.header["pixel_scale"] * self.shape[-2]
            if self.data.ndim == 2:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                if stokes == "I":
                    data = self.data
                    data[data < 0] = np.nan
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_title("Stokes I " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")
                elif stokes == "Q":
                    im1 = ax1.imshow(
                        self.data,
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_title("Stokes Q " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")
                elif stokes == "U":
                    im1 = ax1.imshow(
                        self.data,
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_title("Stokes U " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")
                elif stokes == "V":
                    im1 = ax1.imshow(
                        self.data,
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_title("Stokes V " + title)
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="V [DNs]")
                else:
                    raise ValueError("This is not a Stokes.")
                ax1.set_ylabel("y [arcsec]")
                ax1.set_xlabel("x [arcsec]")
                fig.show()
            elif self.data.ndim == 3:
                if stokes == "all":
                    fig = plt.figure(constrained_layout=True)
                    fig.suptitle(title)
                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(2, 2, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xticks([])
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(2, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_yticks([])
                    ax2.set_xticks([])
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(2, 2, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_ylabel("y [arcsec]")
                    ax3.set_xlabel("x [arcsed]")
                    ax3.set_title("Stokes U")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")

                    ax4 = fig.add_subplot(2, 2, 4)
                    im4 = ax4.imshow(
                        self.data[3],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax4.set_xlabel("x [arcsed]")
                    ax4.set_yticks([])
                    ax4.set_title("Stokes V")
                    fig.colorbar(im4, ax=ax4, orientation="vertical", label="V [DNs]")
                elif stokes == "IQU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_xlabel("x [arcsec]")
                    ax3.set_yticks([])
                    ax3.set_title("Stokes U")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="U [DNs]")
                elif stokes == "QUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        self.data[0],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_xlabel("x [arcsec]")
                    ax3.set_yticks([])
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_xlabel("x [arcsec]")
                    ax3.set_yticks([])
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IUV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 3, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 3, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")

                    ax3 = fig.add_subplot(1, 3, 3)
                    im3 = ax3.imshow(
                        self.data[2],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax3.set_xlabel("x [arcsec]")
                    ax3.set_yticks([])
                    ax3.set_title("Stokes V")
                    fig.colorbar(im3, ax=ax3, orientation="vertical", label="V [DNs]")
                elif stokes == "IQ":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes Q")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Q [DNs]")
                elif stokes == "IU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "IV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    data = self.data[0]
                    data[data < 0] = np.nan
                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        data, cmap="Greys_r", origin="lower", extent=[0, xmax, 0, ymax]
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes I")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="I [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes V")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "QU":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes U")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="U [DNs]")
                elif stokes == "QV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes Q")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Q [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes V")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")
                elif stokes == "UV":
                    fig = plt.figure()
                    fig.suptitle(title)

                    ax1 = fig.add_subplot(1, 2, 1)
                    im1 = ax1.imshow(
                        self.data[0],
                        cmap="Greys_r",
                        vmin=-10,
                        vmax=10,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax1.set_ylabel("y [arcsec]")
                    ax1.set_xlabel("x [arcsec]")
                    ax1.set_title("Stokes U")
                    fig.colorbar(im1, ax=ax1, orientation="vertical", label="U [DNs]")

                    ax2 = fig.add_subplot(1, 2, 2)
                    im2 = ax2.imshow(
                        self.data[1],
                        cmap="Greys_r",
                        vmin=-100,
                        vmax=100,
                        origin="lower",
                        extent=[0, xmax, 0, ymax],
                    )
                    ax2.set_xlabel("x [arcsec]")
                    ax2.set_yticks([])
                    ax2.set_title("Stokes V ")
                    fig.colorbar(im2, ax=ax2, orientation="vertical", label="V [DNs]")

        fig.show()

    def wave(self, idx: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Class method for returning the wavelength sampled at a given index.

        Parameters
        ----------
        idx : int
            The index along the wavelength axis to know the wavelength for.
        """
        return self.wvl[idx]


class CRISPNonUSequence(CRISPSequence):
    """
    This is a class for a sequence of ``CRISPNonU`` objects and operates
    identically to ``CRISPSequence``.

    Parameters
    ----------
    files : list[dict]
        A list of dictionaries containing the parameters for individual
        ``CRISPNonU`` instances. The function
        ``crispy.utils.CRISP_sequence_generator`` can be used to generate this list.
    """

    def __init__(self, files: List[Dict]) -> None:
        self.list = [CRISPNonU(**f) for f in files]

    def __str__(self) -> str:
        try:
            time = self.list[0].file.header["DATE-AVG"][-12:]
            date = self.list[0].file.header["DATE-AVG"][:-13]
            cl = [str(np.round(f.file.header["TWAVE1"], decimals=2)) for f in self.list]
            wwidth = [f.file.header["WWIDTH1"] for f in self.list]
            shape = [
                str(
                    [
                        f.file.header[f"NAXIS{j+1}"]
                        for j in reversed(range(f.file.data.ndim))
                    ]
                )
                for f in self.list
            ]
            el = [f.file.header["WDESC1"] for f in self.list]
            pointing_x = str(self.list[0].file.header["CRVAL1"])
            pointing_y = str(self.list[0].file.header["CRVAL2"])
        except KeyError:
            time = self.list[0].file.header["time_obs"]
            date = self.list[0].file.header["date_obs"]
            cl = [str(f.file.header["crval"][-3]) for f in self.list]
            wwidth = [str(f.file.header["dimensions"][-3]) for f in self.list]
            shape = [str(f.file.header["dimensions"]) for f in self.list]
            el = [f.file.header["element"] for f in self.list]
            pointing_x = str(self.list[0].file.header["crval"][-1])
            pointing_y = str(self.list[0].file.header["crval"][-2])
        sampled_wvls = [f.wvls for f in self.list]

        return f"""
        CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength: {cl}
        Wavelengths sampled: {wwidth}
        Pointing: ({pointing_x}, {pointing_y})
        Shape: {shape}
        Sampled wavlengths: {sampled_wvls}"""
