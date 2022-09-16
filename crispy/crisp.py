from copy import copy
import html
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import astropy.units as u
import matplotlib
import numpy as np
import zarr
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from specutils.utils.wcs_utils import vac_to_air
from sunpy.coordinates import Helioprojective

from .crisp_plotting import (plot_multi_component_spectrum, plot_multi_frame,
                             plot_single_frame, plot_single_spectrum)
from .io import zarr_header_to_wcs
from .mixin import CRISPSequenceSlicingMixin, CRISPSlicingMixin
from .utils import (ObjDict, parameter_docstring, reconstruct_full_frame,
                    rotate_crop_aligned_data, rotate_crop_data)


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
        uniform. Default is False.
    wvl : astropy.Quantity, optional
        The wavelength grid for the data. Will be inferred if not provided. Only
        needed to transfer information due to slicing.
    orig_wvl : astropy.Quantity, optional
        The original (unsliced) wavelength grid for the data. Only needed to transfer
        information due to slicing.
    """

    def __init__(
        self,
        filename: Union[str, ObjDict],
        wcs: Optional[WCS] = None,
        uncertainty: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        nonu: bool = False,
        wvl: Optional[u.Quantity] = None,
        orig_wvl: Optional[u.Quantity] = None,
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

        if wvl is None:
            try:
                if ".fits" in filename:
                    wvl = fits.open(filename)[
                        1
                    ].data  << u.Angstrom # This assumes that the true wavelength points are stored in the first HDU of the FITS file as a numpy array
                else:
                    wvl = self.header["wavels"] << u.Angstrom
            except (AttributeError, KeyError):
                wcs_ndim = len(self.wcs.low_level_wcs.array_shape)
                if wcs_ndim <= 2 or wcs_ndim > 4:
                    raise ValueError("Either too few (check spectral axis), or too many axes.")
                indexing : List[Union[int, slice]] = [0 for _ in range(wcs_ndim)]
                if hasattr(self, "ind"):
                    if isinstance(self.ind, Sequence):
                        if wcs_ndim == 4:
                            entry = self.ind[1]
                        elif wcs_ndim == 3:
                            entry = self.ind[0]
                    else:
                        entry = None
                    if not isinstance(entry, slice):
                        entry = slice(None, None)

                    indexing[-3] = entry
                    w = self.wcs.low_level_wcs._wcs.__getitem__(indexing)
                else:
                    if wcs_ndim == 4:
                        indexing[1] = slice(None, None)
                    elif wcs_ndim == 3:
                        indexing[0] = slice(None, None)
                    w = self.wcs.__getitem__(indexing)
                wvl = w.array_index_to_world(np.arange(self.data.shape[-3])) << u.Angstrom

            orig_wvl = wvl
        elif orig_wvl is None:
            raise ValueError("`wvl` set, but not `orig_wvl`")

        self.wvl = wvl
        self.orig_wvl = orig_wvl

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
        sampled_wvls = str(self.wvls)

        return f"""
        CRISP Observation
        ------------------
        {date} {time}

        Observed: {el}
        Centre wavelength [{self.aa}]: {cl}
        Wavelengths sampled: {wwidth} ({sampled_wvls})
        Pointing [arcsec] (HPLN, HPLT): ({pointing_x}, {pointing_y})
        Shape: {shape}"""

    @property
    def data(self) -> np.ndarray:
        """
        The actual data.
        """
        return self.file.data[...]

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
        return self.wvl

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
        for k, v in crop_dict.items():
            self.file.header[k] = v
        if not isinstance(self.file.header["frame_dims"], list):
            self.file.header["frame_dims"] = repr(crop_dict["frame_dims"])
        a = -crop_dict["angle"]
        c = np.cos(a)
        s = np.sin(a)
        self.file.header["PC1_1"] = c
        self.file.header["PC1_2"] = -s
        self.file.header["PC2_1"] = s
        self.file.header["PC2_2"] = c
        self.file.header["CRPIX1"] -= crop_dict["x_min"]
        self.file.header["CRPIX2"] -= crop_dict["y_min"]
        # NOTE(cmo): WCS Seems to complain about the formatting of comment sometimes, so just squash it
        self.file.header.pop("COMMENT", None)
        try:
            self.wcs = WCS(self.file.header)
        except ValueError:
            self.wcs = zarr_header_to_wcs(self.file.header)
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
            k: self.file.header[k]
            for k in ["frame_dims", "x_min", "x_max", "y_min", "y_max", "angle"]
        }
        if isinstance(crop_dict["frame_dims"], str):
            crop_dict["frame_dims"] = [int(x) for x in crop_dict["frame_dims"][1:-1].split(',')]

        if sep:
            return reconstruct_full_frame(crop_dict, self.data)

        self.rot_data = self.data
        self.file.data = reconstruct_full_frame(crop_dict, self.data)
        self.file.header.pop("PC1_1", None)
        self.file.header.pop("PC1_2", None)
        self.file.header.pop("PC2_1", None)
        self.file.header.pop("PC2_2", None)
        self.file.header["CRPIX1"] += crop_dict["x_min"]
        self.file.header["CRPIX2"] += crop_dict["y_min"]
        # NOTE(cmo): WCS Seems to complain about the formatting of comment sometimes, so just squash it
        self.file.header.pop("COMMENT", None)
        try:
            self.wcs = WCS(self.file.header)
        except ValueError:
            self.wcs = zarr_header_to_wcs(self.file.header)
        self.rotate = False
        return None

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

        self.plot_stokes(stokes="I", unit=unit, air=air, d=d)

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

        wavelength = self.wvl
        self._plot_stokes(stokes, wavelength, unit=unit, air=air, d=d)

    def _plot_stokes(
        self,
        stokes: str,
        wavelength: u.Quantity,
        unit: Optional[u.Unit] = None,
        air: bool = False,
        d: bool = False,
    ) -> None:
        """
        Internal method to plot the Stokes profiles for a given slice of the data.
        Takes wavelength argument to be usable by derived non-uniform classes.

        Parameters
        ----------
        stokes : str
            This is to ensure the plots are labelled correctly. Choose "all" to
            plot the 4 Stokes profiles or a combination e.g. "IQU", "QV" or
            single letter to plot just one of the Stokes parameters e.g. "U".
        wavelength : astropy.Quantity
            The wavelength grid against which to plot the data.
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

        try:
            datetime = self.header["DATE-AVG"]
            el = self.header["WDESC1"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
            el = self.header["element"]

        stokes_components = ["I", "Q", "U", "V"]
        y_labels = {
            "I": "I [DNs]",
            "Q": "Q [DNs]",
            "U": "U [DNs]",
            "V": "V [DNs]",
        }

        if unit is not None:
            wavelength << unit

        if air:
            wavelength = vac_to_air(wavelength)

        if self.data.ndim == 1:
            if stokes not in ["I", "Q", "U", "V"]:
                raise ValueError(
                    f"This ({stokes}) is not a Stokes. Expected (I, Q, U, V)"
                )

            title = f"{datetime} {el} {self.aa} Stokes {y_labels[stokes]}"

            plot_single_spectrum(
                wavelength, self.data, title=title, y_label=y_labels[stokes], d=d
            )

        elif self.data.ndim == 2:
            if stokes == "all":
                title = f"{datetime} {el} {self.aa} All  Stokes"
                components = stokes_components
            elif len(stokes) < 5 and all(s in stokes_components for s in stokes):
                components = [s for s in stokes]
                title = f"{datetime} {el} {self.aa} Stokes {', '.join(components)}"
            else:
                raise ValueError(
                    f"Not all Stokes components requested ({stokes}) are valid (I, Q, U, V)."
                )
            plot_multi_component_spectrum(components, wavelength, self.data, title, d=d)

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

        wvl = np.round(self.wvls << u.Angstrom, decimals=2).value
        del_wvl = np.round(wvl - np.median(self.orig_wvl.value), decimals=2)
        try:
            datetime = self.header["DATE-AVG"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]

        if frame is None:
            frame = "WCS"
        if frame == "WCS" and self.rotate:
            frame = "arcsec"

        extent = None
        if frame == "arcsec":
            try:
                xmax = self.header["CDELT1"] * self.shape[-1]
                ymax = self.header["CDELT2"] * self.shape[-2]
            except KeyError:
                xmax = self.header["pixel_scale"] * self.shape[-1]
                ymax = self.header["pixel_scale"] * self.shape[-2]
            extent = [0.0, xmax, 0.0, ymax]

        title = f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l} = {del_wvl}{self.aa})"
        plot_single_frame(self, frame, title, extent=extent, norm=norm)

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

        stokes_components = ["I", "Q", "U", "V"]

        wvl = np.round(self.wvls << u.Angstrom, decimals=2).value
        del_wvl = np.round(wvl - np.median(wvl), decimals=2)
        try:
            datetime = self.header["DATE-AVG"]
        except KeyError:
            datetime = self.header["date_obs"] + "T" + self.header["time_obs"]
        title = (
            f"{datetime} {self.l}={wvl}{self.aa} ({self.D}{self.l}={del_wvl}{self.aa})"
        )
        if frame is None:
            frame = "WCS"
        if frame == "WCS" and self.rotate:
            frame = "arcsec"

        extent = None
        if frame == "arcsec":
            try:
                xmax = self.header["CDELT1"] * self.shape[-1]
                ymax = self.header["CDELT2"] * self.shape[-2]
            except KeyError:
                xmax = self.header["pixel_scale"] * self.shape[-1]
                ymax = self.header["pixel_scale"] * self.shape[-2]
            extent = [0.0, xmax, 0.0, ymax]

        norms = {
            "I": None,
            "Q": matplotlib.colors.Normalize(vmin=-10, vmax=10),
            "U": matplotlib.colors.Normalize(vmin=-10, vmax=10),
            "V": matplotlib.colors.Normalize(vmin=-100, vmax=100),
        }

        def map_negative_nan(d: np.ndarray) -> np.ndarray:
            d[d < 0.0] = np.nan
            return d

        map_datas = {"I": map_negative_nan, "Q": None, "U": None, "V": None}

        if self.data.ndim == 2:
            if not (len(stokes) == 1 and stokes in stokes_components):
                raise ValueError(
                    f"For 2D data can only plot one Stokes component (expected I, Q, U, V, got {stokes})."
                )

            title = f"Stokes {stokes} {title}"
            plot_single_frame(
                self,
                frame,
                title,
                cb_label=f"{stokes} [DNs]",
                extent=extent,
                norm=norms[stokes],
                map_data=map_datas[stokes],
            )
        elif self.data.ndim == 3:
            if stokes == "all":
                components = stokes_components
            else:
                components = [s for s in stokes]
                if not all(s in stokes_components for s in components):
                    raise ValueError(
                        f"Not all Stokes components requested ({stokes}) are valid (combination of I, Q, U, V expected)."
                    )
            plot_multi_frame(self, components, frame, norms, map_datas, extent)

    def wave(self, idx: Union[int, Sequence[int]]) -> u.Quantity:
        """
        This function will take an index number or range and return the wavelength in Angstroms.

        Parameters
        ----------
        idx : int or numpy.ndarray of ints
            The index or indices along the wavelength axis to be converted to
            physical units.

        Returns
        -------
        astropy.Quantity
            The wavelength or wavelengths indicated by the index/indices passed
            to the function.
        """
        return self.orig_wvl[idx]

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
        wcs_ndim = len(self.wcs.low_level_wcs.array_shape)
        if wcs_ndim == 2:
            result = self.wcs.array_index_to_world(y, x)
        else:
            if wcs_ndim <= 2 or wcs_ndim > 4:
                raise ValueError("Either two few (check spectral axis), or too many axes.")
            if hasattr(self, "ind"):
                wcs_ndim = self.wcs.low_level_wcs._wcs.naxis
                indexing : List[Union[int, slice]] = [0 for _ in range(wcs_ndim)]
                indexing[-2:] = self.ind[-2:]

                result = self.wcs.low_level_wcs._wcs.__getitem__(indexing).array_index_to_world(y, x)
            else:
                indexing = [0 for _ in range(wcs_ndim-2)]
                if len(indexing) > 0:
                    result = self.wcs.__getitem__(indexing).array_index_to_world(y, x)
                else:
                    result = self.wcs.array_index_to_world(y, x)


        if not coord:
            if not unit:
                return result.Tx.value, result.Ty.value
            return result.Tx, result.Ty
        return result

    def from_lonlat(self, lon: float, lat: float) -> Tuple[int, int]:
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
        tuple[int]
            A tuple of the index needed to retrieve the point for a specific
            Helioprojective Longitude and Helioprojective Latitude.
        """
        lon, lat = lon << u.arcsec, lat << u.arcsec
        sc = SkyCoord(lon, lat, frame=Helioprojective)
        wcs_ndim = len(self.wcs.low_level_wcs.array_shape)
        if hasattr(self, "ind"):
            indexing  = copy(self.ind)
            wcs_ndim = self.wcs.low_level_wcs._wcs.naxis
            for i in range(wcs_ndim - 2):
                indexing[i] = 0
            w = self.wcs.low_level_wcs._wcs.__getitem__(indexing)
        else:
            indexing = [0 for _ in range(wcs_ndim - 2)]
            if len(indexing) > 0:
                w = self.wcs.__getitem__(indexing)
            else:
                w = self.wcs
        result = w.world_to_array_index(sc)
        return result

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
                return [f.rotate_crop() for f in self.list]  # type: ignore

            self.full_frame = [*self.data]
            for f in self.list:
                crop, crop_dict = f.rotate_crop(sep=True)  # type: ignore
                f.file.data = crop
                for k, v in crop_dict.items():
                    f.file.header[k] = v
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
            for l in self.list[:2]:
                for k, v in crop_dict.items():
                    l.file.header[k] = v
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
            return [f.reconstruct_full_frame(sep=True) for f in self.list]  # type: ignore

        for f in self.list:
            f.reconstruct_full_frame(sep=False)
            f.rotate = False
        return None

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
            raise ValueError(f'Unexpected index `{idx}`, expected int or "all"')

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
            raise ValueError(f'Unexpected index `{idx}`, expected int or "all"')

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
            raise ValueError(f'Unexpected index `{idx}`, expected int or "all"')

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
            raise ValueError(f'Unexpected index `{idx}`, expected int or "all"')

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

    def __init__(self, filename: Union[str, ObjDict], wcs: Optional[WCS] = None, uncertainty: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> None:
        super().__init__(filename, wcs, uncertainty, mask, nonu=False, wvl=0.0 << u.Angstrom, orig_wvl=0.0 << u.Angstrom)


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
            frame = "WCS"
        if frame == "WCS" and self.rotate:
            frame = "arcsec"

        extent = None
        if frame == "arcsec":
            try:
                xmax = self.header["CDELT1"] * self.shape[-1]
                ymax = self.header["CDELT2"] * self.shape[-2]
            except KeyError:
                xmax = self.header["pixel_scale"] * self.shape[-1]
                ymax = self.header["pixel_scale"] * self.shape[-2]
            extent = [0.0, xmax, 0.0, ymax]

        title = f"{datetime} {el} {self.aa} (wideband)"

        def map_negative_nan(d):
            if d.dtype != np.float32:
                d = d.astype(np.float32)
            d[d < 0.0] = np.nan
            return d

        plot_single_frame(
            self, frame, title, extent=extent, norm=norm, map_data=map_negative_nan
        )


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
    def __init__(self, filename: Union[str, ObjDict], wcs: Optional[WCS] = None, uncertainty: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None, nonu: bool = False, wvl: Optional[u.Quantity] = None, orig_wvl: Optional[u.Quantity] = None) -> None:
        warnings.warn("CRISPNonU is deprecated. Just use CRISP.", DeprecationWarning)
        super().__init__(filename, wcs, uncertainty, mask, nonu, wvl, orig_wvl)

class CRISPNonUSequence(CRISPSequence):
    def __init__(self, files: List[Dict]) -> None:
        warnings.warn("CRISPNonUSequence is deprecated. Just use CRISPSequence.", DeprecationWarning)
        super().__init__(files)