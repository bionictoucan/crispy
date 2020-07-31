import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import SymLogNorm
from .mixin import InversionSlicingMixin
from .utils import ObjDict

class Inversion(InversionSlicingMixin):
    """
    Class for transporting and using the inversions obtained from RADYNVERSION.

    :param filename: The file of the inversion. Can be either an hdf5 file path or an ObjDict.
    :type filename: str or ObjDict
    :param wcs: The world-coordinate system (WCS) of the associated observation.
    :type wcs: astropy.wcs.WCS
    :param z: The height grid that the atmospheric parameters are calculated at. This can be either an hdf5 file path or a numpy.ndarray.
    :type z: str or numpy.ndarray
    :param header: The header information of the associated observation.
    :type header: dict or None, optional

    :cvar ne: The electron number density estimated by RADYNVERSION. This is the median solution for a certain number of draws from the latent space.
    :cvar temp: The electron temperature estimated by RADYNVERSION. This is the median solution for a certain number of draws from the latent space.
    :cvar vel: The bulk velocity flow estimated by RADYNVERSION. This is the median solution for a certain number of draws from the latent space.
    :cvar err: This contains the median absolute deviation (MAD, standard error on the median) for each estimated parameter giving a sense of confidence intervals.
    :cvar wcs: The WCS from the observartion associated with the inversion.
    :cvar z: The height grid the inversions are estimated on.
    :cvar header: The header information from the observation associated with the inversion.
    """
    def __init__(self, filename, wcs, z, header=None):
        if type(filename) == str:
            self.f = h5py.File(filename, "r")
            self.ne = self.f["ne"][:,:,:]
            self.temp = self.f["temperature"][:,:,:]
            self.vel = self.f["vel"][:,:,:]
            self.err = self.f["mad"][:,:,:,:]
            self.wcs = wcs
            if type(z) == str:
                self.z = h5py.File(z, "r").get("z")
            else:
                self.z = z
            self.header = header
        elif type(filename) == ObjDict:
            self.ne = filename["ne"]
            self.temp = filename["temperature"]
            self.vel = filename["vel"]
            self.err = filename["mad"]
            self.wcs = wcs
            self.z = z
            self.header = header

    def __str__(self):
        if self.header is None:
            return "You know more about this data than me, m8."
        else: 
            try :       
                time = self.header["DATE-AVG"][-12:]
                date = self.header["DATE-AVG"][:-13]
                pointing_x = str(self.header["CRVAL1"])
                pointing_y = str(self.header["CRVAL2"])
            except KeyError:
                time = self.header["time-obs"]
                date = self.header["date-obs"]
                pointing_x = str(self.header["crval"][-1])
                pointing_y = str(self.header["crval"][-2])

            return f"""Inversion
            ------------------
            {date} {time}

            Pointing: ({pointing_x}, {pointing_y})"""

    def plot_ne(self, eb=False):
        """
        Class method to plot the electron number density for a given location within the field-of-view. This works by slicing the ``Inversion`` object.

        Parameters
        ----------
        eb : bool, optional
            Whether or not to plot the median absolute deviation (MAD) for the electron number density as errorbars. Default is False.
        """
        point = self.to_lonlat(self.ind[0],self.ind[1])
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]

            title = f"{datetime} ({point[0]},{point[1]})"
        else:
            title = f"({point[0]}, {point[1]})"
        fig = plt.figure()
        ax1 = fig.gca()
        if eb:
            ax1.errorbar(self.z, self.ne, yerr=self.mad[0], capsize=3)
        else:
            ax1.plot(self.z, self.ne)
        ax1.set_ylabel(r"log$_{10}$ n$_{\text{e}}$ \[cm$^{-3}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title(f"Electron Number Density {title}")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_temp(self, eb=False):
        """
        Class method to plot the electron temperature for a given point in the field-of-view. This is done by slicing the ``Inversion`` object.

        Parameters
        ----------
        eb : bool, optional
            Whether or not to plot the median absolute deviation (MAD) of the estimated electron temperatures as errorbars. Default is False.
        """
        point = self.to_lonlat(self.ind[0],self.ind[1])
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]

            title = f"{datetime} ({point[0]},{point[1]})"
        else:
            title = f"({point[0]}, {point[1]})"
        fig = plt.figure()
        ax1 = fig.gca()
        if eb:
            ax1.errorbar(self.z, self.temp, yerr=self.mad[1], capsize=3)
        else:
            ax1.plot(self.z, self.temp)
        ax1.set_ylabel(r"log$_{10}$ T \[K\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title(f"Electron Temperature {title}")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_vel(self, eb=False):
        """
        Class method to plot the bulk velocity for a certain point within the field-of-view. This is done using a slice of the ``Inversion`` instance.

        Parameters
        ----------
        eb : bool, optional
            Whether or not to plot the median absolute deviation (MAD) of the bulk velocity as errorbars. Default is False.
        """
        point = self.to_lonlat(self.ind[0],self.ind[1])
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]

            title = f"{datetime} ({point[0]},{point[1]})"
        else:
            title = f"({point[0]}, {point[1]})"
        fig = plt.figure()
        ax1 = fig.gca()
        if eb:
            ax1.errorbar(self.z, self.vel, yerr=self.mad[2], capsize=3)
        else:
            ax1.plot(self.z, self.vel)
        ax1.set_ylabel(r"Bulk Plasma Flow \[km s$^{-1}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title(f"Bulk Plasma Flow {title}")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_params(self, eb=False):
        """
        Class method to plot the electron number density, electron temperature, and bulk velocity for a certain point within the field-of-view. This is done using a slice of the ``Inversion`` instance.

        Parameters
        ----------
        eb : bool, optional
            Whether or not to plot the median absolute deviation (MAD) for each estimated quantity as errorbars. Default is False.
        """
        point = self.to_lonlat(self.ind[0],self.ind[1])
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]

            title = f"{datetime} ({point[0]},{point[1]})"
        else:
            title = f"({point[0]}, {point[1]})"
        fig = plt.figure()
        fig.suptitle(title)
        ax1 = fig.add_subplot(1, 3, 1)
        if self.eb:
            ax1.errorbar(self.z, self.ne, yerr=self.mad[0], capsize=3)
        else:
            ax1.plot(self.z, self.ne)
        ax1.set_ylabel(r"log$_{10}$ n$_{e}$ \[cm$^{-3}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title("Electron Number Density")
        ax1.tick_params(direction="in")

        ax2 = fig.add_subplot(1, 3, 2)
        if self.eb:
            ax2.errorbar(self.z, self.temp, yerr=self.mad[1], capsize=3)
        else:
            ax2.plot(self.z, self.temp)
        ax2.set_ylabel(r"log$_{10}$ T \[K\]")
        ax2.set_xlabel("z [Mm]")
        ax2.set_title("Electron Temperature")
        ax2.tick_params(direction="in")

        ax3 = fig.add_subplot(1, 3, 3)
        if self.eb:
            ax3.errorbar(self.z, self.vel, yerr=self.mad[2], capsize=3)
        else:
            ax3.plot(self.z, self.vel)
        ax3.set_ylabel(r"Bulk Plasma Flow \[km s$^{-1}\]")
        ax3.set_xlabel("z [Mm]")
        ax3.set_title("Bulk Plasma Flow")
        ax3.tick_params(direction="in")
        fig.show()

    def ne_map(self, frame=None):
        """
        Creates an electron density map at a specified height denoted in the ``Inversion`` slice.

        Parameters
        ----------
        frame : str, optional
            The frame to plot the map in. Default is None therefore uses the WCS frame. Other option is "pix" to plot in the pixel frame.
        """
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[-1]
        height = np.round(self.z[idx], decimals=2)
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]
        else:
            datetime = ""

        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs)
            ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"Electron Number Density {datetime} z={height}Mm")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"Electron Number Density {datetime} z={height}Mm")
            fig.show()

    def temp_map(self, frame=None):
        """
        Creates an electron temperature map at a specified height denoted in the ``Inversion`` slice.

        Parameters
        ----------
        frame : str, optional
            The frame to plot the map in. Default is None therefore uses the WCS frame. Other option is "pix" to plot in the pixel frame.
        """
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[-1]
        height = np.round(self.z[idx], decimals=2)
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]
        else:
           datetime = ""
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs)
            ax1.imshow(self.temp, cmap="hot")
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"Electron Temperature {datetime} z={height}Mm")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            ax1.imshow(self.temp, cmap="cividis")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"Electron Temperature {datetime} z={height}Mm")
            fig.show()

    def vel_map(self, frame=None):
        """
        Creates a bulk velocity map at a specified height denoted in the ``Inversion`` slice.

        Parameters
        ----------
        frame : str, optional
            The frame to plot the map in. Default is None therefore uses the WCS frame. Other option is "pix" to plot in the pixel frame.
        """
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[-1]
        height = np.round(self.z[idx], decimals=2)
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]
        else:
            datetime = ""
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs)
            ax1.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"Bulk Velocity Flow {datetime} z={height}Mm")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            ax1.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"Bulk Velocity Flow {datetime} z={height}Mm")
            fig.show()

    def params_map(self, frame=None):
        """
        Creates maps of electron number density, electron temperature, and bulk velocity at a specified height denoted in the ``Inversion`` slice.

        Parameters
        ----------
        frame : str, optional
            The frame to plot the map in. Default is None therefore uses the WCS frame. Other option is "pix" to plot in the pixel frame.
        """
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[-1]
        height = np.round(self.z[idx], decimals=2)
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]
        else:
            datetime = ""
        if frame is None:
            fig = plt.figure()
            fig.suptitle(f"{datetime} z={height}Mm")
            ax1 = fig.add_subplot(1, 3, 1, projection=self.wcs)
            ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title("Electron Number Density")

            ax2 = fig.add_subplot(1, 3, 2, projection=self.wcs)
            ax2.imshow(self.temp, cmap="hot")
            ax2.set_ylabel("Helioprojective Latitude [arcsec]")
            ax2.set_xlabel("Helioprojective Longitude [arcsec]")
            ax2.set_title("Electron Temperature")

            ax3 = fig.add_subplot(1, 3, 3, projection=self.wcs)
            ax3.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax3.set_ylabel("Helioprojective Latitude [arcsec]")
            ax3.set_xlabel("Helioprojective Longitude [arcsec]")
            ax3.set_title("Bulk Velocity Flow")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title("Electron Number Density")

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(self.temp, cmap="hot")
            ax2.set_ylabel("y [pixels]")
            ax2.set_xlabel("x [pixels]")
            ax2.set_title("Electron Temperature")

            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax3.set_ylabel("y [pixels]")
            ax3.set_xlabel("x [pixels]")
            ax3.set_title("Bulk Velocity Flow")
            fig.show()


    def to_lonlat(self, y, x):
        """
        This function will take a y, x coordinate in pixel space and map it to Helioprojective Longitude, Helioprojective Latitude according to the transform in the WCS. This will return the Helioprojective coordinates in units of arcseconds. Note this function takes arguments in the order of numpy indexing (y,x) but returns a pair longitude/latitude which is Solar-X, Solar-Y.

        Parameters
        ----------
        y : int
            The y-index to be converted to Helioprojective Latitude.
        x : int
            The x-index to be converted to Helioprojective Longitude.
        """
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
        """
        This function takes a Helioprojective Longitude, Helioprojective Latitude pair and converts them to the y, x indices to index the object correctly. The function takes its arguments in the order Helioprojective Longitude, Helioprojective Latitude but returns the indices in the (y,x) format so that the output of this function can be used to directly index the object.

        Parameters
        ----------
        lon : float
            The Helioprojective Longitude in arcseconds.
        lat : float
            The Helioprojective Latitude in arcseconds.
        """
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
                return self.wcs[0,0].world_to_array_index(lon,lat)
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
                    return self.wcs[0].world_to_array_index(lon,lat)
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            return self.wcs.world_to_array_index(lon,lat)
        else:
            raise NotImplementedError("Too many or too little dimensions.")