import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import SymLogNorm
from .mixin import InversionSlicingMixin
from .utils import ObjDict

class Inversion(InversionSlicingMixin):
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
            time = self.file.header.get("DATE-AVG")[-12:]
            date = self.file.header.get("DATE-AVG")[:-13]
            pointing_x = str(self.file.header.get("CRVAL1"))
            pointing_y = str(self.file.header.get("CRVAL2"))

            return f"""Inversion
            ------------------
            {date} {time}

            Pointing: ({pointing_x}, {pointing_y})"""

    def __repr__(self):
        return self.__str__()

    def plot_ne(self):
        fig = plt.figure()
        ax1 = fig.gca()
        ax1.plot(self.z, self.ne)
        ax1.set_ylabel(r"log$_{10}$ n$_{\text{e}}$ \[cm$^{-3}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title("Electron Number Density")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_temp(self):
        fig = plt.figure()
        ax1 = fig.gca()
        ax1.plot(self.z, self.temp)
        ax1.set_ylabel(r"log$_{10}$ T \[K\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title("Electron Temperature")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_vel(self):
        fig = plt.figure()
        ax1 = fig.gca()
        ax1.plot(self.z, self.vel)
        ax1.set_ylabel(r"Bulk Plasma Flow \[km s$^{-1}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title("Bulk Plasma Flow")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_params(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(self.z, self.ne)
        ax1.set_ylabel(r"log$_{10}$ n$_{e}$ \[cm$^{-3}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title("Electron Number Density")
        ax1.tick_params(direction="in")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(self.z, self.temp)
        ax2.set_ylabel(r"log$_{10}$ T \[K\]")
        ax2.set_xlabel("z [Mm]")
        ax2.set_title("Electron Temperature")
        ax2.tick_params(direction="in")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(self.z, self.vel)
        ax3.set_ylabel(r"Bulk Plasma Flow \[km s$^{-1}\]")
        ax3.set_xlabel("z [Mm]")
        ax3.set_title("Bulk Plasma Flow")
        ax3.tick_params(direction="in")
        fig.show()

    def ne_map(self, frame=None):
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs)
            ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title("Electron Number Density")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title("Electron Number Density")
            fig.show()

    def temp_map(self, frame=None):
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs)
            ax1.imshow(self.temp, cmap="hot")
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title("Electron Temperature")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            ax1.imshow(self.temp, cmap="cividis")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title("Electron Temperature")
            fig.show()

    def vel_map(self, frame=None):
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs)
            ax1.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title("Bulk Velocity Flow")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            ax1.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title("Bulk Velocity Flow")
            fig.show()

    def params_map(self, frame=None):
        if frame is None:
            fig = plt.figure()
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
