import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import h5py, yaml, html
import matplotlib.patches as patches
from matplotlib.colors import SymLogNorm
import astropy.units as u
from .crisp import CRISP, CRISPSequence
from matplotlib import ticker
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D

class SpectralViewer:
    def __init__(self, data, wcs=None, uncertainty=None, mask=None):
        self.aa = html.unescape("&#8491;")
        self.l = html.unescape("&lambda;")
        self.a = html.unescape("&alpha;")
        self.D = html.unescape("&Delta;")
        if type(data) == str:
            self.cube = CRISP(file=data, wcs=wcs, uncertainty=uncertainty, mask=mask)
            self.wvls = self.cube.wcs.array_index_to_world(np.arange(self.cube.file.data.shape[0])) << u.m
            self.wvls <<= u.Angstrom
        elif type(data) == list:
            self.cube = CRISPSequence(list=data)
            self.wvls1 = self.cube.list[0].wcs.array_index_to_world(np.arange(self.cube.list[0].file.data.shape[0])) << u.m
            self.wvls1 <<= u.Angstrom
            self.wvls2 = self.cube.list[1].wcs.array_index_to_world(np.arange(self.cube.list[1].file.data.shape[0])) << u.m
            self.wvls2 <<= u.Angstrom
        elif type(data) == CRISP:
            self.cube = data
            self.wvls = self.cube.wcs.array_index_to_world(np.arange(self.cube.file.data.shape[0])) << u.m
            self.wvls <<= u.Angstrom
        elif type(data) == CRISPSequence:
            self.cube = data
            self.wvls1 = self.cube.list[0].wcs.array_index_to_world(np.arange(self.cube.list[0].file.data.shape[0])) << u.m
            self.wvls1 <<= u.Angstrom
            self.wvls2 = self.cube.list[1].wcs.array_index_to_world(np.arange(self.cube.list[1].file.data.shape[0])) << u.m
            self.wvls2 <<= u.Angstrom

        if type(self.cube) == CRISP:
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(1, 2, 1, projection=self.wcs.dropaxis(-1))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            self.ax2.yaxis.set_label_position("right")
            self.ax2.yaxis.ticks_right()
            self.ax2.set_ylabel("I [DNs]")
            self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax2.tick_params(direction="in")

            ll = widgets.SelectionSlider(options=[np.round(l - np.median())])