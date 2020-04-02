import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import h5py, yaml, html
import matplotlib.patches as patches
from matplotlib.colors import SymLogNorm
import astropy.units as u
from .crisp import CRISP, CRISPSequence, CRISPWidebandSequence
from matplotlib import ticker
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D

class SpectralViewer:
    def __init__(self, data, wcs=None, uncertainty=None, mask=None):
        plt.style.use("ggplot")
        self.aa = html.unescape("&#8491;")
        self.l = html.unescape("&lambda;")
        self.a = html.unescape("&alpha;")
        self.D = html.unescape("&Delta;")
        if type(data) == str:
            self.cube = CRISP(file=data, wcs=wcs, uncertainty=uncertainty, mask=mask)
            self.wvls = self.cube.wcs.all_pix2world([0.], [0.], np.arange(self.cube.file.data.shape[0]),0)[2] << u.m
            self.wvls <<= u.Angstrom
        elif type(data) == list:
            self.cube = CRISPSequence(list=data)
            self.wvls1 = self.cube.list[0].wcs.all_pix2world([0.], [0.], np.arange(self.cube.list[0].file.data.shape[0]),0)[2] << u.m
            self.wvls1 <<= u.Angstrom
            self.wvls2 = self.cube.list[1].wcs.all_pix2world([0.], [0.], np.arange(self.cube.list[1].file.data.shape[0]),0)[2] << u.m
            self.wvls2 <<= u.Angstrom
        elif type(data) == CRISP:
            self.cube = data
            self.wvls = self.cube.wcs.all_pix2world([0.], [0.], np.arange(self.cube.file.data.shape[0]),0)[2] << u.m
            self.wvls <<= u.Angstrom
        elif type(data) == CRISPSequence:
            self.cube = data
            self.wvls1 = self.cube.list[0].wcs.all_pix2world([0.], [0.], np.arange(self.cube.list[0].file.data.shape[0]),0)[2] << u.m
            self.wvls1 <<= u.Angstrom
            self.wvls2 = self.cube.list[1].wcs.all_pix2world([0.], [0.], np.arange(self.cube.list[1].file.data.shape[0]), 0)[2] << u.m
            self.wvls2 <<= u.Angstrom

        if type(self.cube) == CRISP:
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(1, 2, 1, projection=self.cube.wcs.dropaxis(-1))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            self.ax2.yaxis.set_label_position("right")
            self.ax2.yaxis.ticks_right()
            self.ax2.set_ylabel("I [DNs]")
            self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax2.tick_params(direction="in")

            ll = widgets.SelectionSlider(options=[np.round(l - np.median(self.wvls), decimals=2).value for l in self.wvls], description = f"{self.D} {self.l} [{self.aa}]")

            widgets.interact(self._img_plot1, ll = ll)
        elif type(self.cube) == CRISPSequence:
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(2, 2, 1, projection=self.cube.list[0].wcs.dropaxis(-1))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax1.xaxis.set_label_position("top")
            self.ax1.xaxis.tick_top()

            self.ax2 = self.fig.add_subplot(2, 2, 3, projection=self.cube.list[1].wcs.dropaxis(-1))
            self.ax2.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax2.set_xlabel("Helioprojective Longitude [arcsec]")

            self.ax3 = self.fig.add_subplot(2, 2, 2)
            self.ax3.yaxis.set_label_position("right")
            self.ax3.yaxis.tick_right()
            self.ax3.set_ylabel("Intensity [DNs]")
            self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax3.xaxis.set_label_position("top")
            self.ax3.xaxis.tick_top()
            self.ax3.tick_params(direction="in")

            self.ax4 = self.fig.add_subplot(2, 2, 4)
            self.ax4.yaxis.set_label_position("right")
            self.ax4.yaxis.tick_right()
            self.ax4.set_ylabel("Intensity [DNs]")
            self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax4.tick_params(direction="in")

            ll1 = widgets.SelectionSlider(
                options=[np.round(l - np.median(self.wvls1), decimals=2).value for l in self.wvls1],
                description=self.cube.list[0].file.header["WDESC1"]+f"{self.aa} {self.D} {self.l} [{self.aa}]",
                style={"description_width" : "initial"}
            )
            ll2 = widgets.SelectionSlider(
                options=[np.round(l - np.median(self.wvls2), decimals=2).value for l in self.wvls2],
                description=self.cube.list[1].file.header["WDESC1"]+f"{self.aa} {self.D} {self.l} [{self.aa}]",
                style={"description_width" : "initial"}
            )

            widgets.interact(self._img_plot2, ll1=ll1, ll2=ll2)

        self.coords = []
        self.px_coords = []
        self.colour_idx = 0

        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description="Clear")
        clear_button.on_click(self._clear)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, clear_button, save_button]))
        widgets.interact(self._file_name, fn= widgets.Text(description="Filename to save as: "), style={"description_width" : "initial"}, layout=widgets.Layout(width="50%"))

    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode is not "":
            return

        if type(self.cube) == CRISP:
            centre_coord = int(event.ydata), int(event.xdata)
            self.px_coords.append(centre_coord)
            circ = patches.Circle(centre_coord[::-1], radius=10, facecolor=f"C{self.colour_idx}", edgecolor="k", linewidth=1)
            self.ax1.add_patch(circ)
            font = {
                "size" : 12,
                "color" : f"C{self.colour_idx}"
            }
            txt = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1}", fontdict=font)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            px = self.cube.wcs.array_index_to_world(*centre_coord) << u.arcsec
            self.ax2.plot(self.wvls, self.cube.file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax2.legend()
            self.px_coords.append(px)
            self.colour_idx += 1
            self.fig.canvas.draw()
        elif type(self.cube) == CRISPSequence:
            centre_coord = int(event.ydata), int(event.xdata) #with WCS, the event data is returned in pixels so we don't need to do the conversion from real world but rather to real world later on
            self.px_coords.append(centre_coord)
            circ1 = patches.Circle(centre_coord[::-1], radius=10, facecolor=f"C{self.colour_idx}", edgecolor="k", linewidth=1)
            circ2 = patches.Circle(centre_coord[::-1], radius=10, facecolor=f"C{self.colour_idx}", edgecolor="k", linewidth=1)
            self.ax1.add_patch(circ1)
            self.ax2.add_patch(circ2)
            font = {
                "size" : 12,
                "color" : f"C{self.colour_idx}"
            }
            txt_1 = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1}", fontdict=font)
            txt_2 = self.ax2.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1}", fontdict=font)
            txt_1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            txt_2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            px = self.cube.list[0].wcs.dropaxis(-1).array_index_to_world(*centre_coord) << u.arcsec
            self.ax3.plot(self.wvls1, self.cube.list[0].file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax4.plot(self.wvls2, self.cube.list[1].file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax3.legend()
            self.ax4.legend()
            self.coords.append(px)
            self.colour_idx += 1
            self.fig.canvas.draw()

    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)

    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        self.colour_idx = 0
        if type(self.cube) == CRISP:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.texts:
                    t.remove()
            self.ax2.clear()
            self.ax2.set_ylabel("Intensity [DNs]")
            self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax2.patches) > 0:
                for p in self.ax2.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.texts:
                    t.remove()
            while len(self.ax2.texts) > 0:
                for t in self.ax2.texts:
                    t.remove()
            self.ax3.clear()
            self.ax3.set_ylabel("Intensity [DNs]")
            self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax4.clear()
            self.ax4.set_ylabel("Intensity [DNs]")
            self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn

    def _img_plot1(self, ll):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        ll_idx = int(np.where(np.round(self.wvls, decimals=2).value == np.round(np.median(self.wvls).value + ll, decimals=2))[0])

        im1 = self.ax1.imshow(self.cube.file.data[ll_idx], cmap="Greys_r")
        self.ax1.set_title(self.cube.file.header["WDESC1"]+f"{self.aa} {self.D} {self.l} = {ll} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")

    def _img_plot2(self, ll1, ll2):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar is not None:
            self.ax2.images[-1].colorbar.remove()

        ll1_idx = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + ll1, decimals=2))[0])
        ll2_idx = int(np.where(np.round(self.wvls2, decimals=2).value == np.round(np.median(self.wvls2).value + ll2, decimals=2))[0])

        im1 = self.ax1.imshow(self.cube.list[0].file.data[ll1_idx], cmap="Greys_r")
        im2 = self.ax2.imshow(self.cube.list[1].file.data[ll2_idx], cmap="Greys_r")
        self.ax1.set_title(self.cube.list[0].file.header["WDESC1"]+f"{self.aa} {self.D} {self.l} = {ll1} {self.aa}")
        self.ax2.set_title(self.cube.list[1].file.header["WDESC1"]+f"{self.aa} {self.D} {self.l} = {ll2} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label="Intensity [DNs]")

class WidebandViewer:
    def __init__(self, files):
        if type(files) == CRISPWidebandSequence:
            self.list = files
        elif type(files) == list and type(files[0]) == dict:
            self.list = CRISPWidebandSequence(files)
        elif type(files) == list and type(files[0]) == str:
            files = [{"file" : f} for f in files]
            self.list = CRISPWidebandSequence(files)