import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import h5py, yaml, html
import matplotlib.patches as patches
from matplotlib.colors import SymLogNorm
import astropy.units as u
from .crisp import CRISP, CRISPSequence, CRISPWideband, CRISPWidebandSequence
from .inversions import Inversion
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
            self.coords.append(px)
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
        plt.style.use("ggplot")
        if type(files) == CRISPWidebandSequence:
            self.cube = files
        elif type(files) == list and type(files[0]) == dict:
            self.cube = CRISPWidebandSequence(files)
        elif type(files) == list and type(files[0]) == str:
            files = [{"file" : f} for f in files]
            self.cube = CRISPWidebandSequence(files)
        elif type(files) == list and type(files[0]) == CRISPWideband:
            self.cube = files
        if type(self.cube) is not list:
            self.time = [f.file.header.get("DATE-AVG")[-12:] for f in self.cube.list]
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(1, 2, 1, projection=self.cube.list[0].wcs)
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            self.ax2.yaxis.set_label_position("right")
            self.ax2.yaxis.ticks_right()
            self.ax2.set_ylabel("I [DNs]")
            self.ax2.set_xlabel("Time [UTC]")
            self.ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.ax2.tick_params(direction="in")

            t = widgets.IntSlider(value=0, min=0, max=len(self.cube.list), step=1, description="Time index: ", style={"description_width" : "initial"})

            widgets.interact(self._img_plot1, t = t)
        else:
            self.time1 = [f.file.header.get("DATE-AVG")[-12:] for f in self.cube[0].list]
            self.time2 = [f.file.header.get("DATE-AVG")[-12:] for f in self.cube[1].list]
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(2, 2, 1, projection=self.cube[0].list[0].wcs)
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax1.xaxis.set_label_position("top")
            self.ax1.xaxis.tick_top()

            self.ax2 = self.fig.add_subplot(2, 2, 3, projection=self.cube[1].list[0].wcs)
            self.ax2.ylabel("Helioprojective Latitude [arcsec]")
            self.ax2.xlabel("Helioprojective Longitude [arcsec]")

            self.ax3 = self.fig.add_subplot(2, 2, 2)
            self.ax3.yaxis.set_label_position("right")
            self.ax3.yaxis.tick_right()
            self.ax3.set_ylabel("I [DNs]")
            self.ax3.set_xlabel("Time [UTC]")
            self.ax3.xaxis.set_label_position("top")
            self.ax3.xaxis.tick_top()
            self.ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.ax3.tick_params(direction="in")

            self.ax4 = self.fig.add_subplot(2, 2, 4)
            self.ax4.yaxis.set_label_position("right")
            self.ax4.yaxis.tick_right()
            self.ax4.set_ylabel("I [DNs]")
            self.ax4.set_xlabel("Time [UTC]")
            self.ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.ax4.tick_params(direction="in")

            t1 = widgets.IntSlider(value=0, min=0, max=len(self.cube[0].list), step=1, description="Time index: ", style={"description_width" : "initial"})
            t2 = widgets.IntSlider(value=0, min=0, max=len(self.cube[1].list), step=1, description="Time index: ", style={"description_width" : "initial"})

            widgets.interact(self._img_plot2, t1=t1, t2=t2)

        self.coords = []
        self.px_coords = []
        self.colour_idx = 0

        self.reveiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

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

        if type(self.cube) == CRISPWidebandSequence:
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
            px = self.cube.list[0].wcs.array_index_to_world(*centre_coord) << u.arcsec
            prof = [f.file.data[centre_coord[0], centre_coord[1]] for f in self.cube.list]
            self.ax2.plot(self.time, prof, marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax2.legend()
            self.coords.append(px)
            self.colour_idx += 1
            self.fig.canvas.draw()
        elif type(self.cube) == list:
            centre_coord = int(event.ydata), int(event.xdata)
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
            px = self.cube[0].list[0].wcs.array_index_to_world(*centre_coord) << u.arcsec
            prof_1 = [f.file.data[centre_coord[0], centre_coord[1]] for f in self.cube[0].list]
            prof_2 = [f.file.data[centre_coord[0], centre_coord[1]] for f in self.cube[1].list]
            self.ax3.plot(self.time1, prof_1, marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax4.plot(self.time2, prof_2, marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
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
        if type(self.cube) == CRISPWidebandSequence:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.texts:
                    t.remove()
            self.ax2.clear()
            self.ax2.set_ylabel("I [DNs]")
            self.ax2.set_xlabel("Time [UTC]")
            self.ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
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
                for t in self.ax1.patches:
                    t.remove()
            while len(self.ax2.patches) > 0:
                for t in self.ax2.patches:
                    t.remove()
            self.ax3.clear()
            self.ax3.set_ylabel("I [DNs]")
            self.ax3.set_xlabel("Time [UTC]")
            self.ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.ax4.clear()
            self.ax4.set_ylabel("I [DNs]")
            self.ax4.set_xlabel("Time [UTC]")
            self.ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn

    def _img_plot1(self, t):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        im1 = self.ax1.imshow(self.cube.list[t].file.data, cmap="Greys_r")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="I [DNs]")

    def _img_plot2(self, t1, t2):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar is not None:
            self.ax2.images[-1].colorbar.remove()

        im1 = self.ax1.imshow(self.cube[0].list[t].file.data, cmap="Greys_r")
        im2 = self.ax2.imshow(self.cube[1].list[t].file.data, cmap="Greys_r")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="I [DNs]")
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label="I [DNs]")

class AtmosViewer:
    def __init__(self, filename, z=None, wcs=None, header=None, eb=False):
        if type(filename) == str:
            assert z is not None
            assert wcs is not None
            self.inv = Inversion(filename=filename, wcs=wcs, z=z, header=header)
        elif type(filename) == Inversion:
            self.inv = filename

        self.coords = []
        self.px_coords = []
        self.colour_idx = 0
        self.eb = eb

        self.fig = plt.figure(figsize=(8,10))
        self.gs = self.fig.add_gridspec(nrows=5, ncols=3)

        self.ax1 = self.fig.add_subplot(self.gs[:2, 0])
        self.ax2 = self.fig.add_subplot(self.gs[:2, 1])
        self.ax3 = self.fig.add_subplot(self.gs[:2, 2])
        self.ax2.tick_params(labelleft=False)
        self.ax3.tick_params(labelleft=False)
        
        self.ax4 = self.fig.add_subplot(self.gs[2, :])
        self.ax4.set_ylabel(r"log $n_{e}$ [cm$^{-3}$]")
        self.ax4.yaxis.set_label_position("right")
        self.ax4.yaxis.tick_right()
        
        self.ax5 = self.fig.add_subplot(self.gs[3, :])
        self.ax5.set_ylabel(r"log T [K]")
        self.ax5.yaxis.set_label_position("right")
        self.ax5.yaxis.tick_right()
        
        self.ax6 = self.fig.add_subplot(self.gs[4, :])
        self.ax6.set_ylabel(r"v [km s$^{-1}$]")
        self.ax6.set_xlabel(r"z [Mm]")
        self.ax6.yaxis.set_label_position("right")
        self.ax6.yaxis.tick_right()
        
        self.ax4.tick_params(labelbottom=False, direction="in")
        self.ax5.tick_params(labelbottom=False, direction="in")
        self.ax6.tick_params(direction="in")
        
        widgets.interact(self._img_plot,
                        z = widgets.SelectionSlider(options=np.round(self.inv.z, decimals=2), description="Image height [Mm]: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="75%")))
        
        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        
        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description='Clear')
        clear_button.on_click(self._clear)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, clear_button, save_button]))
        widgets.interact(self._file_name, fn = widgets.Text(description="Filename to save as: ", style={"description_width" : "initial"}), layout=widgets.Layout(width="50%"))
        
    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode is not "":
            return
        centre_coord = int(event.ydata), int(event.xdata)
        self.px_coords.append(centre_coord)
        circ1 = patches.Circle(centre_coord[::-1], radius=10, facecolor=f"C{self.colour_idx}", edgecolor="k", linewidth=1)
        circ2 = patches.Circle(centre_coord[::-1], radius=10, facecolor=f"C{self.colour_idx}", edgecolor="k", linewidth=1)
        circ3 = patches.Circle(centre_coord[::-1], radius=10, facecolor=f"C{self.colour_idx}", edgecolor="k", linewidth=1)
        self.ax1.add_patch(circ1)
        self.ax2.add_patch(circ2)
        self.ax3.add_patch(circ3)
        font = {
            "size" : 12,
            "color" : f"C{self.colour_idx}"
        }
        txt_1 = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1}", fontdict=font)
        txt_2 = self.ax2.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1}", fontdict=font)
        txt_3 = self.ax3.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1}", fontdict=font)
        txt_1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
        txt_2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
        txt_3.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
        if self.eb:
            self.ax4.errorbar(self.z, self.ne[*centre_coord], yerr=self.err[*centre_coord,0], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax5.errorbar(self.z, self.temp[*centre_coord], yerr=self.err[*centre_coord,1], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax6.errorbar(self.z, self.vel[*centre_coord], yerr=self.err[*centre_coord,2], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
        else:
            self.ax4.plot(self.z, self.ne[*centre_coord], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax5.plot(self.z, self.temp[*centre_coord], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
            self.ax6.plot(self.z, self.vel[*centre_coord], marker=Line2D.filled_markers[self.colour_idx], label=f"{self.colour_idx+1}")
        self.ax4.legend()
        self.ax5.legend()
        self.ax6.legend()
        px = self.inv.wcs.array_index_to_world(*centre_coord) << u.arcsec
        self.colour_idx += 1
        self.coords.append(px)
        self.fig.canvas.draw()
        
    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)
        
    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        self.colour_idx = 0
        while len(self.ax1.patches) > 0:
            for p in self.ax1.patches:
                p.remove()
        while len(self.ax2.patches) > 0:
            for p in self.ax2.patches:
                p.remove()
        while len(self.ax3.patches) > 0:
            for p in self.ax3.patches:
                p.remove()
        while len(self.ax1.texts) > 0:
            for t in self.ax1.texts:
                t.remove()
        while len(self.ax2.texts) > 0:
            for t in self.ax2.texts:
                t.remove()
        while len(self.ax3.texts) > 0:
            for t in self.ax3.texts:
                t.remove()
        self.ax4.clear()
        self.ax4.set_ylabel(r"log n$_{e}$ [cm$^{-3}$]")
        self.ax5.clear()
        self.ax5.set_ylabel(r"log T [K]")
        self.ax6.clear()
        self.ax6.set_ylabel(r"v [km s$^{-1}$]")
        self.ax6.set_xlabel(r"z [Mm]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn
            
    def _img_plot(self, z):
        z_r = np.round(z, decimals=2)
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()
        im1 = self.ax1.imshow(self.ne[:,:,np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), cmap="cividis")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label=r"log $n_{e}$ [cm$^{-3}$]")

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar is not None:
            self.ax2.images[-1].colorbar.remove()
        im2 = self.ax2.imshow(self.temp[:,:,np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), cmap="hot")
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label=r"log T [K]")

        if self.ax3.images == []:
            pass
        elif self.ax3.images[-1].colorbar is not None:
            self.ax3.images[-1].colorbar.remove()
        im3 = self.ax3.imshow(self.vel[:,:, np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), cmap="RdBu", norm=SymLogNorm(1), clim=(-np.max(self.file_obj["vel"][:, np.argwhere(np.round(self.z, decimals=2) == z)]), np.max(self.file_obj["vel"][:,np.argwhere(np.round(self.z, decimals=2) == z)])))
        self.fig.colorbar(im3, ax=self.ax3, orientation="horizontal", label=r"v [km s$^{-1}$]")