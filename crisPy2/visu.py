import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import h5py, yaml, html
import matplotlib.patches as patches
from matplotlib.colors import SymLogNorm
from IPython.core.display import display
from .utils import ObjDict
from astropy.io import fits
from .crisp import CRISP
import astropy.units as u

class AtmosViewer:
    def __init__(self, file_obj, z, eb=False):
        self.file_obj = file_obj
        if type(z) != str:
            self.z = z
        else:
            self.z = h5py.File(z, "r").get("z")
            
        self.coords = []
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
        self.ax4.grid()
        
        self.ax5 = self.fig.add_subplot(self.gs[3, :])
        self.ax5.set_ylabel(r"log T [K]")
        self.ax5.yaxis.set_label_position("right")
        self.ax5.yaxis.tick_right()
        self.ax5.grid()
        
        self.ax6 = self.fig.add_subplot(self.gs[4, :])
        self.ax6.set_ylabel(r"v [km s$^{-1}$]")
        self.ax6.set_xlabel(r"z [Mm]")
        self.ax6.yaxis.set_label_position("right")
        self.ax6.yaxis.tick_right()
        self.ax6.grid()
        
        self.ax4.tick_params(labelbottom=False, direction="in")
        self.ax5.tick_params(labelbottom=False, direction="in")
        self.ax6.tick_params(direction="in")
        
        widgets.interact(self._img_plot,
                        z = widgets.SelectionSlider(options=np.round(self.z, decimals=2), description="Image height [Mm]: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="75%")))
        
        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        
        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description='Clear')
        clear_button.on_click(self._clear)
        display(widgets.HBox([done_button, clear_button]))
        
    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode is not "":
            return
        centre_coord = int(event.xdata), int(event.ydata)
        circ1 = patches.Circle(centre_coord, color="r")
        circ2 = patches.Circle(centre_coord, color="r")
        circ3 = patches.Circle(centre_coord, color="r")
        self.ax1.add_patch(circ1)
        self.ax2.add_patch(circ2)
        self.ax3.add_patch(circ3)
        if self.eb:
            self.ax4.errorbar(self.z, self.file_obj["ne"][event.xdata*event.ydata], yerr=self.file_obj["mad"][event.xdata*event.ydata,0])
            self.ax5.errorbar(self.z, self.file_obj["temperature"][event.xdata*event.ydata], yerr=self.file_obj["mad"][event.xdata*event.ydata,1])
            self.ax6.errorbar(self.z, self.file_obj["vel"][event.xdata*event.ydata], yerr=self.file_obj["mad"][event.xdata*event.ydata,2])
        else:
            self.ax4.plot(self.z, self.file_obj["ne"][event.xdata*event.ydata])
            self.ax5.plot(self.z, self.file_obj["temperature"][event.xdata*event.ydata])
            self.ax6.plot(self.z, self.file_obj["vel"][event.xdata*event.ydata])
        self.coords.append((int(event.ydata), int(event.xdata)))
        self.fig.canvas.draw()
        
    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)
        
    def _clear(self, _):
        self.coords = []
        while len(self.ax1.patches) > 0:
            for p in self.ax1.patches:
                p.remove()
        while len(self.ax2.patches) > 0:
            for p in self.ax2.patches:
                p.remove()
        while len(self.ax3.patches) > 0:
            for p in self.ax3.patches:
                p.remove()
        self.ax4.clear()
        self.ax4.grid()
        self.ax4.set_ylabel(r"log n$_{e}$ [cm$^{-3}$]")
        self.ax5.clear()
        self.ax5.grid()
        self.ax5.set_ylabel(r"log T [K]")
        self.ax6.clear()
        self.ax6.grid()
        self.ax6.set_ylabel(r"v [km s$^{-1}$]")
        self.ax6.set_xlabel(r"z [Mm]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
            
    def _img_plot(self, z):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()
        im1 = self.ax1.imshow(self.file_obj["ne"][:,np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), origin="lower", cmap="cividis")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label=r"log $n_{e}$ [cm$^{-3}$]")

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar is not None:
            self.ax2.images[-1].colorbar.remove()
        im2 = self.ax2.imshow(self.file_obj["temperature"][:,np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), origin="lower", cmap=sol_cm)
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label=r"log T [K]")

        if self.ax3.images == []:
            pass
        elif self.ax3.images[-1].colorbar is not None:
            self.ax3.images[-1].colorbar.remove()
        im3 = self.ax3.imshow(self.file_obj["vel"][:, np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), origin="lower", cmap="RdBu", norm=SymLogNorm(0.01), clim=(-np.max(self.file_obj["vel"][:, np.argwhere(np.round(self.z, decimals=2) == z)]), np.max(self.file_obj["vel"][:,np.argwhere(np.round(self.z, decimals=2) == z)])))
        self.fig.colorbar(im3, ax=self.ax3, orientation="horizontal", label=r"v [km s$^{-1}$]")

class TimeViewer:
    def __init__(self, file_list, z, eb=False):
        self.file_list = file_list
        self.file_obj = self.file_list[467]
        if type(z) != str:
            self.z = z
        else:
            self.z = h5py.File(z, "r").get("z")
            
        self.coords = []
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
        self.ax4.grid()
        
        self.ax5 = self.fig.add_subplot(self.gs[3, :])
        self.ax5.set_ylabel(r"log T [K]")
        self.ax5.yaxis.set_label_position("right")
        self.ax5.yaxis.tick_right()
        self.ax5.grid()
        
        self.ax6 = self.fig.add_subplot(self.gs[4, :])
        self.ax6.set_ylabel(r"v [km s$^{-1}$]")
        self.ax6.set_xlabel(r"t")
        self.ax6.yaxis.set_label_position("right")
        self.ax6.yaxis.tick_right()
        self.ax6.grid()
        
        self.ax4.tick_params(labelbottom=False, direction="in")
        self.ax5.tick_params(labelbottom=False, direction="in")
        self.ax6.tick_params(direction="in")
        
        widgets.interact(self._img_plot,
                        z = widgets.SelectionSlider(options=np.round(self.z, decimals=2), description="Image height [Mm]: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="50%")))
        
        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        
        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description='Clear')
        clear_button.on_click(self._clear)
        display(widgets.HBox([done_button, clear_button]))
        
    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode is not "":
            return
        centre_coord = int(event.xdata), int(event.ydata)
        circ1 = patches.Circle(centre_coord, color="r")
        circ2 = patches.Circle(centre_coord, color="r")
        circ3 = patches.Circle(centre_coord, color="r")
        self.ax1.add_patch(circ1)
        self.ax2.add_patch(circ2)
        self.ax3.add_patch(circ3)
        if self.eb:
            self.ax4.errorbar(range(len(self.file_list)), [x["ne"][event.xdata*event.ydata, 20] for x in self.file_list], yerr=[x["mad"][event.xdata*event.ydata, 0, 20] for x in self.file_list])
            self.ax5.errorbar(range(len(self.file_list)), [x["temperature"][event.xdata*event.ydata, 20] for x in self.file_list], yerr=[x["mad"][event.xdata*event.ydata, 1, 20] for x in self.file_list])
            self.ax6.errorbar(range(len(self.file_list)), [x["vel"][event.xdata*event.ydata, 20] for x in self.file_list], yerr=[x["mad"][event.xdata*event.ydata, 2, 20] for x in self.file_list])
        else:
            self.ax4.plot(range(len(self.file_list)), [x["ne"][event.xdata*event.ydata, 20] for x in self.file_list])
            self.ax5.plot(range(len(self.file_list)), [x["temperature"][event.xdata*event.ydata, 20] for x in self.file_list])
            self.ax6.plot(range(len(self.file_list)), [x["vel"][event.xdata*event.ydata, 20] for x in self.file_list])
        self.coords.append((int(event.ydata), int(event.xdata)))
        self.fig.canvas.draw()
        
    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)
        
    def _clear(self, _):
        self.coords = []
        while len(self.ax1.patches) > 0:
            for p in self.ax1.patches:
                p.remove()
        while len(self.ax2.patches) > 0:
            for p in self.ax2.patches:
                p.remove()
        while len(self.ax3.patches) > 0:
            for p in self.ax3.patches:
                p.remove()        
        self.ax4.clear()
        self.ax4.grid()
        self.ax4.set_ylabel(r"log n$_{e}$ [cm$^{-3}$]")
        self.ax5.clear()
        self.ax5.grid()
        self.ax5.set_ylabel(r"log T [K]")
        self.ax6.clear()
        self.ax6.grid()
        self.ax6.set_ylabel(r"v [km s$^{-1}$]")
        self.ax6.set_xlabel(r"z [Mm]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
            
    def _img_plot(self, z):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()
        im1 = self.ax1.imshow(self.file_obj["ne"][:,np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), origin="lower", cmap="cividis")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label=r"log $n_{e}$ [cm$^{-3}$]")

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar is not None:
            self.ax2.images[-1].colorbar.remove()
        im2 = self.ax2.imshow(self.file_obj["temperature"][:,np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), origin="lower", cmap=sol_cm)
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label=r"log T [K]")

        if self.ax3.images == []:
            pass
        elif self.ax3.images[-1].colorbar is not None:
            self.ax3.images[-1].colorbar.remove()
        im3 = self.ax3.imshow(self.file_obj["vel"][:, np.argwhere(np.round(self.z, decimals=2) == z)].reshape(840,840), origin="lower", cmap="RdBu", norm=SymLogNorm(0.01), clim=(-np.max(self.file_obj["vel"][:, np.argwhere(np.round(self.z, decimals=2) == z)]), np.max(self.file_obj["vel"][:,np.argwhere(np.round(self.z, decimals=2) == z)])))
        self.fig.colorbar(im3, ax=self.ax3, orientation="horizontal", label=r"v [km s$^{-1}$]")

class SpectralViewer:
    def __init__(self, data, hc=False):
        self.hc = hc
        self.aa = html.unescape("&#8491;")
        self.l = html.unescape("&lambda;")
        self.a = html.unescape("&alpha;")
        self.D = html.unescape("&Delta;")
        if type(data) == str:
            self.file = CRISP(files=data)
            if "8542" in data:
                self.wvls = self.file.ca_wvls
            else:
                self.wvls = self.file.ha_wvls
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(1, 2, 1)
            self.ax1.set_ylabel("y [arcseconds]")
            self.ax1.set_xlabel("x [arcseconds]")
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            self.ax2.yaxis.set_label_position("right")
            self.ax2.yaxis.tick_right()
            self.ax2.set_ylabel("Intensity [DNs]")
            self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax2.grid()
            self.ax2.tick_params(direction="in")
            
            ll = widgets.SelectionSlider(options=[np.round(l - np.median(self.wvls), decimals=2) for l in self.wvls], description = f"{self.D} {self.l} [{self.aa}]")

            widgets.interact(self._img_plot1, ll = ll)
        elif type(data) == CRISP:
            self.file = data
            
        elif type(data) == list:
            self.file = CRISP(files=data)
            self.ca_wvls = self.file.ca_wvls
            self.ha_wvls = self.file.ha_wvls
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(2, 2, 1)
            self.ax1.set_ylabel("y [arcseconds]")
            self.ax1.tick_params(labelbottom=False)
            self.ax2 = self.fig.add_subplot(2, 2, 3)
            self.ax2.set_ylabel("y [arcseconds]")
            self.ax2.set_xlabel("x [arcseconds]")
            self.ax3 = self.fig.add_subplot(2, 2, 2)
            self.ax3.yaxis.set_label_position("right")
            self.ax3.yaxis.tick_right()
            self.ax3.set_ylabel("Intensity [DNs]")
            self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax3.grid()
            self.ax3.tick_params(direction="in")
            self.ax4 = self.fig.add_subplot(2, 2, 4)
            self.ax4.yaxis.set_label_position("right")
            self.ax4.yaxis.tick_right()
            self.ax4.set_ylabel("Intensity [DNs]")
            self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax4.grid()
            self.ax4.tick_params(direction="in")
            
            ll1 = widgets.SelectionSlider(options=[np.round(l - np.median(self.ca_wvls), decimals=2) for l in self.ca_wvls], description=f"Ca II {self.D} {self.l} [{self.aa}]")
            ll2 = widgets.SelectionSlider(options=[np.round(l - np.median(self.ha_wvls), decimals=2) for l in self.ha_wvls], description=f"H{self.a} {self.D} {self.l} [{self.aa}]")
            
            widgets.interact(self._img_plot2, ll1 = ll1, ll2 = ll2)
            
                    
        self.coords = []
        self.px_coords = []
        
        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        
        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description='Clear')
        clear_button.on_click(self._clear)
        display(widgets.HBox([done_button, clear_button]))
        
        
    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode is not "":
            return
        
        if "file" in self.__dict__:
            centre_coord = event.xdata, event.ydata
            self.coords.append((event.ydata, event.xdata))
            circ = patches.Circle(centre_coord, radius=0.25, color="r")
            self.ax1.add_patch(circ)
            if self.hc:
                if self.file.header["CRVAL1"] < 0 and self.file.header["CRVAL2"] < 0:
                    y_axis = np.linspace(self.file.header["CRVAL2"]+(self.file.header["CDELT2"]*self.file.data.shape[-2]/2), self.file.header["CRVAL2"]-(self.file.header["CDELT2"]*self.file.data.shape[-2]/2), num=self.file.data.shape[-2]+1)
                    x_axis = np.linspace(self.file.header["CRVAL1"]+(self.file.header["CDELT1"]*self.file.data.shape[-1]/2), self.file.header["CRVAL1"]-(self.file.header["CDELT1"]*self.file.data.shape[-1]/2), num=self.file.data.shape[-1]+1)
                    px_y = int(y_axis[np.abs(y_axis - event.ydata).argmin()])
                    px_x = int(x_axis[np.abs(x_axis - event.xdata).argmin()])
                elif self.file.header["CRVAL1"] > 0 and self.file.header["CRVAL2"] < 0:
                    y_axis = np.linspace(self.file.header["CRVAL2"]+(self.file.header["CDELT2"]*self.file.data.shape[-2]/2), self.file.header["CRVAL2"]-(self.file.header["CDELT2"]*self.file.data.shape[-2]/2), num=self.file.data.shape[-2]+1)
                    x_axis = np.linspace(self.file.header["CRVAL1"]-(self.file.header["CDELT1"]*self.file.data.shape[-1]/2), self.file.header["CRVAL1"]+(self.file.header["CDELT1"]*self.file.data.shape[-1]/2), num=self.file.data.shape[-1]+1)
                    px_y = int(y_axis[np.abs(y_axis - event.ydata).argmin()])
                    px_x = int(x_axis[np.abs(x_axis - event.xdata).argmin()])
                elif self.file.header["CRVAL1"] > 0 and self.file.header["CRVAL2"] > 0:
                    y_axis = np.linspace(self.file.header["CRVAL2"]-(self.file.header["CDELT2"]*self.file.data.shape[-2]/2), self.file.header["CRVAL2"]+(self.file.header["CDELT2"]*self.file.data.shape[-2]/2), num=self.file.data.shape[-2]+1)
                    x_axis = np.linspace(self.file.header["CRVAL1"]-(self.file.header["CDELT1"]*self.file.data.shape[-1]/2), self.file.header["CRVAL1"]+(self.file.header["CDELT1"]*self.file.data.shape[-1]/2), num=self.file.data.shape[-1]+1)
                    px_y = int(y_axis[np.abs(y_axis - event.ydata).argmin()])
                    px_x = int(x_axis[np.abs(x_axis - event.xdata).argmin()])
                elif self.file.header["CRVAL1"] < 0 and self.file.header["CRVAL2"] > 0:
                    y_axis = np.linspace(self.file.header["CRVAL2"]-(self.file.header["CDELT2"]*self.file.data.shape[-2]/2), self.file.header["CRVAL2"]+(self.file.header["CDELT2"]*self.file.data.shape[-2]/2), num=self.file.data.shape[-2]+1)
                    x_axis = np.linspace(self.file.header["CRVAL1"]+(self.file.header["CDELT1"]*self.file.data.shape[-1]/2), self.file.header["CRVAL1"]-(self.file.header["CDELT1"]*self.file.data.shape[-1]/2), num=self.file.data.shape[-1]+1)
                    px_y = int(y_axis[np.abs(y_axis - event.ydata).argmin()])
                    px_x = int(x_axis[np.abs(x_axis - event.xdata).argmin()])
            else:
                px_y = int(event.ydata / self.file.header["CDELT2"])
                px_x = int(event.xdata / self.file.header["CDELT1"])
            self.ax2.plot(self.wvls, self.file.data[:, px_y, px_x])
            self.coords.append((event.ydata, event.xdata))
            self.px_coords.append((px_y, px_x))
            self.fig.canvas.draw()
        else:
            centre_coord = event.xdata, event.ydata
            circ1 = patches.Circle(centre_coord, radius=0.25, color="r")
            circ2 = patches.Circle(centre_coord, radius=0.25, color="r")
            self.ax1.add_patch(circ1)
            self.ax2.add_patch(circ2)
            if self.hc:
                if self.ca.header["CRVAL1"] < 0 and self.ca.header["CRVAL2"] < 0:
                    y_axis = np.linspace(self.ca.header["CRVAL2"]+(self.ca.header["CDELT2"]*self.ca.data.shape[-2]/2), self.ca.header["CRVAL2"]-(self.ca.header["CDELT2"]*self.ca.data.shape[-2]/2), num=self.ca.data.shape[-2]+1)
                    x_axis = np.linspace(self.ca.header["CRVAL1"]+(self.ca.header["CDELT1"]*self.ca.data.shape[-1]/2), self.ca.header["CRVAL1"]-(self.ca.header["CDELT1"]*self.ca.data.shape[-1]/2), num=self.ca.data.shape[-1]+1)
                    px_y = int(y_axis[np.abs(y_axis - event.ydata).argmin()])
                    px_x = int(x_axis[np.abs(x_axis - event.xdata).argmin()])
                elif self.ca.header["CRVAL1"] > 0 and self.ca.header["CRVAL2"] < 0:
                    y_axis = np.linspace(self.ca.header["CRVAL2"]+(self.ca.header["CDELT2"]*self.ca.data.shape[-2]/2), self.ca.header["CRVAL2"]-(self.ca.header["CDELT2"]*self.ca.data.shape[-2]/2), num=self.ca.data.shape[-2]+1)
                    x_axis = np.linspace(self.ca.header["CRVAL1"]-(self.ca.header["CDELT1"]*self.ca.data.shape[-1]/2), self.ca.header["CRVAL1"]+(self.ca.header["CDELT1"]*self.ca.data.shape[-1]/2), num=self.ca.data.shape[-1]+1)
                    px_y = int(y_axis[np.abs(y_axis - event.ydata).argmin()])
                    px_x = int(x_axis[np.abs(x_axis - event.xdata).argmin()])
                elif self.ca.header["CRVAL1"] > 0 and self.ca.header["CRVAL2"] > 0:
                    y_axis = np.linspace(self.ca.header["CRVAL2"]-(self.ca.header["CDELT2"]*self.ca.data.shape[-2]/2), self.ca.header["CRVAL2"]+(self.ca.header["CDELT2"]*self.ca.data.shape[-2]/2), num=self.ca.data.shape[-2]+1)
                    x_axis = np.linspace(self.ca.header["CRVAL1"]-(self.ca.header["CDELT1"]*self.ca.data.shape[-1]/2), self.ca.header["CRVAL1"]+(self.ca.header["CDELT1"]*self.ca.data.shape[-1]/2), num=self.ca.data.shape[-1]+1)
                    px_y = int(y_axis[np.abs(y_axis - event.ydata).argmin()])
                    px_x = int(x_axis[np.abs(x_axis - event.xdata).argmin()])
                elif self.ca.header["CRVAL1"] < 0 and self.ca.header["CRVAL2"] > 0:
                    y_axis = np.linspace(self.ca.header["CRVAL2"]-(self.ca.header["CDELT2"]*self.ca.data.shape[-2]/2), self.ca.header["CRVAL2"]+(self.ca.header["CDELT2"]*self.ca.data.shape[-2]/2), num=self.ca.data.shape[-2]+1)
                    x_axis = np.linspace(self.ca.header["CRVAL1"]+(self.ca.header["CDELT1"]*self.ca.data.shape[-1]/2), self.ca.header["CRVAL1"]-(self.ca.header["CDELT1"]*self.ca.data.shape[-1]/2), num=self.ca.data.shape[-1]+1)
                    px_y = int(y_axis[np.abs(y_axis - event.ydata).argmin()])
                    px_x = int(x_axis[np.abs(x_axis - event.xdata).argmin()])
            else:
                px_y = int(event.ydata / self.ca.header["CDELT2"])
                px_x = int(event.xdata / self.ca.header["CDELT1"])
            self.ax3.plot(self.ca_wvls, self.ca.data[:, px_y, px_x])
            self.ax4.plot(self.ha_wvls, self.ha.data[:, px_y, px_x])
            self.px_coords.append((px_y, px_x))
            self.fig.canvas.draw()
        
    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)
        
    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        if "file" in self.__dict__:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax2.lines) > 0:
                for p in self.ax2.lines:
                    p.remove()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        else:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax2.patches) > 0:
                for p in self.ax2.patches:
                    p.remove()
            while len(self.ax3.lines) > 0:
                for p in self.ax3.lines:
                    p.remove()
            while len(self.ax4.lines) > 0:
                for p in self.ax4.lines:
                    p.remove()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        
    def _img_plot1(self, ll):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        if "ca" in self.file.__dict__:
            if self.hc:
                tr = self.file.unit_conversion(self.file.ca.data.shape[-2:] << u.pix, unit_to="arcsec", centre=True).value
                bl = self.file.unit_conversion((0,0) << u.pix, unit_to="arcsec", centre=True).value
                extent = [bl[1], tr[1], bl[0], tr[0]]
            else:
                tr = self.file.unit_conversion(self.file.ca.data.shape[-2:] << u.pix, unit_to="arcsec").value
                extent = [0, tr[1], 0, tr[0]]
            try:
                ll_idx = int(ll / np.round(self.file.ca.header["CDELT3"], decimals=2) + (self.file.ca.header["CRPIX3"]-1))
            except KeyError:
                ll_idx = int(np.where(np.round(self.wvls, decimals=2) == np.round(np.median(self.wvls) + ll, decimals=2))[0])
            if len(self.file.ca.data.shape) == 4:
                im1 = self.ax1.imshow(self.file.ca.data[0, ll_idx], cmap="Greys_r", extent=extent)
            else:
                im1 = self.ax1.imshow(self.file.ca.data[ll_idx], cmap="Greys_r", extent=extent)
        else:
            if self.hc:
                tr = self.file.unit_conversion(self.file.ha.data.shape[-2:] << u.pix, unit_to="arcsec", centre=True).value
                bl = self.file.unit_conversion((0,0) << u.pix, unit_to="arcsec", centre=True).value
                extent = [bl[1], tr[1], bl[0], tr[0]]
            else:
                tr = self.file.unit_conversion(self.file.ha.data.shape[-2:] << u.pix, unit_to="arcsec").value
                extent = [0, tr[1], 0, tr[0]]
            try:
                ll_idx = int(ll / np.round(self.file.ha.header["CDELT3"], decimals=2) + (self.file.ha.header["CRPIX3"]-1))
            except KeyError:
                ll_idx = int(np.where(np.round(self.wvls, decimals=2) == np.round(np.median(self.wvls) + ll, decimals=2))[0])
            im1 = self.ax1.imshow(self.file.ha.data[ll_idx], cmap="Greys_r", extent=extent)

            
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
            
        if self.hc:
            tr_ca = self.file.unit_conversion(self.file.ca.data.shape[-2:] << u.pix, unit_to="arcsec", centre=True).value
            tr_ha = self.file.unit_conversion(self.file.ha.data.shape[-2:] << u.pix, unit_to="arcsec", centre=True).value
            bl = self.file.unit_conversion((0,0) << u.pix, unit_to="arcsec", centre=True).value

            extent1 = [bl[1], tr_ca[1], bl[0], tr_ca[0]]
            extent2 = [bl[1], tr_ha[1], bl[0], tr_ha[0]]
        else:
            tr_ca = self.file.unit_conversion(self.file.ca.data.shape[-2:] << u.pix, unit_to="arcsec").value
            tr_ha = self.file.unit_conversion(self.file.ha.data.shape[-2:] << u.pix, unit_to="arcsec").value

            extent1 = [0, tr_ca[1], 0, tr_ca[0]]
            extent2 = [0, tr_ha[1], 0, tr_ha[0]]

        try:
            ll1_idx = int(ll1 / np.round(self.ca.header["CDELT3"], decimals=2) + (self.ca.header["CRPIX3"]-1))
            ll2_idx = int(ll2 / np.round(self.ha.header["CDELT3"], decimals=2) + (self.ha.header["CRPIX3"]-1))
        except KeyError:
           ll1_idx = int(np.where(np.round(self.ca_wvls, decimals=2) == np.round(np.median(self.ca_wvls) + ll, decimals=2))[0]) 
           ll2_idx = int(np.where(np.round(self.ha_wvls, decimals=2) == np.round(np.median(self.ha_wvls) + ll, decimals=2))[0]) 
        if len(self.file.ca.data.shape) == 4:
            im1 = self.ax1.imshow(self.file.ca.data[0, ll_idx], cmap="Greys_r", extent=extent)
        else:
            im1 = self.ax1.imshow(self.file.ca.data[ll_idx], cmap="Greys_r", extent=extent)
        im2 = self.ax2.imshow(self.ha.data[ll2_idx], origin="lower", cmap=sol_cm, extent=extent2)
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label="Intensity [DNs]")