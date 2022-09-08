import html
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from .utils import pt_bright

if TYPE_CHECKING:
    from .crisp import CRISP

unicode_runes = {
    "AA": html.unescape("&#8491;"),
    "alpha": html.unescape("&alpha;"),
    "lambda": html.unescape("&lambda;"),
    "Delta": html.unescape("&Delta;"),
}

rc_context_dict = {
    "savefig.bbox": "tight",
    "font.family": "serif",
    "image.origin": "lower",
    "figure.figsize": (10, 6),
    "font.size": 11,
    "font.serif": "New Century Schoolbook",
}


def plot_spectrum_panel_ax(
    ax: matplotlib.axes.Axes,
    wavelength: u.Quantity,
    spectrum: np.ndarray,
    y_label: str,
    d: bool = False,
    set_xlabel: bool = True,
):
    """
    Plot a single one dimensional spectrum panel into ax. Called by `plot_single_spectrum`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    wavelength : astropy.Quantity
        The wavelength axis (x axis).
    spectrum : np.ndarray
        The data to plot.
    y_label : str
        The label to apply to the y axis.
    d : bool, optional
        Whether to plot in :math:`\\Delta \\lambda` space. Default: False.
    set_xlabel : bool, optional
        Whether to apply the label to the x_axis (e.g. for multi-panel plots). Default: True.
    """
    x_axis = wavelength
    x_label = f"{unicode_runes['lambda']} [{unicode_runes['AA']}]"
    if d:
        x_axis = wavelength - np.median(wavelength)
        x_label = (
            f"{unicode_runes['Delta']}{unicode_runes['lambda']} [{unicode_runes['AA']}]"
        )

    ax.plot(x_axis, spectrum, c=pt_bright["blue"], marker="o")
    ax.set_ylabel(y_label)
    if set_xlabel:
        ax.set_xlabel(x_label)


@plt.rc_context(rc_context_dict)
def plot_single_spectrum(
    wavelength: u.Quantity,
    spectrum: np.ndarray,
    title: str,
    y_label: Optional[str] = None,
    d: bool = False,
    fig: Optional[matplotlib.figure.Figure] = None,
):
    """
    Plot the intensity spectrum for a specific coordinate. Configures the figure
    and calls `plots_spectrum_panel_ax`.

    Parameters
    ----------
    wavelength : astropy.Quantity
        The wavelength axis (x axis).
    spectrum : np.ndarray
        The data to plot.
    title : str
        The title to apply to the figure.
    y_label : str
        The label to apply to the y axis.
    d : bool, optional
        Whether to plot in :math:`\\Delta \\lambda` space. Default: False.
    fig : matplotlib.figure.Figure, optional
        An existing figure to be plotted into. One will be created if not provided.
    """
    if fig is None:
        fig = plt.figure()

    if y_label is None:
        y_label = "Intensity [DNs]"

    ax1 = fig.gca()
    plot_spectrum_panel_ax(ax1, wavelength, spectrum, y_label, d=d)
    fig.suptitle(title)
    fig.show()


@plt.rc_context(rc_context_dict)
def plot_multi_component_spectrum(
    components: List[str],
    wavelength: u.Quantity,
    spectrum: np.ndarray,
    title: str,
    d: bool = False,
):
    """
    Plot the Stokes components at a specific location on a multi-panel plot.
    Configures the figure and calls `plots_spectrum_panel_ax`.

    Parameters
    ----------
    components : list of str
        List of the stokes components "I", "Q"... to plot.
    wavelength : astropy.Quantity
        The wavelength axis (x axis).
    spectrum : np.ndarray
        The data to plot. The first axis is expected to be of length 4 (for the
        Stokes components).
    title : str
        The title to apply to the figure.
    d : bool, optional
        Whether to plot in :math:`\\Delta \\lambda` space. Default: False.
    """
    stokes_components = ["I", "Q", "U", "V"]

    num_components = len(components)
    if num_components < 2 or num_components > 4:
        raise ValueError("Number of components should be in range [2, 4]")

    if num_components == 4:
        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()
    else:
        fig, ax = plt.subplots(1, num_components)

    fig.suptitle(title)
    for c, component in enumerate(components):
        set_xlabel = num_components != 4 or c >= 2
        plot_spectrum_panel_ax(
            ax[c],
            wavelength,
            spectrum[stokes_components.index(component)],
            f"{component} [DNs]",
            d=d,
            set_xlabel=set_xlabel,
        )

    if num_components == 4:
        for i in [1, 3]:
            ax[i].yaxis.set_label_position("right")
            ax[i].yaxis.tick_right()
        for i in range(2):
            ax[i].tick_params(labelbottom=False)
    fig.show()


def plot_single_frame_ax(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    im: "CRISP",
    frame: str,
    title: str,
    stokes_index: Optional[int] = None,
    set_xlabel: bool = True,
    set_ylabel: bool = True,
    cb_label: Optional[str] = None,
    extent: Optional[List[int]] = None,
    origin: Optional[str] = None,
    norm: Optional[matplotlib.colors.Normalize] = None,
    map_data: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """
    Plot a single two-dimensional image onto the axis `ax` in figure `fig`.
    Called by `plot_single_frame` and `plot_multi_frame`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to plot in.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    im : CRISP
        The image object containing the data to plot.
    frame : str
        The units to use on the axes, "WCS", "arcsec" or "pix"
    title : str
        The title to apply to the panel.
    stokes_index : int, optional
        The specific Stokes component to plot if multiple are provided (indexes
        the first axis of im.data).
    set_xlabel : bool, optional
        Whether to apply the xlabel (default: True)
    set_ylabel : bool, optional
        Whether to apply the ulabel (default: True)
    cb_label : str, optional
        The label to attach to the colorbar (default: "I [DNs]")
    extent : list of int, optional
        Passed directly to imshow for frames that require setting extent
        ("pix"). Default: None.
    origin : str, optional
        Passed directly to imshow if required.
    norm : matplotlib.colors.Normalize
        The norm object to apply to the data.
    map_data :  (np.ndarray) -> np.ndarray, optional
        Optionally map the data in a specific way before plotting it.
    """
    frame_labels = {
        "WCSx": "Helioprojective Longitude [arcsec]",
        "WCSy": "Helioprojective Latitude [arcsec]",
        "pixx": "x [pixels]",
        "pixy": "y [pixels]",
        "arcsecx": "x [arcsec]",
        "arcsecy": "y [arcsec]",
    }

    if cb_label is None:
        cb_label = "I [DNs]"

    if norm is not None:
        vmin = None
    elif im.data.min() < 0:
        vmin = 0
    else:
        vmin = im.data.min()

    data = im.data[stokes_index] if stokes_index is not None else im.data
    data = data if map_data is None else map_data(data)
    plt_im = ax.imshow(
        data, cmap="Greys_r", vmin=vmin, norm=norm, extent=extent, origin=origin
    )
    if set_xlabel:
        ax.set_xlabel(frame_labels[f"{frame}x"])
    if set_ylabel:
        ax.set_ylabel(frame_labels[f"{frame}y"])
    ax.set_title(title)
    fig.colorbar(plt_im, ax=ax, orientation="vertical", label=cb_label)


@plt.rc_context(rc_context_dict)
def plot_single_frame(
    im: "CRISP",
    frame: str,
    title: str,
    cb_label: Optional[str] = None,
    extent: Optional[List[int]] = None,
    norm: Optional[matplotlib.colors.Normalize] = None,
    map_data: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fig=None,
):
    """
    Plot a single two-dimensional image from a CRISP object.  Calls
    `plot_single_frame_ax`.

    Parameters
    ----------
    im : CRISP
        The image object containing the data to plot.
    frame : str
        The units to use on the axes, "WCS", "arcsec" or "pix"
    title : str
        The title to apply to the panel.
    cb_label : str, optional
        The label to attach to the colorbar (default: "I [DNs]")
    extent : list of int, optional
        Passed directly to imshow for frames that require setting extent
        ("pix"). Default: None.
    norm : matplotlib.colors.Normalize
        The norm object to apply to the data.
    map_data :  (np.ndarray) -> np.ndarray, optional
        Optionally map the data in a specific way before plotting it.
    fig : matplotlib.figure.Figure, optional
        An existing figure to be plotted into. One will be created if not provided.
    """
    projection = im.wcs.low_level_wcs if frame == "WCS" and not im.rotate else None
    origin = "lower" if projection is None else None

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    plot_single_frame_ax(
        fig,
        ax,
        im,
        frame,
        title,
        cb_label=cb_label,
        extent=extent,
        origin=origin,
        norm=norm,
        map_data=map_data,
    )
    fig.show()


@plt.rc_context(rc_context_dict)
def plot_multi_frame(
    im: "CRISP",
    components: List[str],
    frame: str,
    norms: Optional[Dict[str, Optional[matplotlib.colors.Normalize]]] = None,
    map_datas: Optional[Dict[str, Optional[Callable[[np.ndarray], np.ndarray]]]] = None,
    extent: Optional[List[int]] = None,
    fig=None,
):
    """
    Plot 2D maps of multiple Stokes components from a CRISP image object. Calls
    `plot_single_frame_ax`.

    Parameters
    ----------
    im : CRISP
        The image object containing the data to plot.
    components : list of str
        List of the stokes components "I", "Q"... to plot.
    frame : str
        The units to use on the axes, "WCS", "arcsec" or "pix"
    norms : dict of str to matplotlib.colors.Normalize or None, optional
        A mapping from the Stokes component to the norm object to apply to the
        data, or None (to be ignored).
    map_datas :  dict of str to (np.ndarray) -> np.ndarray or None, optional
        A mapping from the Stokes component to an optional function to map the
        data in a specific way before plotting it.
    extent : list of int, optional
        Passed directly to imshow for frames that require setting extent
        ("pix"). Default: None.
    fig : matplotlib.figure.Figure, optional
        An existing figure to be plotted into. One will be created if not provided.
    """
    stokes_components = ["I", "Q", "U", "V"]
    num_components = len(components)

    projections = {s: None for s in stokes_components}
    if frame == "WCS":
        projections = {
            s: SlicedLowLevelWCS(im.wcs.low_level_wcs, i)
            for i, s in enumerate(stokes_components)
        }

    if norms is None:
        norms = {s: None for s in stokes_components}
    if map_datas is None:
        map_datas = {s: None for s in stokes_components}

    if fig is None:
        fig = plt.figure(figsize=(10, 8), constrained_layout=False)

    subplot_shape = (2, 2) if num_components == 4 else (1, num_components)
    for c, component in enumerate(components):
        ax = fig.add_subplot(
            *subplot_shape,
            c + 1,
            projection=projections[component],
        )
        set_xlabel = num_components != 4 or c >= 2
        set_ylabel = True
        if num_components == 4:
            if c == 1 or c == 3:
                set_ylabel = False
        else:
            if c != 0:
                set_ylabel = False

        origin = "lower" if projections[component] is None else None
        plot_single_frame_ax(
            fig,
            ax,
            im,
            frame,
            title=f"Stokes {component}",
            stokes_index=stokes_components.index(component),
            cb_label=f"{component} [DNs]",
            set_xlabel=set_xlabel,
            set_ylabel=set_ylabel,
            extent=extent,
            origin=origin,
            norm=norms[component],
            map_data=map_datas[component],
        )
        if num_components == 4:
            if c < 2:
                ax.xaxis.tick_top()
                ax.tick_params(axis="x", labelbottom=False)
            if c == 1 or c == 3:
                ax.yaxis.tick_right()
                ax.tick_params(axis="y", labelleft=False)

    fig.show()
