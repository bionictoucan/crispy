# crispy: A Python Package for Using Imaging Spectropolatimetric Data in Solar Physics

Author: John A. Armstrong, *Univeristy of Glasgow*

Maintained: Chris Osborne, *University of Glasgow*

The following repository contains the source code for the `crispy` Python
package, aptly named due to my working with the CRisp Imagining
SpectroPolarimeter (CRISP) instrument mounted at the Swedish 1-m Solar Telescope
(SST) in La Palma, Spain. (However, the code will work with *any* imaging
spectropolarimetric data regardless of instrument so long as it follows normal
FITS standards or a custom zarr standard I have grown accustomed to using, see
the docs for more details).

## Why does this exist?

``crispy`` started its life as the base code for data viewing/augmentation/analysis for my PhD as the tools for data exploration and exploitation of this kind of solar physics data did not previously exist.

## What is imaging spectropolarimetric data?

The type of data that ``crispy`` is built to deal with is **optical** imaging spectropolarimetry data. This kind of data consists of measurements of a number of Stokes profiles at specific narrowband wavelength points over an extended field-of-view over a given length of time. What this means is that we have five-dimensional data structures ordered (t, stokes, &lambda;, y, x). This kind of data is very powerful for exploring the lower solar atmosphere due to having relatively high spectral, polarimetric and spatial resolutions. The time resolution can be not shabby too.

## How do I get this moderately cool Python package?

``crispy`` can be found on PyPI using:

```
pip install sst-crispy
```

or can be installed from source using:

```
git clone https://github.com/bionic-toucan/crispy
cd crispy
python setup.py install --user
```

## I don't want to download the examples/docs just the code pls

If you are cloning this repository and only after the raw code there is a way to
only retrieve this based on `this
<https://askubuntu.com/questions/460885/how-to-clone-only-some-directories-from-a-git-repository>`_.

```
mkdir crispy && cd crispy
git init
git remote add -f origin https://github.com/bionictoucan/crispy

git config core.sparseCheckout true

echo "crispy/" >> .git/info/sparse-checkout

git pull origin main
```

## Acknowledgements
I would like to thank Chris Osborne (https://github.com/Goobley) for the improved rotation algorithm used for data rotated in the images plane to coincide with the helioprojective plane and also for listening to me complaining all the time and telling me about the wonders of properties.
