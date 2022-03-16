import codecs, os, sys, re, setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)

    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="sst-crispy",
    version=find_version("crispy/__init__.py"),
    author="John Armstrong",
    author_email="j.armstrong@strath.ac.uk",
    description="A Python package for using data from the Swedish 1 m Solar Telescope's CRisp Imaging SpectroPolarimeter instrument.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bionictoucan/crispy",
    packages=setuptools.find_packages(),
    install_requires = [
        "numpy <= 1.22.2",
        "astropy <= 5.0.1",
        "matplotlib <= 3.5.1",
        "zarr <= 2.11.0",
        "tqdm <= 4.62.3",
        "cycler <= 0.10.0",
        "specutils <= 1.6.0",
        "numba <= 0.51.0",
        "weno4 <= 1.1.1",
        "sunpy <= 3.1.3",
        "ipywidgets <= 7.6.5",
        "jupyterlab <= 3.2.9"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">3.6"
)
