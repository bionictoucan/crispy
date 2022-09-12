from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from .utils import ObjDict
from typing import Union, Sequence


class CRISPSlicingMixin(NDSlicingMixin):
    """
    This is the parent class that will allow the CRISP objects to be sliced
    without having to create new objects.
    """

    def __getitem__(self, item: Union[int, Sequence]):
        kwargs = self._slice(item)
        cl = self.__class__(**kwargs)
        cl.ind = self._normalise_ind(item)
        return cl

    def _slice_wvl(self, item, wave_idx):
        if item is None:
            return None

        if isinstance(item, (int, slice)) and wave_idx == 0:
            return self.wvl[item]
        else:
            try:
                idx = item[wave_idx]
            except IndexError:
                return self.wvl

            return self.wvl[idx]

    def _normalise_ind(self, item):
        """Make ind the same length as the data dimensionality by inserting empty slices.
        """
        ind = [slice(None, None) for _ in range(self.data.ndim)]
        if isinstance(item, Sequence):
            ind[:len(item)] = item
        else:
            ind[0] = item
        return ind

    def _slice(self, item: Union[int, Sequence]):
        kwargs = {}
        kwargs["filename"] = ObjDict({})
        kwargs["filename"]["data"] = self.data[item]
        kwargs["uncertainty"] = self._slice_uncertainty(item)
        kwargs["mask"] = self._slice_mask(item)
        kwargs["wcs"] = self._slice_wcs(item)
        kwargs["filename"]["header"] = self.header
        kwargs["nonu"] = self.nonu
        try:
            wave_idx = self.wcs.naxis - self.wcs.axis_type_names.index('WAVE') - 1
            kwargs["wvl"] = self._slice_wvl(item, wave_idx)
            kwargs["orig_wvl"] = self.orig_wvl
        except ValueError:
            pass

        return kwargs


class CRISPSequenceSlicingMixin(CRISPSlicingMixin):
    """
    This is the parent class that will allow the CRISPSequence objects to be
    sliced without having to create new objects.
    """

    def __getitem__(self, item: Union[int, Sequence]):
        args = self._slice(item)
        return self.__class__(args)

    def _slice(self, item: Union[int, Sequence]):
        args = [f._slice(item) for f in self.list]
        return args


class InversionSlicingMixin(NDSlicingMixin):
    """
    This is the parent class that will allow the Inversion objects to be sliced
    without having to create new objects.
    """

    def __getitem__(self, item: Union[int, Sequence]):
        kwargs = self._slice(item)
        cl = self.__class__(**kwargs)
        cl.ind = item
        return cl

    def _slice(self, item: Union[int, Sequence]):
        kwargs = {}
        # if type(item) == tuple:
        #     err_item = list(item)
        #     err_item.insert(1, slice(None))
        #     err_item = tuple(err_item)
        # else:
        #     err_item = item
        kwargs["filename"] = ObjDict({})
        kwargs["filename"]["ne"] = self.ne[item]
        kwargs["filename"]["temperature"] = self.temp[item]
        kwargs["filename"]["vel"] = self.vel[item]
        kwargs["filename"]["ne_err"] = self.ne_err[item]
        kwargs["filename"]["temperature_err"] = self.temp_err[item]
        kwargs["filename"]["vel_err"] = self.vel_err[item]
        kwargs["wcs"] = self._slice_wcs(item)
        if isinstance(item, int):
            kwargs["z"] = self.z[item]
        else:
            kwargs["z"] = self.z[item[0]]
        kwargs["header"] = self.header

        return kwargs
