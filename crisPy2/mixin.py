from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from .utils import ObjDict

class CRISPSlicingMixin(NDSlicingMixin):
    '''
    This is the parent class that will allow the CRISP objects to be sliced without having to create new objects.
    '''

    def __getitem__(self, item):
        kwargs = self._slice(item)
        return self.__class__(**kwargs)

    def _slice(self, item):
        kwargs = {}
        kwargs["file"] = ObjDict({})
        kwargs["file"]["data"] = self.file.data[item]
        kwargs["uncertainty"] = self._slice_uncertainty(item)
        kwargs["mask"] = self._slice_mask(item)
        kwargs["wcs"] = self._slice_wcs(item)
        kwargs["file"]["header"] = self.file.header

        return kwargs

class CRISPSequenceSlicingMixin(CRISPSlicingMixin):
    '''
    This is the parent class that will allow the CRISPSequence objects to be sliced without having to create new objects.
    '''

    def __getitem__(self, item):
        args = self._slice(item)
        return self.__class__(args)

    def _slice(self, item):
        args = [f._slice(item) for f in self.list]
        return args