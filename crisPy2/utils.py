from cycler import cycler

class ObjDict(dict):
    '''
    This is an abstract class for allowing the keys of a dictionary to be accessed like class attributes.
    '''

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: "+name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: "+name)

## The following is the Paul Tol colourblind colour schemes for use in plots from https://personal.sron.nl/~pault/ ##

pt_bright = {
    "blue" : "#4477AA",
    "cyan" : "#66CCEE",
    "green" : "#228833",
    "yellow" : "#CCBB44",
    "red" : "#EE6677",
    "purple" : "#AA3377",
    "grey" : "#BBBBBB"
}

pt_hic = {
    "white" : "#FFFFFF",
    "yellow" : "#DDAA33",
    "red" : "#BB5566",
    "blue" : "#004488",
    "black" : "#000000"
}

pt_vibrant = {
    "blue" : "#0077BB",
    "cyan" : "#33BBEE",
    "teal" : "#009988",
    "orange" : "#EE7733",
    "red" : "#CC3311",
    "magenta" : "#EE3377",
    "grey" : "#BBBBBB"
}

pt_muted = {
    "indigo" : "#332288",
    "cyan" : "#88CCEE",
    "teal" : "#44AA99",
    "green" : "#117733",
    "olive" : "#999933",
    "sand" : "#DDCC77",
    "rose" : "#CC6677",
    "wine" : "#882255",
    "purple" : "#AA4499",
    "pale grey" : "#DDDDDD"
}

pt_bright_cycler = cycler(color=list(pt_bright.values()))

pt_hic_cycler = cycler(color=list(pt_hic.values()))

pt_vibrant_cycler = cycler(color=list(pt_vibrant.values()))

pt_muted_cycler = cycler(color=list(pt_muted.values()))

def CRISP_sequence_constructor(files, wcs=None, uncertainty=None, mask=None, nonu=False):
    file_dict_list = []

    for j, f in enumerate(files):
        d = {}
        d["filename"] = f
        if wcs is not None:
            d["wcs"] = wcs[j]
        if uncertainty is not None:
            d["uncertainty"] = uncertainty[j]
        if mask is not None:
            d["mask"] = mask[j]
        d["nonu"] = nonu

        file_dict_list.append(d)

    return file_dict_list

def find_corners(img, rows=False, reverse=False):
    '''
    This is a function to find the corners of a CRISP observation since the images are rotated in the image plane.
    Parameters
    ----------
    img : numpy.ndarray
        The image to be rotated. Since CRISP observations usually have multiple images, all of the images will be rotated the same amount in the image plane for a single observation so the corners only need to be found for one -- this is usually taken as the core wavelength (for a waveband measurement).
    rows : bool, optional
        Whether or not to search along the rows. Default searches down the columns.
    reverse : bool, optional
        Determines which direction the search takes place. This also depends on the rows parameter. Default searches from top to bottom if rows is False and left to right if rows is True.
    Returns
    -------
    corner : numpy.ndarray
        An array containing the image plane coordinates of the corner that is to be found. The coordinate is returned in (y,x) format.
    Since CRISP images are rectangular, only 3 of 4 corners are required to be able to obtain the whole image.
    If rows and reverse are both False, then the algorithm finds the top-left corner. If rows is True but reverse is False, the algorithm finds the top-right corner. If rows is False and reverse is True, the algorithm find the bottom-right corner. If rows and reverse are both True, then the algorithm finds the bottom-left corner.
    '''

    if reverse and not rows:
        y_range = range(img.shape[0])
        x_range = reversed(range(img.shape[1]))
    elif reverse and rows:
        y_range = reversed(range(img.shape[0]))
        x_range = range(img.shape[1])
    else:
        y_range = range(img.shape[0])
        x_range = range(img.shape[1])

    if not rows:
        for i in x_range:
            for j in y_range:
                if img[j, i] != img[0, 0]:
                    if img[j, i] == img[0, 0] + 1 or img[j, i] == img[0, 0] - 1:
                        pass
                    else:
                        corner = np.array([j, i])
                        return corner
    else:
        for j in y_range:
            for i in x_range:
                if img[j, i] != img[0, 0]:
                    if img[j, i] == img[0, 0] + 1 or img[j, i] == img[0, 0] - 1:
                        pass
                    else:
                        corner = np.array([j, i])
                        return corner

def im_rotate(img_cube):
    '''
    This is a function that will find the corners of the image and rotate it with respect to the x-axis and crop so only the map is left in the array.
    Parameters
    ----------
    img_cube : numpy.ndarray
        The image cube to be rotated.
    Returns
    -------
    img_cube : numpy.ndarray
        The rotated image cube.
    '''

    mid_wvl = img_cube.shape[0] // 2
    #we need to find three corners to be able to rotate and crop properly and two of these need to be the bottom corners
    bl_corner = find_corners(img_cube[mid_wvl], rows=True, reverse=True)
    br_corner = find_corners(img_cube[mid_wvl], reverse=True)
    tr_corner = find_corners(img_cube[mid_wvl], rows=True)
    unit_vec = (br_corner - bl_corner) / np.linalg.norm(br_corner - bl_corner)
    angle = np.arccos(np.vdot(unit_vec, np.array([0, 1]))) #finds the angle between the image edge and the x-axis
    angle_d = np.rad2deg(angle)

    #find_corners function finds corners in the frame where the origin is the natural origin of the image i.e. (y, x) = (0, 0). However, the rotation is done with respect to the centre of the image so we must change the corner coordinates to the frame where the origin of the image is the centre of the image. Furthermore, as we will be performing an affine transformation on the corner coordinates to obtain our crop ranges we need to add a faux z-axis for the rotation to occur around e.g. add a third dimension. Also as the rotation requires interpolation, the corners are not easily identifiable after the rotation by the find_corners method so find their transformation rotation directly is the best way to do it
    bl_corner = np.array([bl_corner[1]-(img_cube.shape[-1]//2), bl_corner[0] - (img_cube.shape[-2]//2), 1])
    br_corner = np.array([br_corner[1]-(img_cube.shape[-1]//2), br_corner[0] - (img_cube.shape[-2]//2), 1])
    tr_corner = np.array([tr_corner[1]-(img_cube.shape[-1]//2), tr_corner[0] - (img_cube.shape[-2]//2), 1])
    rot_matrix = np.array([[np.cos(-angle), np.sin(-angle), img_cube.shape[-1]//2], [-np.sin(-angle), np.cos(-angle), img_cube.shape[-2]//2], [0,0,1]])
    new_bl_corner = np.rint(np.matmul(rot_matrix, bl_corner))
    new_br_corner = np.rint(np.matmul(rot_matrix, br_corner))
    new_tr_corner = np.rint(np.matmul(rot_matrix, tr_corner))

    for j in range(img_cube.shape[0]):
        img_cube[j] = rotate(img_cube[j], -angle_d, reshape=False, output=np.int16, cval=img_cube[j,0,0])
    img_cube = img_cube[:, int(new_tr_corner[1]):int(new_bl_corner[1]), int(new_bl_corner[0]):int(new_br_corner[0])]

    return img_cube