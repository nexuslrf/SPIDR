from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from torchvision.utils import make_grid
from os.path import join
import torch.nn.functional as F
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_property2dict(target_dict, object, props):
    for prop in props:
        target_dict[prop] = getattr(object, prop)


def normalize(v, axis=0):
    # axis = 0, normalize each col
    # axis = 1, normalize each row
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-9)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


def unique_lst(list1):
    x = np.array(list1)
    return np.unique(x)


def read_map(path):
    arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise RuntimeError(f"Failed to read\n\t{path}")
    # RGB
    if arr.ndim == 3 or arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return rgb

    raise NotImplementedError(arr.shape)

def resize(arr, new_h=None, new_w=None, method='cv2'):
    """Resizes an image, with the option of maintaining the aspect ratio.

    Args:
        arr (numpy.ndarray): Image to binarize. If multiple-channel, each
            channel is resized independently.
        new_h (int, optional): Target height. If ``None``, will be calculated
            according to the target width, assuming the same aspect ratio.
        new_w (int, optional): Target width. If ``None``, will be calculated
            according to the target height, assuming the same aspect ratio.
        method (str, optional): Accepted values: ``'cv2'`` and ``'tf'``.

    Returns:
        numpy.ndarray: Resized image.
    """
    h, w = arr.shape[:2]
    if new_h is not None and new_w is not None:
        if int(h / w * new_w) != new_h:
            print((
                "Aspect ratio changed in resizing: original size is %s; "
                "new size is %s"), (h, w), (new_h, new_w))
    elif new_h is None and new_w is not None:
        new_h = int(h / w * new_w)
    elif new_h is not None and new_w is None:
        new_w = int(w / h * new_h)
    else:
        raise ValueError("At least one of new height or width must be given")

    if method in ('cv', 'cv2', 'opencv'):
        interp = cv2.INTER_LINEAR if new_h > h else cv2.INTER_AREA
        resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)
    else:
        raise NotImplementedError(method)

    return resized

def write_uint(arr_uint, outpath):
    r"""Writes an ``uint`` array as an image to disk.

    Args:
        arr_uint (numpy.ndarray): A ``uint`` array.
        outpath (str): Output path.

    Writes
        - The resultant image.
    """
    if arr_uint.ndim == 3 and arr_uint.shape[2] == 1:
        arr_uint = np.dstack([arr_uint] * 3)

    img = Image.fromarray(arr_uint)
    img.save(outpath)

def write_arr(arr_0to1, outpath, img_dtype='uint8', clip=False):
    r"""Writes a ``float`` array as an image to disk.

    Args:
        arr_0to1 (numpy.ndarray): Array with values roughly :math:`\in [0,1]`.
        outpath (str): Output path.
        img_dtype (str, optional): Image data type. Defaults to ``'uint8'``.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Writes
        - The resultant image.

    Returns:
        numpy.ndarray: The resultant image array.
    """
    arr_min, arr_max = arr_0to1.min(), arr_0to1.max()
    if clip:
        if arr_max > 1:
            print("Maximum before clipping: %f", arr_max)
        if arr_min < 0:
            print("Minimum before clipping: %f", arr_min)
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    else:
        assert arr_min >= 0 and arr_max <= 1, \
            "Input should be in [0, 1], or allow it to be clipped"
    # Float array to image
    img_arr = (arr_0to1 * np.iinfo(img_dtype).max).astype(img_dtype)

    write_uint(img_arr, outpath)

    return img_arr