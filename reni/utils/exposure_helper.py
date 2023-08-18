# MIT License

# Copyright (c) 2021 Mark Boss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#### ALL CODE FROM https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition #####

import numpy as np

def sRGBToLinear(x: np.ndarray) -> np.ndarray:
    """
    Convert sRGB color space to linear color space.

    Args:
        x: An array of sRGB values.

    Returns:
        An array of linear values.
    """
    return np.where(x >= 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def linearTosRGB(x: np.ndarray) -> np.ndarray:
    """
    Convert linear color space to sRGB color space.

    Args:
        x: An array of linear values.

    Returns:
        An array of sRGB values.
    """
    return np.where(x >= 0.0031308, 1.055 * np.power(x, 1.0 / 2.4) - 0.055, x * 12.92)


def calculate_ev100_from_metadata(aperture_f: float, shutter_s: float, iso: int):
    """
    Calculate EV100 from camera metadata.

    Args:
        aperture_f: The aperture value.
        shutter_s: The shutter speed value.
        iso: The ISO value.

    Returns:
        The EV100 value.
    """
    ev_s = np.log2((aperture_f * aperture_f) / shutter_s)
    ev_100 = ev_s - np.log2(iso / 100)
    return ev_100


def calculate_luminance_from_ev100(ev100, q=0.65, S=100):
    """
    Calculate luminance from EV100.

    Args:
        ev100: The EV100 value.
        q: The constant q value.
        S: The constant S value.

    Returns:
        The luminance value.
    """
    return (78 / (q * S)) * np.power(2.0, ev100)


def convert_luminance(x: np.ndarray) -> np.ndarray:
    """
    Convert RGB color space to luminance.

    Args:
        x: An array of RGB values.

    Returns:
        An array of luminance values.
    """
    return 0.212671 * x[..., 0] + 0.71516 * x[..., 1] + 0.072169 * x[..., 2]


def smoothStep(x, edge0=0.0, edge1=1.0):
    """
    Smoothly interpolate between two values.

    Args:
        x: The input value.
        edge0: The lower edge value.
        edge1: The upper edge value.

    Returns:
        The interpolated value.
    """
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def compute_avg_luminance(x: np.ndarray) -> np.ndarray:
    """
    Compute the average luminance of an image.

    Args:
        x: An array of RGB values.

    Returns:
        An array of average luminance values.
    """
    L = np.nan_to_num(convert_luminance(x))
    L = L * center_weight(L)

    if len(x.shape) == 3:
        axis = (0, 1)
    elif len(x.shape) == 4:
        axis = (1, 2)
    else:
        raise ValueError(
            "Only 3 dimensional (HWC) or 4 dimensionals (NHWC) images are supported"
        )
    avgL1 = np.average(L, axis=axis)
    return avgL1


def compute_ev100_from_avg_luminance(avgL, S=100.0, K=12.5):
    """
    Compute EV100 from average luminance.

    Args:
        avgL: The average luminance value.
        S: The constant S value.
        K: The constant K value.

    Returns:
        The EV100 value.
    """
    return np.log2(avgL * S / K)  # or 12.7


def convert_ev100_to_exp(ev100, q=0.65, S=100):
    """
    Convert EV100 to exposure value.

    Args:
        ev100: The EV100 value.
        q: The constant q value.
        S: The constant S value.

    Returns:
        The exposure value.
    """
    maxL = (78 / (q * S)) * np.power(2.0, ev100)
    return np.clip(1.0 / maxL, 1e-7, None)


def compute_auto_exp(
    x: np.ndarray, clip: bool = True, returnEv100: bool = True
) -> np.ndarray:
    """
    Compute auto exposure for an image.

    Args:
        x: An array of RGB values.
        clip: Whether to clip the output values.
        returnEv100: Whether to return the EV100 value.

    Returns:
        An array of auto-exposed RGB values and optionally the EV100 value.
    """
    avgL = np.clip(compute_avg_luminance(x), 1e-5, None)
    ev100 = compute_ev100_from_avg_luminance(avgL)

    ret = apply_ev100(x, ev100, clip)
    if returnEv100:
        return ret, ev100
    else:
        return ret


def apply_ev100(x: np.ndarray, ev100, clip: bool = True):
    """
    Apply exposure correction to an image.

    Args:
        x: An array of RGB values.
        ev100: The EV100 value.
        clip: Whether to clip the output values.

    Returns:
        An array of corrected RGB values.
    """
    exp = convert_ev100_to_exp(ev100)  # This can become an invalid number. why?

    if len(x.shape) == 3:
        exposed = x * exp
    else:
        exposed = x * exp.reshape((exp.shape[0], *[1 for _ in x.shape[1:]]))
    if clip:
        exposed = np.clip(exposed, 0.0, 1.0)

    return exposed


def center_weight(x):
    """
    Compute the center weight of an image.

    Args:
        x: An array of RGB values.

    Returns:
        An array of center weight values.
    """
    def smoothStep(x, edge0=0.0, edge1=1.0):
        """
        Smoothly interpolate between two values.
        """
        x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return x * x * x * (x * (x * 6 - 15) + 10)

    idx = np.argwhere(np.ones_like(x))
    idxs = np.reshape(idx, (*x.shape, len(x.shape)))

    if len(x.shape) == 2:
        axis = (0, 1)
    elif len(x.shape) == 3:
        axis = (1, 2)
        idxs = idxs[..., 1:]
    else:
        raise ValueError(
            "Only 2 dimensional (HW) or 3 dimensionals (NHW) images are supported"
        )

    center_idx = np.array([x.shape[axis[0]] / 2, x.shape[axis[1]] / 2])
    center_dist = np.linalg.norm(idxs - center_idx, axis=-1)

    return 1 - smoothStep(center_dist / x.shape[axis[1]] * 2)