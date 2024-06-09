import numpy as np
import numpy.typing as npt
from contours import Contour, Circle
import skimage.draw as draw
import skimage.transform as transform
from iminuit import Minuit
from line_alg import wu_line
from numba import njit

# vline_aa = np.vectorize(draw.line_aa)

# @njit
def darkness_along_line_aa(target, x1, y1, x2, y2, thickness):
    """
    Calculate the average darkness along a line in the target image.

    This function uses Bresenham's algorithm to get the coordinates of a line
    between two points and then calculates the average pixel value along that
    line, weighted by anti-aliasing factors.

    Parameters:
    -----------
    target : np.ndarray
        The target image in which darkness is measured.
    x1, y1 : int
        The starting coordinates of the line.
    x2, y2 : int
        The ending coordinates of the line.

    Returns:
    --------
    mean_darkness : float
        The average darkness along the line.
    """
    # Bresenham's algorithm to get the coordinates of the line
    line_x, line_y, val_aa = wu_line(x1, y1, x2, y2, thickness)

    # Extract pixel values along the line from both images
    values_target = target[line_x, line_y] * val_aa
    # weights = weights[line_x, line_y] * val_aa

    # Calculate the root mean square error
    mean_darkness = np.mean(values_target)
    return mean_darkness


class Solver:
    """
    A class to solve the sting placement optimization problem using various
    modes.

    Attributes:
    -----------
    shape : Contour
        The contour shape of the 2D perimeter.
    img : np.ndarray
        The input image, will be extended to a square and converted to
        grayscale.
    mode : str
        The mode of the solver. Default is "greedy_dark".
    dpmm : float
        Dots per millimeter for internal computation. Default is 12.

    Methods:
    --------
    __init__(self, shape, img, mode="greedy_dark", dpmm=12):
        Initializes the Solver with the given shape, image, mode, and dpmm.

    _build_target_function(self):
        Builds and returns the target function for minimization based on the
        solver's mode.

    s0(self):
        Returns the starting values for the solver based on the mode.
    """

    img_type = npt.NDArray[np.uint8]

    def __init__(
        self, contour: "Contour", img: img_type, mode="greedy_dark", dpmm=12.0, line_thickness = 0.4
    ):
        """
        Initializes the Solver with the given shape, image, mode, and dpmm.

        Parameters:
        -----------
        shape : Contour
            The contour shape of the 2D perimeter. Only concave shapes are
            supported!
        img : np.ndarray
            The input image.
        mode : str, optional
            The mode of the solver. Default is "greedy_dark".
        dpmm : float, optional
            Dots per millimeter for internal computation. Default is 12.
        """
        self.contour = contour
        self.dpmm = dpmm  # dots per millimeter for the internal computation
        self.mode = mode
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = np.asarray(img, dtype=np.float32)
        contour_dx, contour_dy = self.contour.get_extension()  # in mm
        img_height, img_width = img.shape
        # the scaling needs some testing
        scale_x = contour_dx * self.dpmm / img_width
        scale_y = contour_dy * self.dpmm / img_height
        self.img_scale_factor = max(scale_x, scale_y)
        # adding plus one to avoid out of bound in line tracer.
        # In short, image must be one larger than max index
        img = transform.resize(
            img,
            (
                round(img_height * self.img_scale_factor + 1),
                round(img_width * self.img_scale_factor + 1),
            ),
            anti_aliasing=True,
        )
        self._img = np.asarray(
            img, dtype=np.float32
        )  # internal working copy that is rescaled and will be modified
        self._line_thickness = line_thickness * dpmm
        self._border_padding = int(np.ceil(self._line_thickness/2))
        self._img = cv2.copyMakeBorder(self._img, self._border_padding, self._border_padding, self._border_padding, self._border_padding, cv2.BORDER_REFLECT)

    def _build_target_function(self):
        """
        Builds and returns the target function for minimization based on the
        solver's mode.

        Returns:
        --------
        function
            The target function for minimization.
            Inputs are two floats with the start and the end of the string with
            values in the range [0,1] that define on wich point of the countur
            the string will be places

        Raises:
        -------
        NotImplementedError
            If the solver mode is not implemented.
        """
        match self.mode:
            case "greedy_dark":

                def minimization_target(
                    s1: float | npt.ArrayLike, s2: float | npt.ArrayLike
                ):
                    # getting the shifted corrdinates, so that the min is [0, 0]
                    x1, y1 = self.contour.get_coordinates(
                        s1, do_pos_shift=True
                    )
                    x2, y2 = self.contour.get_coordinates(
                        s2, do_pos_shift=True
                    )
                    # x1 = np.rint(x1 * self.dpmm).astype(int)
                    # y1 = np.rint(y1 * self.dpmm).astype(int)
                    # x2 = np.rint(x2 * self.dpmm).astype(int)
                    # y2 = np.rint(y2 * self.dpmm).astype(int)
                    x1 = x1 * self.dpmm + self._border_padding
                    y1 = y1 * self.dpmm + self._border_padding
                    x2 = x2 * self.dpmm + self._border_padding
                    y2 = y2 * self.dpmm + self._border_padding
                    # penalty term, if the distance is too small
                    # squared increase from 0 to 1 for values of abs(s2-s1)<0.1
                    penalty = (
                        np.clip((1 - np.abs(s1 - s2)) * 10 - 9, 0, None) ** 2
                    )
                    error = (
                        darkness_along_line_aa(self._img, x1, y1, x2, y2, self._line_thickness)
                        + penalty
                    )
                    return error

                return minimization_target
            case _:
                raise NotImplementedError(
                    f"solver for mode `{self.mode}` not implemented!"
                )

    @property
    def s0(self):
        match self.mode:
            case "greedy_dark":
                return (0, 0.5)
            case _:
                raise NotImplementedError(
                    f"starting values for mode `{self.mode}` not implemented!"
                )

    def next_move(self):
        m = Minuit(self._build_target_function(), s1=self.s0[0], s2=self.s0[1])
        m.limits["s1", "s2"] = (0, 1)
        m.scan(ncall=50)
        m.migrad()
        print(m)


if __name__ == "__main__":
    import cv2

    circle = Circle((0, 0), 50)
    img = cv2.imread(
        r"C:\Users\CapDaniels\Meine Ablage\Documents\CodingProjects\pythonProjects\string_art\test_images\8.jpg"
    )

    solver = Solver(circle, img)
    solver.next_move()
