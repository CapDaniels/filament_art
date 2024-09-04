import matplotlib
matplotlib.use("tkagg")

import numpy as np

# import numpy.typing as npt
from contours import Contour, Circle
import skimage.transform as transform

from line_alg import wu_line
from numba import njit
import matplotlib.pyplot as plt
import itertools
import cv2

# import cProfile
# import pstats
from gcode import GSketch
from gcode_builder import GcodeStringer
import warnings
from pathlib import Path
from tkinter import filedialog
from typing import Tuple
import numpy.typing as npt
import tkinter as tk
# import time


img_type = cv2.typing.MatLike


def normalize(v: Tuple[float, float] | Tuple[float, float, float] | list[float] | npt.ArrayLike):
    """Normalizing a vector.

    Args:
        v (Tuple[float, float] | Tuple[float, float, float] | list[float] |
        npt.ArrayLike): the vector in question. If v.ndim == 2, an array of
        vectors which are normalized along the last dimension is returned.

    Returns:
        np.ndarray: the normalized vector
    """ 
    v = np.array(v)
    if v.ndim == 1:
        return v / np.sqrt(np.sum(np.square(v), axis=-1))
    return v / np.sqrt(np.sum(np.square(v), axis=-1))[:, np.newaxis]


@njit
def line_overlap_penalty_f(x, overlap_penalty=0.2):
    """Calculates a penalty for one pixel, depending on the number of drawn
    already lines.

    Args:
        x (density): how many aa-lines are at one pixel

    Returns:
        float: the penalty value
    """

    if x == 1.0:
        return 0.0
    return -(x - 1) * overlap_penalty


@njit()
def error_along_line_aa_njit(
    error_img,
    canvas,
    x1,
    y1,
    x2,
    y2,
    thickness,
    overlap_penalty=0.2,
    img_weights=None,
):
    """This function uses a modified version of Wu's line algorithm to get the
    coordinates of the pixels along a line between two points. Then the weighted
    average error of the target image is calculated. This can understod as
    searching for the next string, thats adds the most darkness, where it is
    needed. The weight of each pixel is a product of the given image weights and
    the anti-aliasing value of the Wu-line. Additional peanalties are applied if
    a line crosses many 'inactive' pixels or if a line crosses to many already
    drawn lines.

    Args:
        error_img (np.ndarray):
            The error image in which contains the current error.
        canvas (np.ndarray):
            The current canvas. Ranges from [1, -inf) and contains the already
            drawn lines.
        x1, y1 : int
            The starting coordinates of the line.
        x2, y2 : int
            The ending coordinates of the line.
        thickness (float): The thickness of the line.
        overlap_penalty (float, optional): Additional penalty factor for
            crossing existing strings. Defaults to 0.2.
        img_weights (np.ndarray, optional): Weights of the image.
            Defaults to None.

    Returns:
        float: the error of the line, smaller is better.
    """
    # modifies Wu's algorithm to get the coordinates of the line
    # weights here mean the alpha values of the drawn line while img_weights are
    # the weights of the image mask
    line_x, line_y, aa_vals = wu_line(x1, y1, x2, y2, thickness)

    # if len(line_x) == 0:
    #     return np.inf

    # can be thought of as the remaining darkness along the line
    # this is a minimization problem, so lower is better
    error = 0.0
    weight_penalty = 0.0
    line_denstiy_penalty = 0.0
    weights = aa_vals.copy()
    if img_weights is not None:
        img_weights_sum = 0.0
        for idx, (x, y) in enumerate(zip(line_x, line_y)):
            w = img_weights[x, y]
            img_weights_sum += w
            weights[idx] *= w
        if img_weights_sum == 0.0:
            # otherwise we would divide by zero
            return np.inf

        # punish lines that choose only a few "active" pixels
        # unused for now
        weight_penalty = 25 * 1 / img_weights_sum

        # weigths *= img_weights
    for x, y, w in zip(line_x, line_y, weights):
        error += error_img[x, y] * w
        line_denstiy_penalty += line_overlap_penalty_f(
            canvas[x, y], overlap_penalty
        )

    weights_sum = sum(weights)
    if weights_sum == 0.0:
        return np.inf

    return (
        error / (sum(weights))
        + line_denstiy_penalty / len(line_x)
        + weight_penalty
    )

class Solver:
    """
    A class to solve the sting placement optimization problem using various
    modes. (Of which I've implemented one so far...)
    """

    _save_path = None

    def __init__(
        self,
        name: str,
        contour: "Contour",
        img: img_type,
        img_weights: img_type | None = None,
        weights_importance: float = 1.0,
        mode="greedy_dark",
        dpmm: float = 10.0,
        line_thickness: float = 0.4,
        opacityd: float = 1.0,
        opacitys: float = 1.0,
        n_points: int = 600,
        overlap_penalty: float = 0.25,  # 0.1
    ):
        """
        Initializes the Solver with the given shape, image, mode, and dpmm.

        Parameters:
        -----------
        name : str
            The name of the Solver.
        contour : Contour
            The contour shape of the 2D perimeter. Only concave shapes are
            supported!
        img : np.ndarray
            The input image.
        img_weights : np.ndarray, optional
            Weights for the image mask. If provided, will be used to optimize
            the sting placement.
        weights_importance : float, optional
            Importance of weights in the optimization process. Default is 1.0.
        mode : str, optional
            The mode of the solver. Supported modes are only "greedy_dark" for
            now. Default is "greedy_dark".
        dpmm : float, optional
            Dots per millimeter for internal computation. Default is 10.0.
        line_thickness : float, optional
            Thickness of the sting lines. Default is 0.4.
        opacityd : float, optional
            Opacity of the dark parts of the sting. Default is 1.0.
        opacitys : float, optional
            Opacity of the light parts of the sting. Default is 1.0.
        n_points : int, optional
            Number of sting placement points along the contour. Default is 600.
        overlap_penalty : float, optional
            Penalty factor for overlapping stings. Higher values penalize more
            overlaps, resulting in lower local string densitys. Default is 0.25.
        """
        self.name = name
        self.contour = contour
        self.dpmm = dpmm  # dots per millimeter for the internal computation
        self.mode = mode
        self.overlap_penalty = overlap_penalty
        # drawing opaticity for the resulting image
        # affects the penalty of crossing lines
        self.opacityd = opacityd
        # solver opaticity, used when updating the solver target image
        self.opacitys = opacitys
        self.n_points = n_points
        self.string_count = 0
        self.curr_score = 0
        self.n_points = n_points
        # line thickness in dots for internal use
        self._line_thickness = line_thickness * dpmm
        self.border_padding = int(np.ceil(self._line_thickness / 2)) + 1

        # TODO: push this as a messagebox to main GUI
        if self._line_thickness < 1.0:
            warnings.warn(
                f"Your DPI is low for the selected thickness. A value above {1/(line_thickness+0.0001):.2f} is adviced."
            )

        self._init_adjacency_matrix()
        self._init_image(img)
        self._init_image_weights(img_weights, weights_importance)

        self._output_canvas = np.ones(
            self._error_img.shape
        )  # to draw the resulting image

    def _prepare_img(self, img):
        """ Removing color component and rescaling to [0, 1].

        Args:
            img (np.ndarray): the image.

        Returns:
            np.ndarray: the processed image.
        """        
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.asarray(img, dtype=np.float32) / 255

    def _add_border_padding(self, img: img_type, padd: int | None = None):
        """
        Adds padding to an image using 'reflection'.

        Args:
            img (np.ndarray):
                The input image to add padding to.
            padd (int | None):
                Number of pixels to pad on each side. If None, uses the
                calculated border_padding. Defaults to None.

        Returns:
            np.ndarray: The padded image.
        """
        if padd is None:
            padd = self.border_padding
        return cv2.copyMakeBorder(
            img,
            padd,
            padd,
            padd,
            padd,
            cv2.BORDER_REFLECT,
        ) 

    def _init_adjacency_matrix(self):
        """
        Initializes an adjacency matrix for the given number of points and
        calculates the coordinates corresponding to each point along the
        contour.

        Returns:
            None
        """
        self.s_vals = np.linspace(0.0, 1.0, self.n_points + 1, endpoint=False)
        self.xy_points = np.array(
            [self._get_coordinates(s) for s in self.s_vals]
        )
        self.curr_sidx = None
        self._solved_first = False
        self.s_connections = []
        # adjacency matrix to avoid doubling the strings
        self.adjacency_matrix = np.full(
            [self.n_points, self.n_points], False, dtype=bool
        )

    def _init_image(self, img):
        """Initializes the image used by the solver as a target and the canvas
        where the solver draws the final image.
        
        Args:
            img (numpy.ndarray): The input image.
        
        Returns:
            None
        """
        self.img = self._prepare_img(img)

        contour_dx, contour_dy = self.contour.get_extension()  # in mm
        self.og_img_height, self.og_img_width = self.img.shape

        scale_x = contour_dx * self.dpmm / self.og_img_width
        scale_y = contour_dy * self.dpmm / self.og_img_height
        self.img_scale_factor = max(scale_x, scale_y)

        # Resize the image
        # adding plus one to avoid out of bound in line tracer.
        # In short, image must be one larger than max index
        # Using skimage, since cv2.resize looks bad for upscale
        img = transform.resize(
            self.img,
            (
                round(self.og_img_height * self.img_scale_factor + 1),
                round(self.og_img_width * self.img_scale_factor + 1),
            ),
            anti_aliasing=True,
        )  # type: ignore

        # internal working copy that is rescaled and will be modified
        _img_solver_target = np.asarray(img, dtype=np.float32)

        # more padding, depending on the line thickness
        # otherwise out of bound error in line tracer
        # future versions could inplement this by discarding oob values
        # in the solver.
        self._error_img = self._add_border_padding(_img_solver_target)

    def _init_image_weights(self, img_weights, weights_importance=1.0):
        """Initializes the image weights for use in the solver.
        
        Args:
            img_weights (numpy.ndarray): An image containing weights for each
                pixel. Darker means less important.
                If None, no weights are used.
            weights_importance (float): A scaling factor for the weights.
                If 0.0, weights will be ignored. Defaults to 1.0. 
        
        Returns:
            None
        """
        if img_weights is None:
            self.img_weights = None
            self._img_solver_weights = None
            return
        img_weights = self._prepare_img(img_weights)
        imgw_height, imgw_width = img_weights.shape
        if (self.og_img_height != imgw_height) or (
            self.og_img_width != imgw_width
        ):
            raise ValueError("Image and weight image must have the same size!")
        img_weights = transform.resize(
            img_weights,
            (
                round(self.og_img_height * self.img_scale_factor + 1),
                round(self.og_img_width * self.img_scale_factor + 1),
            ),
            anti_aliasing=True,
        )  # type: ignore
        img_weights = weights_importance * img_weights + (
            1 - weights_importance
        )
        # internal working copy that is rescaled and will be modified
        _img_solver_weights = np.asarray(img_weights, dtype=np.float32)
        self._img_solver_weights = self._add_border_padding(
            _img_solver_weights
        )

    def _get_coordinates(self, s1):
        """Calculates the coordinates for a given parameter value along the
        contour.

        Args:
            s1 (float): The parameter value along the contour.

        Returns:
            tuple: A tuple containing the x and y coordinates of the point on
                the contour corresponding to s1. The corrdiantes are defined,
                so that (0, 0) are the smallest return values.
        """
        x1, y1 = self.contour.get_coordinates(s1, do_pos_shift=True)
        x1 = x1 * self.dpmm + self.border_padding
        y1 = y1 * self.dpmm + self.border_padding
        return (x1, y1)

    @staticmethod
    @njit
    def _closeness_penalty(s1, s2):
        """Calculates a penalty for points that are too close together.

        The penalty is designed to encourage the solver to place points with
        a certain distance apart. This helps to avoid very short strings.

        Args:
            s1 (float): The first point's value along the contour.
            s2 (float): The second point's value along the contour.

        Returns:
            float: A penalty value, ranging from 0.0 to a maximum based on the 
                   distance between s1 and s2. A smaller distance results in
                   a higher penalty.

        """
        if s1 > s2:
            s2, s1 = s1, s2
        diff = s2 - s1
        if diff > 0.5:
            diff = 1 - diff
        if diff > 0.1:
            return 0.0
        return ((0.1 - diff) * 10) ** 2

    @staticmethod
    @njit
    def _angular_penalty(angle: float):
        """Penalty for angles larger than a threshold.
        Args:
            angle (float): The angle in radians between two points.
        Returns:
            float: The penalty value. 0 if the angle is greater than the
                threshold, otherwise a penalty proportional to the difference
                from the threshold.
        """
        thresh = 0.075
        angle = angle / np.pi
        if angle > thresh:
            return 0.0
        return ((thresh - angle) * 1 / thresh) ** 2

    def _build_target_function(self):
        """Builds and returns the target function for minimization based on the
        solver's mode.

        Raises:
        NotImplementedError: If the solver mode is not implemented.

        Returns:
            callable: The target function for minimization.
            Inputs are two floats with the start and the end of the string with
            values in the range [0,1] that define on wich point of the countur
            the string will be places
        """        

        match self.mode:
            case "greedy_dark":

                def minimization_target(xy1, xy2):
                    # penalty term, if the distance is too small
                    # squared increase from 0 to 1 for values of abs(s2-s1)<0.1
                    error = error_along_line_aa_njit(
                        self._error_img,
                        self._output_canvas,
                        xy1[0],
                        xy1[1],
                        xy2[0],
                        xy2[1],
                        self._line_thickness,
                        self.overlap_penalty,
                        self._img_solver_weights,
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
                return np.random.uniform(low=0.0, high=1.0, size=2)
            case _:
                raise NotImplementedError(
                    f"starting values for mode `{self.mode}` not implemented!"
                )

    def solve_next(self):
        "solving the next string."
        if not self._solved_first:
            return self._solve_first()
        best_fval = np.inf
        best_sidx = None
        f = self._build_target_function()
        # only do connections which do not already exist
        possible_connections = np.arange(self.n_points, dtype=int)[
            ~np.logical_or(
                self.adjacency_matrix[self.curr_sidx, :],
                self.adjacency_matrix[:, self.curr_sidx],
            )
        ]
        possible_connections = possible_connections[
            self.curr_sidx != possible_connections
        ]
        positions = self.contour.get_coordinates(self.s_connections[-2:])
        prev_pos, curr_pos = np.array(positions).T
        prev_vec = prev_pos - curr_pos
        possible_pos = self.contour.get_coordinates(
            self.s_vals[possible_connections]
        )
        possible_vecs = possible_pos.T - curr_pos
        angles = np.arccos(
            np.dot(normalize(prev_vec), normalize(possible_vecs).T)
        )

        _ = f((0, 0), (1, 1))  # run this once as a njit warmup

        scores = [
            f(self.xy_points[self.curr_sidx], self.xy_points[idx])
            + self._closeness_penalty(
                self.s_vals[self.curr_sidx], self.s_vals[idx]
            )
            + self._angular_penalty(angle)
            for idx, angle in zip(possible_connections, angles)
            if not idx == self.curr_sidx
        ]

        best_score_idx = np.argmin(scores)
        best_fval = scores[best_score_idx]
        best_sidx = possible_connections[best_score_idx]

        self.string_count += 1
        print(
            f"adding string {self.string_count:05}: {self.s_vals[self.curr_sidx]:.4f}, {self.s_vals[best_sidx]:.4f}, score: {best_fval:.8f}"
        )
        # draw negative line to remove from target
        self._update_err_img(self.curr_sidx, best_sidx, opacity=self.opacitys)
        old_sidx = self.curr_sidx
        self.curr_sidx = best_sidx
        # pass both for potential future updates
        self.s_connections.append(self.s_vals[self.curr_sidx])
        self._update_output_canvas(old_sidx, self.curr_sidx)
        if old_sidx < self.curr_sidx:
            self.adjacency_matrix[old_sidx, self.curr_sidx] = True
        else:
            self.adjacency_matrix[self.curr_sidx, old_sidx] = True
        self.curr_score = best_fval
        return old_sidx, best_sidx

    def _solve_first(self):
        "solving the first string"
        print("Solving the first string, this may take a while!")
        best_fval = np.inf
        best_sidxs = [0, 0]
        best_conn_idx = 0
        f = self._build_target_function()
        possible_connections = list(
            itertools.combinations(range(self.n_points), 2)
        )
        scores = [
            f(self.xy_points[idx1], self.xy_points[idx2])
            + self._closeness_penalty(self.s_vals[idx1], self.s_vals[idx2])
            for idx1, idx2 in possible_connections
        ]

        best_conn_idx = np.argmin(scores)
        best_fval = scores[best_conn_idx]
        best_sidxs = list(possible_connections[best_conn_idx])

        self.string_count += 1
        print(
            f"adding string {self.string_count:05}: {self.s_vals[best_sidxs[0]]:.4f}, {self.s_vals[best_sidxs[1]]:.4f}, score: {best_fval:.8f}"
        )
        # draw negative line to remove from target
        self._update_err_img(*best_sidxs, opacity=self.opacitys)
        self.curr_sidx = best_sidxs[1]
        self._solved_first = True
        self.s_connections.append(self.s_vals[best_sidxs[0]])
        self.s_connections.append(self.s_vals[best_sidxs[1]])
        if best_sidxs[0] > best_sidxs[1]:
            best_sidxs[0], best_sidxs[1] = best_sidxs[1], best_sidxs[0]
        self.adjacency_matrix[*best_sidxs,] = True

        self._update_output_canvas(
            *best_sidxs,
        )
        self.curr_score = best_fval
        return (*best_sidxs,)

    def _draw_line(self, sidx1, sidx2, img, opacity=None, do_clip=True):
        """Draws an anti-aliased line between two points along the contour

        Args:
            sidx1 (int): line start index of the points along the countour.
            sidx2 (int): line end index of the points along the countour.
            img (np.ndarray): The image that will be modified
            opacity (float, optional): The opacity that will be used.
                Defaults to `self.opacityd`.
            do_clip (bool, optional): If False, the values of the resulting
                image will be [0, inf). If True, then only values in [0, 1]
                are returned. Defaults to True.

        Returns:
            np.ndarray: the resulting image
        """        
        if opacity is None:
            opacity = self.opacityd
        xy1, xy2 = self.xy_points[sidx1], self.xy_points[sidx2]
        x, y, val = wu_line(
            xy1[0], xy1[1], xy2[0], xy2[1], thickness=self._line_thickness
        )
        new_px_vals = img[x, y] + np.array(val) * opacity
        if do_clip:
            new_px_vals = np.clip(new_px_vals, 0.0, 1.0)
        img[x, y] = new_px_vals
        return img

    def _update_err_img(self, sidx1, sidx2, opacity=None):
        """Updating the error image by drawing a new line

        Args:
            sidx1 (int): line start index of the points along the countour.
            sidx2 (int): line end index of the points along the countour.
            opacity (float, optional): Opacity used in the operation.
                Defaults to None.
        """        
        
        if opacity is None:
            opacity = 1.0
        xy1, xy2 = self.xy_points[sidx1], self.xy_points[sidx2]
        x, y, val = wu_line(
            xy1[0], xy1[1], xy2[0], xy2[1], thickness=self._line_thickness
        )
        old_px_vals = self._error_img[x, y]
        add_px_vals = np.array(val) * opacity

        new_px_vals = np.clip(old_px_vals + add_px_vals, None, 1.0)
        self._error_img[x, y] = new_px_vals

    def _update_output_canvas(self, sidx1, sidx2):
        """"update the output canvas by drawing a new line between s1 and s2"

        Args:
            sidx1 (int): line start index of the points along the countour.
            sidx2 (int): line end index of the points along the countour.
        """        
        self._output_canvas = self._draw_line(
            sidx1, sidx2, self._output_canvas, -self.opacityd, do_clip=False
        )

    @property
    def image(self):
        return np.clip(np.rint(self._output_canvas * 255).astype(int), 0, 255)

    def save_img(self, filename):
        cv2.imwrite(filename, self.image)

    def _save_project(self, path=None, n_segments=500, save_stl=True):
        """Legacy code. Nice for debugging but should generally be avoided."""
        if path is None:
            path = f"./{self.name}/"
        path = Path(path)
        path.mkdir(exist_ok=True)
        if save_stl:
            self.contour.save_model(path=path, n_segments=n_segments)
        self.save_img(path / "result.png")
        print("saved:", path / "result.png")

    @property
    def save_path(self):
        return self._save_path

    def save_and_quit_with_dialog(self, event:tk.Event):
        """Starts a filedialog where a save location is selected. The saved the
        canvas, end the mainloop and closes the window.

        Args:
            event (tk.Event):
                unused

        Returns:
            None
        """

        initialdir = "./test_output"
        if not Path(initialdir).absolute().exists():
            initialdir = "."

        self._save_path = (
            Path(
                filedialog.askdirectory(
                    title="Select a save directory!",
                    initialdir=initialdir,
                )
            )
            / self.name
        )
        self._save_path.mkdir(exist_ok=True)
        self.save_img(self._save_path / "result.png")
        plt.close("all")  # all open plots are correctly closed after each run
        self._running = False


if __name__ == "__main__":
    pass
