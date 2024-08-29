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

# import time


img_type = cv2.typing.MatLike


def normalize(v):
    v = np.array(v)
    if v.ndim == 1:
        return v / np.sqrt(np.sum(np.square(v), axis=-1))
    return v / np.sqrt(np.sum(np.square(v), axis=-1))[:, np.newaxis]


@njit
def line_overlap_penalty_f(x, overlap_penalty=0.2):
    """_summary_

    Args:
        x (density): how many aa-lines are at one pixel

    Returns:
        float: the penalty value of one pixel
    """

    if x == 1.0:
        return 0.0
    return -(x - 1) * overlap_penalty


@njit()
def darkness_along_line_aa_njit(
    target,
    canvas,
    x1,
    y1,
    x2,
    y2,
    thickness,
    overlap_penalty=0.2,
    img_weights=None,
):
    """
    Calculate the average darkness along a line in the target image.

    This function uses Bresenham's algorithm to get the coordinates of a line
    between two points and then calculates the average pixel value along that
    line, weighted by anti-aliasing factors.

    Parameters:
    -----------
    target : np.ndarray
        The target image in which darkness is measured.
    convas : np.ndarray
        The current canvas. Ranges from [1, -inf).
    x1, y1 : int
        The starting coordinates of the line.
    x2, y2 : int
        The ending coordinates of the line.

    Returns:
    --------
    mean_darkness : float
        The average darkness along the line.
    """
    # modifies Wu's algorithm to get the coordinates of the line
    # weights here mean the alpha values of the drawn line while img_weights are
    # the weights of the image mask
    line_x, line_y, aa_vals = wu_line(x1, y1, x2, y2, thickness)

    # if len(line_x) == 0:
    #     return np.inf

    # can be thought of as the remaining darkness along the line
    # this is a minimization problem, so lower is better
    score = 0.0
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
        score += target[x, y] * w
        line_denstiy_penalty += line_overlap_penalty_f(
            canvas[x, y], overlap_penalty
        )

    weights_sum = sum(weights)
    if weights_sum == 0.0:
        return np.inf

    return (
        score / (sum(weights))
        + line_denstiy_penalty / len(line_x)
        + weight_penalty
    )


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
            self._img_solver_target.shape
        )  # to draw the resulting image

    def _prepare_img(self, img):
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.asarray(img, dtype=np.float32) / 255

    def _add_border_padding(self, img: img_type, padd: int | None = None):
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
        self._img_solver_target = self._add_border_padding(_img_solver_target)

    def _init_image_weights(self, img_weights, weights_importance=1.0):
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
        # getting the shifted corrdinates, so that the min is [0, 0]
        x1, y1 = self.contour.get_coordinates(s1, do_pos_shift=True)
        x1 = x1 * self.dpmm + self.border_padding
        y1 = y1 * self.dpmm + self.border_padding
        return (x1, y1)

    @staticmethod
    @njit
    def _closeness_penalty(s1, s2):
        # penalty term, if the distance is too small
        # squared increase from 0 to 1 for values of abs(s2-s1)<0.1
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
    def _angular_penalty(angle):
        """Penalty for angles larger than a threshold.
        Args:
            angle (float): The angle in radians between two points.
        Returns:
            float: The penalty value. 0 if the angle is greater than the threshold,
                   otherwise a penalty proportional to the difference from the threshold.
        """
        thresh = 0.075
        angle = angle / np.pi
        if angle > thresh:
            return 0.0
        return ((thresh - angle) * 1 / thresh) ** 2

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

                def minimization_target(xy1, xy2):
                    # penalty term, if the distance is too small
                    # squared increase from 0 to 1 for values of abs(s2-s1)<0.1
                    error = darkness_along_line_aa_njit(
                        self._img_solver_target,
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
        self._update_img(self.curr_sidx, best_sidx, opacity=self.opacitys)
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
        # find best connection where s1idx and s2idx are free
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
        self._update_img(*best_sidxs, opacity=self.opacitys)
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

    def _update_img(self, sidx1, sidx2, opacity=None):
        if opacity is None:
            opacity = 1.0
        xy1, xy2 = self.xy_points[sidx1], self.xy_points[sidx2]
        x, y, val = wu_line(
            xy1[0], xy1[1], xy2[0], xy2[1], thickness=self._line_thickness
        )
        old_px_vals = self._img_solver_target[x, y]
        add_px_vals = np.array(val) * opacity

        new_px_vals = np.clip(old_px_vals + add_px_vals, None, 1.0)
        self._img_solver_target[x, y] = new_px_vals

    def _update_output_canvas(self, s1, s2):
        self._output_canvas = self._draw_line(
            s1, s2, self._output_canvas, -self.opacityd, do_clip=False
        )

    @property
    def image(self):
        return np.clip(np.rint(self._output_canvas * 255).astype(int), 0, 255)

    def save_img(self, filename):
        cv2.imwrite(filename, self.image)

    def save_project(self, path=None, n_segments=500, save_stl=True):
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

    def save_and_quit_with_dialog(self, event):
        self._save_path = (
            Path(
                filedialog.askdirectory(
                    title="Select a save directory!",
                    initialdir="./test_output",
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
