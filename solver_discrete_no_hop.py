# TODO: weights

import numpy as np
import numpy.typing as npt
from contours import Contour, Circle
import skimage.transform as transform
# from iminuit import Minuit
from line_alg import wu_line
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import itertools
import cv2
import cProfile
import pstats
from gcode import GSketch
from gcode_builder import GcodeStringer
import warnings


img_type = cv2.typing.MatLike


@njit
def darkness_along_line_aa_njit(target, x1, y1, x2, y2, thickness, img_weights=None):
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
    # modifies Wu's algorithm to get the coordinates of the line
    line_x, line_y, weigths = wu_line(x1, y1, x2, y2, thickness)

    # Extract pixel values along the line from both images
    mean_darkness = 0.0
    if img_weights is not None:
        weights_sum = 0
        for idx, (x, y) in enumerate(zip(line_x, line_y)):
            w = img_weights[x, y]
            weights_sum += w
            weigths[idx] *= w
        if weights_sum == 0:
            # otherwise we would divide by zero
            return np.inf

        # weigths *= img_weights
    for x, y, w in zip(line_x, line_y, weigths):
        mean_darkness += target[x, y] * w

    return mean_darkness / sum(weigths)


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

    def __init__(
        self,
        contour: "Contour",
        img: img_type,
        img_weights: img_type | None = None,
        mode="greedy_dark",
        dpmm=12.0,
        line_thickness=0.4,
        opacity=1.0,
        n_points=200,
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
        line_thickness = float(line_thickness)
        dpmm = float(dpmm)
        self.contour = contour
        self.dpmm = dpmm  # dots per millimeter for the internal computation
        self.mode = mode
        self.opacity = (
            opacity  # only used in drawing, but not in finding the best move
        )
        self.overlap_handling = "kink"  # internal solver setting without user accsess
        self.kink_factor = 0.1  # internal solver setting without user accsess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = np.asarray(img, dtype=np.float32) / 255
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
        ) # type: ignore
        self._img = np.asarray(
            img, dtype=np.float32
        )  # internal working copy that is rescaled and will be modified
        # line thickness in dots for internal use
        self._line_thickness = line_thickness * dpmm
        if self._line_thickness < 1.0:
            warnings.warn(f"Your DPI is low for the selected thickness. A value above {1/line_thickness:.2f} is adviced.")
        self._border_padding = int(np.ceil(self._line_thickness / 2)) + 1
        self._img = cv2.copyMakeBorder(
            self._img,
            self._border_padding,
            self._border_padding,
            self._border_padding,
            self._border_padding,
            cv2.BORDER_REFLECT,
        )
        if img_weights is not None:
            img_weights = cv2.cvtColor(img_weights, cv2.COLOR_BGR2GRAY)
            self.img_weights = np.asarray(img_weights, dtype=np.float32) / 255
            imgw_height, imgw_width = img_weights.shape
            if (img_height != imgw_height) or (img_width != imgw_width):
                raise ValueError("Image and weight image must have the same size!")
            img_weights = transform.resize(
                img_weights,
                (
                    round(img_height * self.img_scale_factor + 1),
                    round(img_width * self.img_scale_factor + 1),
                ),
                anti_aliasing=True,
            ) # type: ignore
            self._img_weights = np.asarray(
                img_weights, dtype=np.float32
            )  # internal working copy that is rescaled and will be modified
            self._img_weights = cv2.copyMakeBorder(
                self._img_weights,
                self._border_padding,
                self._border_padding,
                self._border_padding,
                self._border_padding,
                cv2.BORDER_REFLECT,
            )
        else:
            self.img_weights = None
            self._img_weights = None
        self._img_line_count = self._img.copy()
        self.string_count = 0
        self.n_points = n_points
        self.s_vals = np.linspace(0.0, 1.0, n_points+1, endpoint=False)
        self.xy_points = np.array([self._get_coordinates(s) for s in self.s_vals])
        self.curr_sidx = None
        self._solved_first = False
        self.s_connections = []
        self._img_canvas = np.ones(self._img.shape)  # to draw the resulting image


    def _get_coordinates(self, s1):
        # getting the shifted corrdinates, so that the min is [0, 0]
        x1, y1 = self.contour.get_coordinates(s1, do_pos_shift=True)
        # x2, y2 = self.contour.get_coordinates(s2, do_pos_shift=True)
        x1 = x1 * self.dpmm + self._border_padding
        y1 = y1 * self.dpmm + self._border_padding
        # x2 = x2 * self.dpmm + self._border_padding
        # y2 = y2 * self.dpmm + self._border_padding
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
        return ((0.1 - diff) * 10)**2

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
                    error = (
                        darkness_along_line_aa_njit(
                            self._img, xy1[0], xy1[1], xy2[0], xy2[1], self._line_thickness, self._img_weights
                        )
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
        scores = [f(self.xy_points[self.curr_sidx], self.xy_points[idx]) + self._closeness_penalty(self.s_vals[self.curr_sidx], self.s_vals[idx]) for idx in range(self.n_points) if not idx == self.curr_sidx]

        best_sidx = np.argmin(scores)
        best_fval = scores[best_sidx]

        self.string_count += 1
        print(
            f"adding string {self.string_count:05}: {self.s_vals[self.curr_sidx]:.4f}, {self.s_vals[best_sidx]:.4f}, score: {best_fval:.8f}"
        )
        # draw negative line to remove from target
        self._update_img(self.curr_sidx, best_sidx, opacity=self.opacity)
        old_sidx = self.curr_sidx
        self.curr_sidx = best_sidx
        # pass both for potential future updates
        self.s_connections.append(self.s_vals[self.curr_sidx])
        self.update_img_canvas(old_sidx, self.curr_sidx)
        return old_sidx, best_sidx

    def _solve_first(self):
        # find best connection where s1idx and s2idx are free
        print("Solving the first string, this may take a while!")
        best_fval = np.inf
        best_sidxs = [0, 0]
        best_conn_idx = 0
        f = self._build_target_function()
        possible_connections = list(itertools.combinations(range(self.n_points), 2))
        scores = [f(self.xy_points[idx1], self.xy_points[idx2]) + self._closeness_penalty(self.s_vals[idx1], self.s_vals[idx2]) for idx1, idx2 in possible_connections]

        best_conn_idx = np.argmin(scores)
        best_fval = scores[best_conn_idx]
        best_sidxs = possible_connections[best_conn_idx]

        self.string_count += 1
        print(
            f"adding string {self.string_count:05}: {self.s_vals[best_sidxs[0]]:.4f}, {self.s_vals[best_sidxs[1]]:.4f}, score: {best_fval:.8f}"
        )
        # draw negative line to remove from target
        self._update_img(*best_sidxs, opacity=self.opacity)
        self.curr_sidx = best_sidxs[1]
        self._solved_first = True
        self.s_connections.append(self.s_vals[best_sidxs[0]])
        self.s_connections.append(self.s_vals[best_sidxs[1]])
        self.update_img_canvas(*best_sidxs,)
        return *best_sidxs,

    def _update_img(self, sidx1, sidx2, opacity=None):
        if opacity is None:
            opacity = self.opacity
        xy1, xy2 = self.xy_points[sidx1], self.xy_points[sidx2]
        x, y, val = wu_line(xy1[0], xy1[1], xy2[0], xy2[1], thickness=self._line_thickness)
        old_px_vals = self._img[x, y]
        add_px_vals = np.array(val) * opacity
        match self.overlap_handling:
            case "clip":
                new_px_vals = np.clip(old_px_vals + add_px_vals, None, 1.0)
                self._img[x, y] = new_px_vals
            case "linear":
                new_px_vals = old_px_vals + add_px_vals
                self._img[x, y] = new_px_vals
            case "kink":
                def func(x):
                    return np.where(x < 1.0, x , (1 - self.kink_factor) + x * self.kink_factor)
                self._img_line_count[x, y] += add_px_vals
                self._img[x, y] = func(self._img_line_count[x, y])

    def draw_line(self, sidx1, sidx2, img, opacity=None, do_clip=True):
        if opacity is None:
            opacity = self.opacity
        xy1, xy2 = self.xy_points[sidx1], self.xy_points[sidx2]
        x, y, val = wu_line(xy1[0], xy1[1], xy2[0], xy2[1], thickness=self._line_thickness)
        new_px_vals = img[x, y] + np.array(val) * opacity
        if do_clip:
            new_px_vals = np.clip(new_px_vals, 0.0, 1.0)
        img[x, y] = new_px_vals
        return img
    
    def update_img_canvas(self, s1, s2):
        self._img_canvas = self.draw_line(
            s1, s2, self._img_canvas, -self.opacity
        )
    
    @property
    def image(self):
        return np.clip(np.rint(self._img_canvas*255).astype(int), 0, 255)


class SolverGUI(Solver):
    def __init__(
        self,
        contour: "Contour",
        img: img_type,
        img_weights: img_type | None = None,
        mode="greedy_dark",
        dpmm=12.0,
        line_thickness=0.4,
        opacity=1.0,
        n_points=200
    ):
        super().__init__(
            contour=contour,
            img=img,
            img_weights=img_weights,
            mode=mode,
            dpmm=dpmm,
            line_thickness=line_thickness,
            opacity=opacity,
            n_points=n_points
        )
        # 1 = black
        # Create figure and subplots
        # plt.ion()
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        for i, ax in enumerate(self.axs.flat):
            ax.set_title(f"Plot {i+1}")
        self.mpl_og_img = self.axs[0, 0].imshow(
            self.img, cmap="gray", vmin=0, vmax=1
        )
        self.mpl_img = self.axs[0, 1].imshow(
            self._img, cmap="gray", vmin=0, vmax=1
        )
        self.mpl_img_canvas = self.axs[1, 0].imshow(
            self._img_canvas, cmap="gray", vmin=0, vmax=1
        )
        if self._img_weights is not None:
            self.axs[1, 1].imshow(
                self._img * self._img_weights, cmap="gray"
            )

        # Adjust layout to make room for the button
        self.fig.subplots_adjust(bottom=0.2)

        # Create a button below the subplots
        button_labels = ["1", "100", "500", "1000"]
        self._buttons = []
        for i, label in enumerate(button_labels):
            ax_button = plt.axes(
                (0.1 + i * 0.2, 0.05, 0.15, 0.05)
            )  # [left, bottom, width, height]
            button = widgets.Button(ax_button, label)
            button.on_clicked(self._next_lines_GUI_call)
            button.ax.set_label(label)
            self._buttons.append(button)

    def start_gui(self):
        plt.show(block=True)

    def solve_n_lines(self, n=1):
        for i in range(n):
            self.solve_next()
            self._update_gui()

    def _next_lines_GUI_call(self, event):
        n_calls = int(event.inaxes.get_label())
        self.solve_n_lines(n_calls)

    def _update_gui(self):
        self.mpl_img_canvas.set_data(self._img_canvas)
        self.mpl_img.set_data(self._img)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    circle = Circle((0, 0), 100)
    img = cv2.imread(
        r".\..\string_art\test_images\Snoopy_Peanuts.png"
    )
    img_weights = None
    # img_weights = cv2.imread(
    #     r".\..\string_art\test_images\11_mask.jpg"
    # )
    solver = SolverGUI(circle, img, img_weights=img_weights, line_thickness=0.2, dpmm=5.0, n_points=500)
    solver.start_gui()
    print(solver.s_connections)
    # solver.solve_next()
    # cProfile.run("solver.solve_next()", 'restats')
    # p = pstats.Stats('restats')
    # # p.strip_dirs().sort_stats(-1).print_stats()
    # p.sort_stats('cumulative').print_stats(100)
    # p.print_stats()
    gsketch = GSketch("test_main", nozzle_diameter=0.4, filament_diameter=1.75)
    gcode_stringer = GcodeStringer(solver, gsketch)
    gcode_stringer.process_Gcode()
    print(gsketch.get_GCode())


