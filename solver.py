import numpy as np
import numpy.typing as npt
from contours import Contour, Circle
import skimage.draw as draw
import skimage.transform as transform
from iminuit import Minuit
from line_alg import wu_line
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
# from skopt import gp_minimize
# from skopt.plots import plot_convergence
from scipy.optimize import basinhopping

img_type = npt.NDArray[np.uint8]


# vline_aa = np.vectorize(draw.line_aa)

@njit
def darkness_along_line_aa_njit(target, x1, y1, x2, y2, thickness):
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
    line_x, line_y, val_aa = wu_line(x1, y1, x2, y2, thickness)

    # Extract pixel values along the line from both images
    mean_darkness = 0.0
    for x, y, val in zip(line_x, line_y, val_aa):
        mean_darkness += target[x, y] * val

    return mean_darkness / sum(val_aa)

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
    # modifies Wu's algorithm to get the coordinates of the line
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

    def __init__(
        self, contour: "Contour", img: img_type, mode="greedy_dark", dpmm=12.0, line_thickness = 0.4, opacity=1.0
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
        self.opacity = opacity  # only used in drawing, but not in finding the best move
        self.do_clip = True  # tunining variable without user accsess
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
        )
        self._img = np.asarray(
            img, dtype=np.float32
        )  # internal working copy that is rescaled and will be modified
        self._line_thickness = line_thickness * dpmm
        self._border_padding = int(np.ceil(self._line_thickness/2)) + 1
        self._img = cv2.copyMakeBorder(self._img, self._border_padding, self._border_padding, self._border_padding, self._border_padding, cv2.BORDER_REFLECT)
        self.string_count = 0

    def _get_coordinates(self, s1, s2):
        # getting the shifted corrdinates, so that the min is [0, 0]
        x1, y1 = self.contour.get_coordinates(
            s1, do_pos_shift=True
        )
        x2, y2 = self.contour.get_coordinates(
            s2, do_pos_shift=True
        )
        x1 = x1 * self.dpmm + self._border_padding
        y1 = y1 * self.dpmm + self._border_padding
        x2 = x2 * self.dpmm + self._border_padding
        y2 = y2 * self.dpmm + self._border_padding
        return x1, y1, x2, y2

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

                # def minimization_target(
                #     s1: float | npt.ArrayLike, s2: float | npt.ArrayLike
                # ):
                def minimization_target(s: tuple[float, float]):
                    s1, s2 = s
                    x1, y1, x2, y2 = self._get_coordinates(s1,s2)
                    # penalty term, if the distance is too small
                    # squared increase from 0 to 1 for values of abs(s2-s1)<0.1
                    penalty = (
                        np.clip((1 - (np.abs((s1%1) - (s2%1)))) * 10 - 9, 0, None) ** 2
                    )
                    error = (
                        darkness_along_line_aa_njit(self._img, x1, y1, x2, y2, self._line_thickness)
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
                return np.random.uniform(low=0.0, high=1.0, size=2)
            case _:
                raise NotImplementedError(
                    f"starting values for mode `{self.mode}` not implemented!"
                )

    def solve_next(self):
        # m = Minuit(self._build_target_function(), s1=self.s0[0], s2=self.s0[1])
        # m.limits["s1", "s2"] = (0, 1)
        # m.simplex()
        # m.migrad()
        # if m.fval > 0.8:
        #     print(m)
        #     raise RuntimeError("Error in the iminuit minimization")

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds" : ((0, 1), (0, 1))}
        # print("running minimizer!")
        best_x, best_fval = [0, 0], np.inf
        # run this in an external loop to update the staring value of the minimizer
        for i in range(100):
            res = basinhopping(self._build_target_function(), self.s0, minimizer_kwargs=minimizer_kwargs, niter=0, T=.2, stepsize=.4, disp=False)
            if res.fun < best_fval:
                best_fval = res.fun
                best_x = res.x

        self.string_count += 1
        print(f"adding string {self.string_count:05}: {best_x[0]:.4f}, {best_x[1]:.4f}, score: {best_fval:.8f}")
        # draw negative line to remove from target
        self._img = self._update_img(*best_x, self._img, opacity=self.opacity)
        return *best_x, 

    def _update_img(self, s1, s2, img, opacity=None):
        if opacity is None: opacity = self.opacity
        x1, y1, x2, y2 = self._get_coordinates(s1, s2)
        x, y, val = wu_line(x1, y1, x2, y2, thickness=self._line_thickness)
        new_px_vals = img[x, y] + np.array(val) * opacity
        if self.do_clip:
            new_px_vals = np.clip(new_px_vals, 0., 1.)
        img[x, y] = new_px_vals
        return img


class SolverGUI(Solver):
    def __init__(self, contour: "Contour", img: img_type, mode="greedy_dark", dpmm=12.0, line_thickness = 0.4, opacity=1.0):
        super().__init__(contour = contour, img = img, mode=mode, dpmm=dpmm, line_thickness = line_thickness, opacity=opacity)
        # 1 = black
        self._img_canvas = np.ones(self._img.shape)
        # Create figure and subplots
        # plt.ion()
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        for i, ax in enumerate(self.axs.flat):
            ax.set_title(f"Plot {i+1}")
        self.mpl_og_img = self.axs[0, 0].imshow(self.img, cmap="gray", vmin =0, vmax=1)
        self.mpl_img = self.axs[0, 1].imshow(self._img, cmap="gray", vmin =0, vmax=1)
        self.mpl_img_canvas = self.axs[1, 0].imshow(self._img_canvas, cmap="gray", vmin =0, vmax=1)

        # Adjust layout to make room for the button
        self.fig.subplots_adjust(bottom=0.2)

        # Create a button below the subplots
        button_labels = ['1', '10', '100', '1000']
        self._buttons = []
        for i, label in enumerate(button_labels):
            ax_button = plt.axes([0.1 + i * 0.2, 0.05, 0.15, 0.05])  # [left, bottom, width, height]
            button = widgets.Button(ax_button, label)
            button.on_clicked(self._next_line_GUI_call)
            button.ax.set_label(label)
            self._buttons.append(button)

    def start_gui(self):
        plt.show(block=True)

    def _next_line_GUI_call(self, event):
        n_calls = int(event.inaxes.get_label())
        for i in range(n_calls):
            s1, s2 = self.solve_next()
            self.update_img_canvas(s1, s2)

    def update_img_canvas(self, s1, s2):
        self._img_canvas = self._update_img(s1, s2, self._img_canvas, -self.opacity)
        self.mpl_img_canvas.set_data(self._img_canvas)
        self.mpl_img.set_data(self._img)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    



if __name__ == "__main__":
    import cv2

    circle = Circle((0, 0), 100)
    img = cv2.imread(
        r"C:\Users\CapDaniels\Meine Ablage\Documents\CodingProjects\pythonProjects\string_art\test_images\4.jpg"
    )

    # solver = Solver(circle, img)
    # solver.solve_next()
    solver = SolverGUI(circle, img, line_thickness=0.4, dpmm=10.0)
    solver.start_gui()
    # solver.next_line('')

