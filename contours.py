# TODO: solve circular import

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Annotated, TypeVar, Literal, Tuple
import model_maker
from pathlib import Path

DType = TypeVar("DType", bound=np.generic)


class Contour(ABC):
    def __init__(self, name="contour", width=5.0, height=5.0):
        """A contour object, that will be used a the frame. 

        Args:
            name (str, optional): the name of the object. Defaults to name of
                class.
            width (float, optional): width of the frame crossection.
                Defaults to 5.0.
            height (float, optional): height of the frame crossection.
                Defaults to 5.0.
        """        
        self.name = name
        self.width = width  # for the stl
        self.height = height  # for the stl

    def get_coordinates(self, s: float | npt.ArrayLike, do_pos_shift=False):
        """Get the xy-coordinated of a poition along the contour.

        Args:
            s (float | npt.ArrayLike): the position. Must be in range [0, 1].
            do_pos_shift (bool, optional): If `False`: the center of the contour
                is (0, 0). If `True`: translates the contour, so that the
                minimum of the returned coordinates is 0, respectively. Defaults
                to False.

        Raises:
            ValueError: If s is outside the interval [0, 1].

        Returns:
            np.ndarray: the corrdinates
        """        
        s = np.array(s)
        if np.any(s < 0) or np.any(s > 1):
            raise ValueError("s must be in the intervall [0,1]!")
        return self._get_coordinates(s, do_pos_shift)

    @abstractmethod
    def _get_coordinates(
        self, s, do_pos_shift
    ) -> (
        Annotated[npt.NDArray[DType], Literal[2, "N"]]
        | Annotated[npt.NDArray[DType], Literal[2]]
    ):
        pass

    @abstractmethod
    def get_extension(self) -> npt.NDArray[np.float64]:
        """returns the cardianal corrdiante extension of the contour

        Returns:
            np.ndarray: contourr extension in mm (size_x, size_y)
        """
        pass

    def save_model(self, path=None, n_segments=500):
        """Saving the model as a `stl` file.

        Args:
            path (str | os.PathLike, optional): The savepath. Defaults to
                `./{self.name}.stl`.
            n_segments (int, optional): Number of segments that will be used to
                define the object. Defaults to 500.
        """        
        self.stl_saver = model_maker.Stl_maker(lambda s:self.get_coordinates(s, do_pos_shift=False), self.width, self.height, n_segments)
        if path is None:
            path = self.name + ".stl"
        else:
            path = Path(path)
            if path.suffix == "":
                path = path / (self.name + ".stl")
        self.stl_saver.save_stl(path)
        print("saved:", path)


class Circle(Contour):
    def __init__(self, center: Tuple[float, float], radius:float, name="circle", width=5.0, height=5.0):
        self.center = center
        self.radius = radius
        super().__init__(name, width=width, height=height)

    def _get_coordinates(self, s, do_pos_shift):
        coords = np.array(
            [
                np.cos(s * 2 * np.pi) * self.radius + self.center[0],
                np.sin(s * 2 * np.pi) * self.radius + self.center[1],
            ]
        )
        if do_pos_shift:
            coords = coords - self.center + self.radius
        return coords

    def get_extension(self):
        return np.array([self.radius * 2, self.radius * 2])

class Rect(Contour):
    def __init__(self, center: Tuple[float, float], extension_x:float, extension_y:float, name="rect", width=5.0, height=5.0):
        self.center = np.array(center)
        self.extension_x = extension_x
        self.extension_y = extension_y
        super().__init__(name, width=width, height=height)

    def _get_coordinates(self, s, do_pos_shift):
        s = np.array(s)
        if s.ndim == 0:
            return self._get_coordinate(s, do_pos_shift=do_pos_shift)
        coords = [self._get_coordinate(i, do_pos_shift=do_pos_shift) for i in s]
        coords = np.array(coords).T
        return coords

    def _get_coordinate(self, s, do_pos_shift):
        if s < 0.25:
            x = self.extension_x / 2
            y = - self.extension_y / 2 + self.extension_y * (s/0.25)
        elif s >= 0.25 and s < 0.5:
            x = self.extension_x / 2 - self.extension_x * ((s-0.25)/0.25)
            y = self.extension_y / 2
        elif s >= 0.5 and s < 0.75:
            x = - self.extension_x / 2
            y = self.extension_y / 2 - self.extension_y * ((s-0.5)/0.25)
        else:
            x = - self.extension_x / 2 + self.extension_x * ((s-0.75)/0.25)
            y = - self.extension_y / 2
        coords = np.array((x,y))
        if do_pos_shift:
            coords += np.array([self.extension_x/2, self.extension_y/2])
        else:
            coords += self.center
        return coords

    def get_extension(self):
        return np.array([self.extension_x, self.extension_y])
    
    def save_model(self, path=None, n_segments=None):
        # rect needs per definition only four segments
        # the passed value will be discarded
        return super().save_model(path, n_segments=4)


if __name__ == "__main__":
    pass
    # rect = Rect((0,0), 10, 10)
    # print(rect.get_extension())
    # print(rect.get_coordinates(np.arange(0,1,0.1)))
    # print(rect.get_coordinates(np.arange(0,1,0.1), do_pos_shift=True))
    # model = model_maker.Stl_maker(lambda s:rect.get_coordinates(s, do_pos_shift=False), 5, 5, 4)
    # model.create_model()
    # model.display_model()
