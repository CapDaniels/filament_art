# TODO: solve circular import

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Annotated, TypeVar, Literal
import model_maker
from pathlib import Path

DType = TypeVar("DType", bound=np.generic)


class Contour(ABC):
    def __init__(self, name="shape", width=5.0, height=5.0):
        self.name = name
        self.width = width  # for the stl
        self.height = height  # for the stl

    def get_coordinates(self, s: float | npt.ArrayLike, do_pos_shift=False):
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
            contourr extension in mm
            npt.NDArray[np.float64]: [size_x, size_y]
        """
        pass

    def save_model(self, path=None, width=None, height=None, n_segments=500):
        width = width or self.width
        height = height or self.height
        self.stl_saver = model_maker.Stl_maker(lambda s:self.get_coordinates(s, do_pos_shift=False), width, height, n_segments)
        if path is None:
            path = self.name + ".stl"
        else:
            path = Path(path)
            if path.suffix == "":
                path = path / (self.name + ".stl")
        self.stl_saver.save_stl(path)
        print("saved:", path)


class Circle(Contour):
    def __init__(self, center, radius, name="circle", widht=5.0, height=5.0):
        self.center = center
        self.radius = radius
        super().__init__(name, width=widht, height=height)

    def _get_coordinates(self, s, do_pos_shift):
        if not do_pos_shift:
            return np.array(
                [
                    np.cos(s * 2 * np.pi) * self.radius + self.center[0],
                    np.sin(s * 2 * np.pi) * self.radius + self.center[1],
                ]
            )
        else:
            return np.array(
                [
                    np.cos(s * 2 * np.pi) * self.radius + self.radius,
                    np.sin(s * 2 * np.pi) * self.radius + self.radius,
                ]
            )

    def get_extension(self):
        return np.array([self.radius * 2, self.radius * 2])


if __name__ == "__main__":
    circle = Circle((110, 110), 70)
    # print(circle.get_coordinates(np.array([0, 0.4, 1])))

    assert np.allclose(
        np.array([[180.0, 53.36881039, 180.0], [110.0, 151.14496766, 110.0]]),
        circle.get_coordinates(np.array([0, 0.4, 1])),
    )
    print(circle.get_extension())
