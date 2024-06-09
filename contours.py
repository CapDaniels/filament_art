from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class Contour(ABC):
    def __init__(self, name="shape"):
        self.name = name

    def get_coordinates(self, s: float | npt.ArrayLike, do_pos_shift=False):
        s = np.array(s)
        if np.any(s < 0) or np.any(s > 1):
            raise ValueError("s must be in the intervall [0,1]!")
        return self._get_coordinates(s, do_pos_shift)

    @abstractmethod
    def _get_coordinates(self, s, do_pos_shift):
        pass

    @abstractmethod
    def get_extension(self) -> npt.NDArray[np.float64]:
        """returns the cardianal corrdiante extension of the contour

        Returns:
            contourr extension in mm
            npt.NDArray[np.float64]: [size_x, size_y]
        """
        pass


class Circle(Contour):
    def __init__(self, center, radius, name="cicle"):
        self.center = center
        self.radius = radius
        super().__init__(name)

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
