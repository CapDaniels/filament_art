import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from collections.abc import MutableSequence
import pathlib
import contours
import warnings


def L2_norm(v1: npt.NDArray, v2: npt.NDArray):
    return np.sqrt(np.sum(np.square(v2 - v1), axis=-1))


class GSketch(MutableSequence):
    _current_gsketch = None

    def __init__(self, name: str, nozzle_diameter=0.4, filament_diameter=1.75):
        if not name.isidentifier():
            raise ValueError(
                "Invalid name. Names should follow the same rules as python variables!"
            )
        self.name = name
        self.nozzle_diameter = nozzle_diameter
        self.filament_diameter = filament_diameter
        self.list = list()
        GSketch._current_gsketch = self

    def check(self, gcode: "GCode") -> None:
        if not isinstance(gcode, GCode):
            raise ValueError("Can only apped objects that derive from GCode!")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        del self.list[i]

    def __setitem__(self, i, gcode: "GCode"):
        self.check(gcode)
        self.list[i] = gcode

    def insert(self, i, gcode: "GCode"):
        self.check(gcode)
        self.list.insert(i, gcode)

    @property
    def extruder_ratio(self):
        return (self.filament_diameter / self.nozzle_diameter) ** 2

    def get_curr_pos(self, var_names: list[str] | None = None) -> list[float | None]:
        if var_names is None:
            var_names = ["X", "Y", "Z"]
        var_names = [var_name.upper() for var_name in var_names]
        values = [None] * len(var_names)

        for code in self.list[::-1]:
            command_attr = code.command_attr
            for idx, var_name in enumerate(var_names):
                # only overwrite var with value if var is None and var_name is in command_attr.keys()
                if values[idx] is None:
                    values[idx] = command_attr.get(var_name, None)  
            if not (None in values):
                break
        return values

    def get_GCode(self) -> str:
        gout = [code_obj.getGstring() for code_obj in self.list]
        gout = "\n".join(gout)
        return gout + "\n"

    def save_GCode(self, path: str | pathlib.PurePath):
        with open(path, "w") as f:
            f.writelines(self.get_GCode())
    
    @classmethod
    def gcgs(cls):
        "Get Current GSketch"
        if cls._current_gsketch is None:
            raise ValueError("Getting current GSketch before GSketch is initialized!")
        return cls._current_gsketch


class GCode(ABC):
    def __init__(self):
        if GSketch._current_gsketch is None:
            raise AttributeError("You need to initialize a GSketch first!")
        GSketch._current_gsketch.append(self)

    def getGstring(self):
        g_str = self.command_name
        for c_attr_name, c_attr_val in self.command_attr.items():
            if c_attr_name == "E":
                g_str += f" {c_attr_name}{c_attr_val:.5f}"
            else:
                g_str += f" {c_attr_name}{c_attr_val:.3f}"
        return g_str

    @property
    @abstractmethod
    def command_name(self) -> str:
        pass

    @property
    @abstractmethod
    def command_attr(self) -> dict[str, float]:
        pass

    def __repr__(self) -> str:
        return (
            f"Gcode command: {self.command_name}, Gcode Attributes: {self.command_attr}"
        )


class G1(GCode):
    _command_name = "G1"

    def __init__(self, x=None, y=None, z=None, e=None, f=None):
        self._command_attr = {}
        names, vals = ["X", "Y", "Z", "E", "F"], [x, y, z, e, f]
        for name, val in zip(names, vals):
            if val is not None:
                self._command_attr[name] = val

        # check if any argument (besides f) is set
        if len(self._command_attr) == 0:
            raise ValueError("You need to define at least any of x, y, z, e or f!")

        super().__init__()

    @property
    def command_name(self):
        return self._command_name

    @property
    def command_attr(self):
        return self._command_attr


class GThinStretch(G1):
    # could be also implemented as a classmethod, but like the separation
    def __init__(self, thickness, x=None, y=None, z=None, f=None):
        gsketch = GSketch.gcgs()

        if not (x or y or z):
            raise ValueError("You need to define at least any of x, y or z!")

        if thickness > 2 * gsketch.nozzle_diameter:
            raise ValueError(
                "Strings should not be wider than 2 times the nozzle diameter!"
            )

        pos_old = np.array(gsketch.get_curr_pos())
        x_new = x or pos_old[0]
        y_new = y or pos_old[1]
        z_new = z or pos_old[2]
        pos_new = np.array([x_new, y_new, z_new])
        dist = L2_norm(pos_old, pos_new)

        # distance times the fraction of target area and filament cross section area  (shortend some terms)
        e = dist * (thickness / gsketch.filament_diameter) ** 2

        super().__init__(x=x_new, y=y_new, z=z_new, e=e, f=f)


def GString(
    thickness: float, x: float | None = None, y: float | None = None, z_hop=0.4, vf=600., hf=1000., ramp_angle=45
):
    if (ramp_angle <= 0) or (ramp_angle > 90):
        raise ValueError("`ramp_angle` must be in teh interval [0, 90]!")
    ramp_angle = np.radians(ramp_angle)
    z_base = GSketch.gcgs().get_curr_pos(["Z"])[0]
    if z_base is None:
        warnings.warn("No prior z-height found. Defaulting to 0!")
        z_base = 0.0

    pos_old = np.array(GSketch.gcgs().get_curr_pos(["X", "Y"]))
    x_new = x or pos_old[0]
    y_new = y or pos_old[1]
    pos_new = np.array([x_new, y_new])
    def path(s):
        return pos_old + (pos_new - pos_old) * s
    s1 = z_hop / np.tan(ramp_angle) / L2_norm(pos_new, pos_old)
    s2 = 1 - s1
    reached_full_height = True
    if s1 > 0.5:
        reached_full_height = False
        s1, s2 = 0.5, 0.5
        z_hop = (L2_norm(pos_new, pos_old) / 2) * np.tan(ramp_angle)
    x1, y1 = path(s1)
    GThinStretch(thickness, x=x1, y=y1, z=z_base + z_hop, f=hf)
    x2, y2 = path(s2)
    if reached_full_height:
        GThinStretch(thickness, x=x2, y=y2, f=hf)
    GThinStretch(thickness, x=x_new, y=y_new, z=z_base, f=hf)




def GFollowContour(contour: contours.Contour, s1: float, s2: float, stepsize=0.0025):
    if s1 < s2:
        s_dist = s2 - s1
        if s_dist > 0.5:
            stepsize = -stepsize
            s1 += 1
    else:
        s_dist = s1 - s2
        stepsize = -stepsize
        if s_dist > 0.5:
            stepsize = -stepsize
            s2 += 1
    s_vals = np.append(np.arange(s1, s2, step=stepsize), [s2]) % 1
    xs, ys = contour.get_coordinates(s_vals)
    for x, y in zip(xs, ys):
        G1(x=x, y=y)


if __name__ == "__main__":
    gsketch = GSketch("test_main", nozzle_diameter=0.4, filament_diameter=1.75)
    G1(x=3, y=2)
    G1(z=1)
    G1(x=4)
    GThinStretch(0.2, 5, 5, 5)
    gcode_str = gsketch[-1].getGstring()

    target_test_gcode = "G1 X5.000 Y5.000 Z5.000 E0.06660"
    print(gsketch.get_GCode())
    assert gcode_str == target_test_gcode

    circle = contours.Circle((0, 0), 10)
    GFollowContour(circle, 0.5, 0.25)
    # print(GFollowContour(circle, 0.25, 0.5))
    # print(GFollowContour(circle, 0.25, 0.99))
    # print(GFollowContour(circle, 0.99, 0.25))
    print(gsketch.get_GCode())

    print(L2_norm(np.array([[1,1,0], [10,0,0]]), np.zeros((2,3))))
