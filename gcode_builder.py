from solver_discrete_no_hop import Solver
from gcode import GSketch, G1, GString
from contours import Circle
import cv2 as cv2


class GcodeStringer:
    def __init__(self, solver: "Solver", gsketch: GSketch, thickness=0.4, z_hop=1):
        self.solver = solver
        self.GSketch = gsketch
        self.thickness = thickness
        self.z_hop = z_hop

    def process_Gcode(self):
        s_vals = self.solver.s_connections
        if len(s_vals) < 2:
            raise AttributeError("Cannot generate GCode if solver did not run!")
        prev_gsketch = GSketch._current_gsketch
        # TODO init movemnet
        for s in self.solver.s_connections[1:]:
            xy = solver.contour.get_coordinates(s, do_pos_shift=False)
            print(s, xy)
            # TODO: fix behavor if z_base is not defined when calling Gstring
            GString(self.thickness, x=xy[0], y=xy[1], z_hop=self.z_hop)
        GSketch._current_gsketch = prev_gsketch


if __name__ == "__main__":
    circle = Circle((0, 0), 100)
    img = cv2.imread(r".\..\string_art\test_images\4.jpg")
    solver = Solver(circle, img, line_thickness=0.2, dpmm=10.0, n_points=500)
    for i in range(10):
        solver.solve_next()
    gsketch = GSketch("test_main", nozzle_diameter=0.4, filament_diameter=1.75)
    gcode_stringer = GcodeStringer(solver, gsketch)
    gcode_stringer.process_Gcode()
    print(gsketch.get_GCode())
