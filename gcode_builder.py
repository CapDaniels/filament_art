from gcode import GSketch, G1, GString
import cv2 as cv2
import warnings


class GcodeStringer:
    def __init__(self, solver: "Solver", gsketch: GSketch, thickness=0.4, z_hop=1., z_base=None, feed_rate=1000., ramp_angle=45):
        self.solver = solver
        self.gsketch = gsketch
        self.thickness = thickness
        self.z_hop = z_hop
        if z_base is None:
            for coord_name in ['z']:  # may add f later
                last_val = self.gsketch.get_curr_pos([coord_name])[0]
                if last_val is None:
                    warnings.warn(f"No previous {coord_name} value was found, defaulting to 0!")
                self.z_base = 0.0
        else:
            self.z_base = z_base
        self.vf = 600.
        self.hf = feed_rate
        self.ramp_angle = ramp_angle

    def process_Gcode(self):
        s_vals = self.solver.s_connections
        if len(s_vals) < 2:
            raise AttributeError("Cannot generate GCode if solver did not run!")
        prev_gsketch = GSketch._current_gsketch
        GSketch._current_gsketch = self.gsketch
        G1(z=self.z_base + 10, f=self.vf)
        x, y = self.solver.contour.get_coordinates(self.solver.s_connections[0], do_pos_shift=False)
        G1(x=x, y=y, f=self.hf)
        G1(z=self.z_base, f=self.vf)
        for s in self.solver.s_connections[1:]:
            xy = self.solver.contour.get_coordinates(s, do_pos_shift=False)
            GString(self.thickness, x=xy[0], y=xy[1], z_hop=self.z_hop, hf=self.hf, vf=self.vf, ramp_angle=self.ramp_angle)
        GSketch._current_gsketch = prev_gsketch


if __name__ == "__main__":
    from solver_discrete_no_hop import Solver
    from contours import Circle
    circle = Circle((0, 0), 100)
    img = cv2.imread(r".\..\string_art\test_images\4.jpg")
    solver = Solver(circle, img, line_thickness=0.2, dpmm=1.0, n_points=100)
    for i in range(10):
        solver.solve_next()
    gsketch = GSketch("test_main", nozzle_diameter=0.4, filament_diameter=1.75)
    gcode_stringer = GcodeStringer(solver, gsketch)
    gcode_stringer.process_Gcode()
    print(gsketch.get_GCode())
