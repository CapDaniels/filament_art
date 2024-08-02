import cv2
from gcode import G1, GFollowContour, GSketch, GThinStretch
import contours
from gcode_builder import GcodeStringer
from solver_discrete_no_hop import SolverGUI, Solver
import numpy as np


if __name__ == "__main__":
    circle = contours.Circle((110, 110), 100)

    pic_name="I6.jpg"
    img = cv2.imread(
        fr".\..\string_art\test_images\{pic_name}"
    )
    # img_weights = None
    img_weights = cv2.imread(
        r"C:\Users\CapDaniels\Meine Ablage\Documents\CodingProjects\pythonProjects\string_art\test_images\I6_mask.jpg"
    )

    thickness= 0.1
    solver = SolverGUI(circle, img, img_weights=img_weights, line_thickness=thickness, dpmm=10.0, n_points=1000, kink_factor=0.1, weights_importance=0.5, opacitys=0.8, opacityd=1.)
    solver.start_gui()
    # for i in range(20):
    #     solver.solve_next()
    gsketch = GSketch("test_main", nozzle_diameter=0.4, filament_diameter=1.75)
    gcode_stringer = GcodeStringer(solver, gsketch, z_hop=1, z_base=5, feed_rate=800., thickness=thickness, ramp_angle=10)
    gcode_stringer.process_Gcode()
    # print(gsketch.get_GCode())
    gsketch.save_GCode(f"./test/{pic_name}.gcode")
    print("saved:", f"./test/{pic_name}.gcode")
    cv2.imwrite(f"./test/{pic_name}", solver.image)
    print("saved:", f"./test/{pic_name}")

    # test headleass
    # kf = 0.3
    # for kf in np.arange(0, 0.35, 0.05):
    #     print("KF=", kf)
    #     solver = Solver(circle, img, img_weights=img_weights, line_thickness=0.2, dpmm=5.0, n_points=500)
    #     solver.kink_factor = kf
    #     for i in range(1100):
    #         solver.solve_next()
    #     gsketch = GSketch("test_main", nozzle_diameter=0.4, filament_diameter=1.75)
    #     gcode_stringer = GcodeStringer(solver, gsketch)
    #     gcode_stringer.process_Gcode()
    #     # print(gsketch.get_GCode())
    #     # gsketch.save_GCode("./test/test1.gcode")
    #     cv2.imwrite(f"./test/{pic_name}_headless_kf{kf:.3f}.png", solver.image)