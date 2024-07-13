import cv2
from gcode import G1, GFollowContour, GSketch, GThinStretch
import contours
from gcode_builder import GcodeStringer
from solver_discrete_no_hop import SolverGUI, Solver
import numpy as np


if __name__ == "__main__":
    circle = contours.Circle((0, 0), 100)
    img = cv2.imread(
        r".\..\string_art\test_images\Snoopy_Peanuts.png"
    )
    img_weights = None
    # img_weights = cv2.imread(
    #     r".\..\string_art\test_images\11_mask.jpg"
    # )


    # solver = SolverGUI(circle, img, img_weights=img_weights, line_thickness=0.2, dpmm=5.0, n_points=500)
    # solver.start_gui()
    # gsketch = GSketch("test_main", nozzle_diameter=0.4, filament_diameter=1.75)
    # gcode_stringer = GcodeStringer(solver, gsketch)
    # gcode_stringer.process_Gcode()
    # # print(gsketch.get_GCode())
    # gsketch.save_GCode("./test/test1.gcode")
    # cv2.imwrite( "./test/test1.png", solver.image)



    # test headleass
    kf = 0.3
    for kf in np.arange(0, 1, 0.025):
        print("KF=", kf)
        solver = Solver(circle, img, img_weights=img_weights, line_thickness=0.2, dpmm=5.0, n_points=500)
        solver.kink_factor = kf
        for i in range(600):
            solver.solve_next()
        gsketch = GSketch("test_main", nozzle_diameter=0.4, filament_diameter=1.75)
        gcode_stringer = GcodeStringer(solver, gsketch)
        gcode_stringer.process_Gcode()
        # print(gsketch.get_GCode())
        # gsketch.save_GCode("./test/test1.gcode")
        cv2.imwrite(f"./test/test1_headless_kf{kf:.2f}.png", solver.image)