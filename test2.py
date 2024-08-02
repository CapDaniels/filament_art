import cv2
from gcode import *
import contours
from gcode_builder import GcodeStringer
from solver_discrete_no_hop import SolverGUI, Solver
import numpy as np
import itertools

# tesing different printing parameters
# irrelevent for prod, but I wanted this on git, so I can easily change between desktop and laptop


if __name__ == "__main__":

    x1, x2 = 15., 205.
    x = [x1, x2]
    z = 3.
    z_hop = 1.
    y = 110. - 120/2 + 2

    gsketch = GSketch("testing_parameters")

    thicknesses = np.linspace(0.2, 0.4, 9, endpoint=True)
    angles = [5, 15, 45]
    feedrates = [600, 100, 1400]

    G1(x2, y, z=z+10)
    G1(z=z, f=150)

    for idx, (th, a, f) in enumerate(itertools.product(thicknesses, angles, feedrates)):
        print(th, a, f)
        GString(th, x=x[idx%2], vf=f, hf=f, ramp_angle=a, z_hop=z_hop)
        y += 1.4
        G1(y=y)
    
    gsketch.save_GCode("./test/parameter_testing/thickness_angles_feedrates.gcode")
    