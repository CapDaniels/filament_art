from gcode.gcode import *
import numpy as np

# TODO: add retract, try with thicker strings, add wipe at end

gsketch = GSketch("main")

x1 = 30 + 2
x2 = 180 - 2

y1 = 110 - 14
y2 = 110 + 14

thickness_start = 0.6
thickness_end = 0.3

n_strings = 32
z_hop =  1
string_height = 3.05
travel_height = string_height + z_hop + 0.5
speed = 1800

G1(z=travel_height)
G1(x1, y1)
# set Z_base
G1(z=string_height)
G1(e=-5, f=2500)  # retract
G1(f=speed)

for y_base, thickness in zip(np.linspace(y1, y2, n_strings), np.linspace(thickness_start, thickness_end, n_strings)):
    G1(x=x1, y=y_base)
    G1(z=string_height)
    G1(e=5, f=2500)  # engage
    G1(f=speed)
    GString(x=x2, thickness=thickness, z_hop=z_hop)
    G1(e=-5, f=2500)  # retract
    G1(f=speed)
    G1(z=travel_height)

print(gsketch.get_GCode())

gsketch.save_GCode("./out.gcode")