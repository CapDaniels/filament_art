import contours
import tkinter as tk
import cv2
import numpy as np
from gcode_builder import GcodeStringer
from gcode import GSketch, Mx
from gui import Tk_Settings_GUI, Solver_GUI
from utils import load_image

""" the main file is still very messy. This is partly due to missing GUI
    features, which are hardcoded here for now!
"""

def create_contour(settings):
    """Creates a contour object from the given settings.
    Only temporary untill contour selection is working in the UI"""
    # manually doing this here for now...
    contour_settings = {
        "center": (
            settings["print_vol_x"] / 2,
            settings["print_vol_y"] / 2,
        ),
        "radius": settings["frame_size"] / 2,
        "width": settings["frame_width"],
        "height": settings["frame_height"],
    }
    return contours.Circle(**contour_settings)

def main():
    # main settings window
    tk_root = tk.Tk()
    tk_settings = Tk_Settings_GUI(tk_root)
    settings = tk_settings.run()

    if settings is None:
        exit()

    # this is temporary. In later versions, you can define the contour in the
    # settings window.
    contour = create_contour(settings)
    _, img, _ = load_image(settings["input_image_path"])
    img = np.array(img)
    if img.ndim > 2:
        if img.shape[-1] == 2:
            img = img[:, :, 0]  # convert to cv2 format
        else:
            img = img[:, :, 2::-1]  # convert to cv2 format
    mask = None
    if settings["mask_image_path"] is not None:
        mask = np.array(load_image(settings["mask_image_path"])[1])
        if mask.ndim > 2:
            mask = mask[:, :, 2::-1]  # convert to cv2 format
    solver = Solver_GUI(
        name=settings["project_name"],
        contour=contour,
        img=img,
        img_weights=mask,
        dpmm=settings["internal_dpmm"],
        line_thickness=settings["string_thickness"],
        n_points=settings["n_points"],
    )
    # returns after GUI is closed
    if not solver.start_gui():
        exit()
    contour.save_model(solver.save_path / "frame.stl")

    # creating the stringing GCode
    gsketch = GSketch(
        settings["project_name"],
        nozzle_diameter=settings["nozzle_diameter"],
        filament_diameter=settings["filament_diameter"],
    )
    # hardcoding fan speed and temperature for now.
    Mx(106, S=255)  # max part cooling fan
    Mx(104, S=200)  # 200 deg. C. nozzle temp
    gcode_stringer = GcodeStringer(
        solver,
        gsketch,
        z_hop=1.2,
        z_base=5,
        feed_rate=1300.0,
        thickness=settings["string_thickness"],
        ramp_angle=15,
    )
    gcode_stringer.process_Gcode()
    gsketch.save_GCode(solver.save_path / "strings.gcode")


if __name__ == "__main__":
    main()
