from solver import SolverGUI
import contours
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import re
from solver import SolverGUI
import contours
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import re
import contours
import cv2
import numpy as np
from gcode_builder import GcodeStringer
from gcode import GSketch, Mx

def dict_print(d):
    for k, v in d.items():
        print(k, "=", v)


def load_image(filename=None, display_wrnbox=True, bw=False, size=None):
    """Loads an image from file, crops it and optioanlly resizes it.

    Args:
        filename (str, optional): The path to the image file. If None, a file dialog will be opened.
        display_wrnbox (bool, optional): Whether to display warning messages in case of errors. Defaults to True.
        bw (bool, optional): Whether to convert the image to grayscale (black and white). Defaults to False.
        size (int or tuple, optional): The target size for resizing the image. If None, no resizing will be performed. Defaults to None.

    Returns:
        tuple: A tuple containing the filename and the loaded image object, or None, None if an error occurs.
    """
    if filename is None:
        filename = filedialog.askopenfilename(
            title="Select an input picture",
            initialdir="./test_images",
            filetypes=(
                ("Picture files", ".jepg .jpg .png .tiff"),
                ("all files", "*.*"),
            ),
        )

    if filename:
        filename = Path(filename)
        if not (filename.exists() and filename.is_file()):
            if display_wrnbox:
                messagebox.showerror(
                    "The files does not exist!",
                )
            return None, None
        try:
            image = Image.open(filename)
            if bw:
                image = image.convert("L")
            width, height = image.size

            # Determine which border to crop
            crop_top = 0
            crop_bottom = height
            crop_left = 0
            crop_right = width
            if width > height:
                crop_left = int(width / 2 - height / 2)
                crop_right = int(width / 2 + height / 2)
            else:
                crop_top = int(height / 2 - width / 2)
                crop_bottom = int(height / 2 + width / 2)

            cropped_image = image.crop(
                (crop_left, crop_top, crop_right, crop_bottom)
            )
            if size is not None:
                cropped_image = cropped_image.resize(
                    (
                        size,
                        size,
                    )
                )

            return filename, cropped_image

        except IOError:
            if display_wrnbox:
                messagebox.showerror(
                    "Something went wrong while loading the image!"
                )
            return None, None

    return None, None


class Tk_Settings:
    default_input_path = "./test_images/willie.png"
    default_mask_path = "./test_images/willie_mask.png"
    no_img_text = "No file loaded."
    default_project_name = "willie"
    default_img_display_size = 300
    default_print_vol_x = 220
    default_print_vol_y = 220
    _settings = {}

    def __init__(self, root):
        row_tracker = 0
        # loading default image
        self.input_image_path, self.input_image = load_image(
            self.default_input_path, display_wrnbox=False, size=self.default_img_display_size
        )
        # laoding a possible default mask image
        self.mask_image_path, self.mask_image = load_image(
            self.default_mask_path, display_wrnbox=False, size=self.default_img_display_size
        )
        # Defining UI elements
        self.root = root
        self.frame = tk.Frame(root)
        self.title_label = tk.Label(
            self.frame, text="Filament Art Settings", font=("Arial", 25)
        )
        self.title_label.grid(row=row_tracker, column=0, columnspan=2, pady=15)
        row_tracker += 1

        # Adding Project Name Entry
        tk.Label(self.frame, text="Project Name:").grid(
            row=row_tracker, column=0
        )
        self.project_name_var = tk.StringVar(self.frame)
        self.project_name_var.set(self.default_project_name)
        self.project_name_entry = tk.Entry(
            self.frame, textvariable=self.project_name_var
        )
        self.project_name_entry.grid(row=row_tracker, column=1, padx=5, pady=5)
        row_tracker += 1

        self.var_nzzdia = tk.Variable(name="var_nzzdia")
        self.var_nzzdia.set(0.4)
        tk.Label(self.frame, text="Nozzle diameter (mm):").grid(
            row=row_tracker, column=0
        )
        nzzdia_spinbox = tk.Spinbox(
            self.frame,
            from_=0.05,
            to=1.0,
            increment=0.05,
            textvariable=self.var_nzzdia,
        )
        nzzdia_spinbox.grid(row=row_tracker, column=1, padx=5, pady=5)
        row_tracker += 1

        tk.Label(self.frame, text="Filement diameter (mm):").grid(
            row=row_tracker, column=0
        )

        self.var_flmntdia = tk.Variable(self.frame, name="var_flmntdia")
        self.var_flmntdia.set(1.75)
        flmntdia_frame = tk.Frame(self.frame)
        tk.Radiobutton(
            flmntdia_frame,
            text="1.75 mm",
            variable=self.var_flmntdia,
            value=1.75,
        ).pack(anchor="w")
        tk.Radiobutton(
            flmntdia_frame,
            text="3 mm",
            variable=self.var_flmntdia,
            value=3.0,
        ).pack(anchor="w")
        flmntdia_frame.grid(row=row_tracker, column=1, padx=5, pady=5)
        row_tracker += 1

        tk.Label(self.frame, text="Print volume in x (mm):").grid(
            row=row_tracker, column=0
        )
        self.var_print_vol_x = tk.IntVar(self.frame, name="var_print_vol_x")
        self.var_print_vol_x.set(self.default_print_vol_x)
        print_vol_x_spinbox = tk.Spinbox(
            self.frame,
            from_=50,
            to=400,
            increment=10,
            textvariable=self.var_print_vol_x,
            command=self._update_max_frame_size,
        )
        print_vol_x_spinbox.grid(row=row_tracker, column=1, padx=5, pady=5)
        print_vol_x_spinbox.bind("<FocusOut>", self._update_max_frame_size)
        row_tracker += 1

        tk.Label(self.frame, text="Print volume in y (mm):").grid(
            row=row_tracker, column=0
        )
        self.var_print_vol_y = tk.IntVar(self.frame, name="var_print_vol_y")
        self.var_print_vol_y.set(self.default_print_vol_y)
        print_vol_y_spinbox = tk.Spinbox(
            self.frame,
            from_=50,
            to=400,
            increment=10,
            textvariable=self.var_print_vol_y,
            command=self._update_max_frame_size,
        )
        print_vol_y_spinbox.grid(row=row_tracker, column=1, padx=5, pady=5)
        print_vol_y_spinbox.bind("<FocusOut>", self._update_max_frame_size)
        row_tracker += 1

        tk.Label(self.frame, text="Max. printed frame size (mm):").grid(
            row=row_tracker, column=0
        )
        self.var_frame_size = tk.IntVar(name="var_frame_size")
        self.var_frame_size.set(200)
        self.frame_size_spinbox = tk.Spinbox(
            self.frame,
            from_=50,
            to=min(self.default_print_vol_x, self.default_print_vol_y),
            increment=10,
            textvariable=self.var_frame_size,
        )
        self.frame_size_spinbox.grid(row=row_tracker, column=1, padx=5, pady=5)
        row_tracker += 1

        tk.Label(self.frame, text="Frame width (mm):").grid(
            row=row_tracker, column=0
        )
        self.var_frame_width = tk.IntVar(name="var_frame_width")
        self.var_frame_width.set(5)
        self.frame_width_spinbox = tk.Spinbox(
            self.frame,
            from_=1,
            to=20,
            increment=1,
            textvariable=self.var_frame_width,
        )
        self.frame_width_spinbox.grid(
            row=row_tracker, column=1, padx=5, pady=5
        )
        row_tracker += 1

        tk.Label(self.frame, text="Frame height (mm):").grid(
            row=row_tracker, column=0
        )
        self.var_frame_height = tk.IntVar(name="var_frame_height")
        self.var_frame_height.set(5)
        self.frame_height_spinbox = tk.Spinbox(
            self.frame,
            from_=1,
            to=20,
            increment=1,
            textvariable=self.var_frame_height,
        )
        self.frame_height_spinbox.grid(
            row=row_tracker, column=1, padx=5, pady=5
        )
        row_tracker += 1

        tk.Label(self.frame, text="Number of connection points:").grid(
            row=row_tracker, column=0
        )
        self.var_n_points = tk.IntVar(name="var_n_points")
        self.var_n_points.set(200)
        self.n_points_spinbox = tk.Spinbox(
            self.frame,
            from_=10,
            to=2000,
            increment=10,
            textvariable=self.var_n_points,
        )
        self.n_points_spinbox.grid(row=row_tracker, column=1, padx=5, pady=5)
        row_tracker += 1

        tk.Label(
            self.frame, text="Set internal resolution (dots per mm):"
        ).grid(row=row_tracker, column=0)
        self.var_internal_dpmm = tk.Variable(name="var_internal_dpmm")
        self.var_internal_dpmm.set(5.0)
        self.internal_dpmm_spinbox = tk.Spinbox(
            self.frame,
            from_=1.0,
            to=20.0,
            increment=1.0,
            textvariable=self.var_internal_dpmm,
        )
        self.internal_dpmm_spinbox.grid(
            row=row_tracker, column=1, padx=5, pady=5
        )
        row_tracker += 1

        tk.Button(
            self.frame,
            text="Browse input picture",
            command=self._load_input_image,
        ).grid(row=row_tracker, column=0, padx=5, pady=5)

        if self.input_image is not None:
            input_photo = ImageTk.PhotoImage(self.input_image)
            self.input_photo_label = tk.Label(
                self.frame,
                image=input_photo,
                width=self.default_img_display_size,
                height=self.default_img_display_size,
            )
            self.input_photo_label.image = input_photo  # keeping a reference
        else:
            self.input_photo_label = tk.Label(
                self.frame, text="No file loaded."
            )
        self.input_photo_label.grid(row=row_tracker, column=1, padx=5, pady=5)
        row_tracker += 1

        tk.Button(
            self.frame,
            text="Browse mask picture (optional)",
            command=self._load_mask_image,
        ).grid(row=row_tracker, column=0, padx=5, pady=5)
        tk.Button(
            self.frame,
            text="Unload mask picture",
            command=self._unload_mask_image,
        ).grid(row=row_tracker + 1, column=0, padx=5, pady=5)

        if self.mask_image is not None:
            mask_photo = ImageTk.PhotoImage(self.mask_image)
            self.mask_photo_label = tk.Label(
                self.frame,
                image=mask_photo,
                width=self.default_img_display_size,
                height=self.default_img_display_size,
            )
            self.mask_photo_label.image = mask_photo  # keeping a reference
        else:
            self.mask_photo_label = tk.Label(self.frame, text=self.no_img_text)
        self.mask_photo_label.grid(
            row=row_tracker, column=1, padx=5, pady=5, rowspan=2
        )
        row_tracker += 2

        tk.Label(self.frame, text="Filament string thickness (mm):").grid(
            row=row_tracker, column=0
        )
        self.var_string_thickness = tk.Variable(
            self.frame, name="var_string_thickness"
        )
        self.var_string_thickness.set(0.3)
        self.string_thickness_spinbox = tk.Spinbox(
            self.frame,
            from_=0.1,
            to=0.6,
            increment=0.05,
            textvariable=self.var_string_thickness,
        )
        self.string_thickness_spinbox.grid(
            row=row_tracker, column=1, padx=5, pady=5
        )
        row_tracker += 1

        tk.Button(
            self.frame,
            text="All done, start the solver!",
            command=self.get_settings,
        ).grid(row=row_tracker, column=0, columnspan=2, padx=5, pady=5)

        self.frame.grid()

    def _load_input_image(self):
        input_image_path, input_image = load_image(size=self.default_img_display_size)
        if input_image_path is None:
            return
        self.input_image_path, self.input_image = input_image_path, input_image
        input_photo = ImageTk.PhotoImage(self.input_image)
        self.input_photo_label.configure(
            text="",
            image=input_photo,
            width=self.default_img_display_size,
            height=self.default_img_display_size,
        )
        self.input_photo_label.image = input_photo  # keeping a reference

    def _load_mask_image(self):
        mask_image_path, mask_image = load_image(bw=True, size=self.default_img_display_size)
        if mask_image_path is None:
            return
        self.mask_image_path, self.mask_image = mask_image_path, mask_image
        mask_photo = ImageTk.PhotoImage(self.mask_image)
        self.mask_photo_label.configure(
            text="",
            image=mask_photo,
            width=self.default_img_display_size,
            height=self.default_img_display_size,
        )
        self.mask_photo_label.image = mask_photo  # keeping a reference

    def _unload_mask_image(self):
        self.mask_photo_label.configure(
            text=self.no_img_text, image="", width=0, height=0
        )
        self.mask_image = None
        self.mask_image_path = None

    def _update_max_frame_size(self, event=None):
        max_size = min(self.var_print_vol_x.get(), self.var_print_vol_y.get())
        if self.var_frame_size.get() > max_size:
            self.var_frame_size.set(max_size)
        self.frame_size_spinbox.config(to=max_size)

    def get_settings(self, destroy_window=True):
        if self.input_image is None:
            messagebox.showinfo(
                "Info",
                "You cannot start the solver without loading an input image!",
            )
            return

        if self.project_name_var.get() == "":
            messagebox.showinfo(
                "Info",
                "You cannot start the solver without a project name!",
            )
            return
        # Regex for a valid filename (adapted for most OS)
        regex = r"^[a-zA-Z0-9._\-\s]+$"
        if not re.match(regex, self.project_name_var.get()):
            messagebox.showinfo(
                "Info",
                "Invalid project name!",
            )
            return

        # some sanity checks:
        nzzldia = float(self.var_nzzdia.get())
        if nzzldia < 0.2:
            if not messagebox.askyesno(
                "Continue?",
                f"Your selected nozzlediameter of {nzzldia:.2f} mm is quite small.\nDo you want to continue?",
            ):
                return
        nzzldia = nzzldia

        self._settings = {
            "project_name": self.project_name_var.get(),
            "nozzle_diameter": nzzldia,
            "filament_diameter": float(self.var_flmntdia.get()),
            "print_vol_x": self.var_print_vol_x.get(),
            "print_vol_y": self.var_print_vol_y.get(),
            "frame_size": self.var_frame_size.get(),
            "frame_width": self.var_frame_width.get(),
            "frame_height": self.var_frame_height.get(),
            "n_points": self.var_n_points.get(),
            "internal_dpmm": float(self.var_internal_dpmm.get()),
            "input_image_path": Path(self.input_image_path)
            .absolute()
            .as_posix(),
            "mask_image_path": (
                Path(self.mask_image_path).absolute().as_posix()
                if self.mask_image_path is not None
                else None
            ),
            "string_thickness": float(self.var_string_thickness.get()),
        }
        print("Settings: ")
        dict_print(self._settings)
        print()
        if destroy_window:
            self.root.destroy()

    @property
    def settings(self):
        return self._settings


def create_contour(settings):
    """Creates a contour object from the given settings."""
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
    tk_root.title("Filament Art Settings")
    tk_settings = Tk_Settings(tk_root)

    tk_root.mainloop()

    settings = tk_settings.settings
    tk_root.quit()
    print(settings)

    contour = create_contour(settings)
    _, img = load_image(settings["input_image_path"])
    img = np.array(img)[:, :, 2::-1]  # convert to cv2 format
    _img = cv2.imread(settings["input_image_path"])
    mask = (
        np.array(load_image(settings["mask_image_path"])[1])[:, :, 2::-1]
        if settings["mask_image_path"] is not None
        else None
    )
    solver = SolverGUI(
        name=settings["project_name"],
        contour=contour,
        img=img,
        img_weights=mask,
        dpmm=settings["internal_dpmm"],
        line_thickness=settings["string_thickness"],
        n_points=settings["n_points"],
    )
    if not solver.start_gui():
        exit()
    # returns after GUI is closed
    contour.save_model(solver.save_path / "frame.stl")

    # creating the stringing GCode
    gsketch = GSketch(settings["project_name"], nozzle_diameter=settings["nozzle_diameter"], filament_diameter=settings["filament_diameter"])
    # hardcoding fan speed and temperature for now.
    Mx(106, S=255)  # max part cooling fan
    Mx(104, S=200)  # 200 deg. C. nozzle temp
    gcode_stringer = GcodeStringer(solver, gsketch, z_hop=1.2, z_base=5, feed_rate=1300., thickness=settings["string_thickness"], ramp_angle=15)
    gcode_stringer.process_Gcode()
    # print(gsketch.get_GCode())
    gsketch.save_GCode(solver.save_path / "strings.gcode")


if __name__ == "__main__":
    main()
