from solver import SolverGUI
import contours
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path


# class AnnSpinbox(tk.Frame):
#     "An annotated Spinbox"

#     def __init__(self, root, label_kwargs, spin_kwargs):
#         super().__init__(root)
#         self.ann_label = tk.Label(self, **label_kwargs)
#         self.ann_label.pack(side="left", padx=5, pady=5)
#         self.ann_spinbox = tk.Spinbox(self, **spin_kwargs)
#         self.ann_spinbox.pack(side="right", padx=5, pady=5)


def dict_print(d):
    for k, v in d.items():
        print(k, "=", v)


class Tk_Settings:
    default_input_path = "./test_images/Utah_teapot.png"
    default_mask_path = "./test_images/Utah_teapot_mask.png"
    no_img_text = "No file loaded."
    default_img_display_size = 300
    default_print_vol_x = 220
    default_print_vol_y = 220
    _settings = None

    def __init__(self, root):
        # loading default image
        self.input_image_path, self.input_image = self._load_image(
            self.default_input_path, display_wrnbox=False
        )
        # laoding a possible default mask image
        self.mask_image_path, self.mask_image = self._load_image(
            self.default_mask_path, display_wrnbox=False
        )

        # Defining UI elements
        self.root = root
        self.frame = tk.Frame(root)
        self.title_label = tk.Label(
            self.frame, text="Filament Art Settings", font=("Arial", 25)
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=15)

        self.var_nzzdia = tk.Variable(name="var_nzzdia")
        self.var_nzzdia.set(0.4)
        tk.Label(self.frame, text="Nozzle diameter (mm):").grid(
            row=1, column=0
        )
        nzzdia_spinbox = tk.Spinbox(
            self.frame,
            from_=0.05,
            to=1.0,
            increment=0.05,
            textvariable=self.var_nzzdia,
        )
        nzzdia_spinbox.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.frame, text="Filement diameter (mm):").grid(
            row=2, column=0
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
        flmntdia_frame.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self.frame, text="Print volume in x (mm):").grid(
            row=3, column=0
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
        print_vol_x_spinbox.grid(row=3, column=1, padx=5, pady=5)
        print_vol_x_spinbox.bind("<FocusOut>", self._update_max_frame_size)

        tk.Label(self.frame, text="Print volume in y (mm):").grid(
            row=4, column=0
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
        print_vol_y_spinbox.grid(row=4, column=1, padx=5, pady=5)
        print_vol_y_spinbox.bind("<FocusOut>", self._update_max_frame_size)

        tk.Label(self.frame, text="Max. printed frame size (mm):").grid(
            row=5, column=0
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
        self.frame_size_spinbox.grid(row=5, column=1, padx=5, pady=5)

        tk.Button(
            self.frame,
            text="Browse input picture",
            command=self._load_input_image,
        ).grid(row=6, column=0, padx=5, pady=5)
        tk.Button(
            self.frame,
            text="Browse mask picture (optional)",
            command=self._load_mask_image,
        ).grid(row=7, column=0, padx=5, pady=5)
        tk.Button(
            self.frame,
            text="Unload mask picture",
            command=self._unload_mask_image,
        ).grid(row=8, column=0, padx=5, pady=5)

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
        self.input_photo_label.grid(row=6, column=1, padx=5, pady=5)

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
        self.mask_photo_label.grid(row=7, column=1, padx=5, pady=5, rowspan=2)

        tk.Button(
            self.frame,
            text="All done, start the solver!",
            command=self.start_solver,
        ).grid(row=9, column=0, columnspan=2, padx=5, pady=5)

        self.frame.grid()

    def _load_input_image(self):
        input_image_path, input_image = self._load_image()
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
        mask_image_path, mask_image = self._load_image(bw=True)
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

    def _load_image(self, filename=None, display_wrnbox=True, bw=False):
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
            try:
                img = Image.open(filename)
                if bw:
                    img = img.convert("L")
                width, height = img.size

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

                cropped_img = img.crop(
                    (crop_left, crop_top, crop_right, crop_bottom)
                )
                resized_img = cropped_img.resize(
                    (
                        self.default_img_display_size,
                        self.default_img_display_size,
                    )
                )

                return filename, resized_img

            except IOError:
                if display_wrnbox:
                    messagebox.showerror(
                        "Something went wrong while loading the image!"
                    )
                return None, None

        return None, None

    def _update_max_frame_size(self, event=None):
        max_size = min(self.var_print_vol_x.get(), self.var_print_vol_y.get())
        if self.var_frame_size.get() > max_size:
            self.var_frame_size.set(max_size)
        self.frame_size_spinbox.config(to=max_size)

    def start_solver(self):
        if self.input_image is None:
            messagebox.showinfo(
                "Info",
                "You cannot start the solver without loading an input image!",
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
            "nozzle_diameter": nzzldia,
            "filament_diameter": float(self.var_flmntdia.get()),
            "print_vol_x": self.var_print_vol_x.get(),
            "print_vol_y": self.var_print_vol_y.get(),
            "frame_size": self.var_frame_size.get(),
            "input_image_path": Path(self.input_image_path)
            .absolute()
            .as_posix(),
            "mask_image_path": (
                Path(self.mask_image_path).absolute().as_posix()
                if self.mask_image_path is not None
                else None
            ),
        }
        print("Settings: ")
        dict_print(self._settings)
        print()
        self.root.quit()

    @property
    def settings(self):
        return self._settings


def main():
    # main settings window
    tk_root = tk.Tk()
    tk_root.title("Filament Art Settings")
    tk_settings = Tk_Settings(tk_root)

    tk_root.mainloop()

    print(tk_settings.settings)


if __name__ == "__main__":
    main()
