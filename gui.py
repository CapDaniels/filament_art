import os
from pathlib import Path
from PIL import Image, ImageTk
from pathlib import Path
import re
from utils import load_image, nicer_dict_print
import tkinter as tk
from typing import Callable
from tkinter import filedialog, messagebox
from solver import Solver, img_type
from contours import Contour
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets


class Tk_Settings_GUI:
    default_input_path = "./test_images/benchy.png"
    default_mask_path = ""
    no_img_text = "No file loaded."
    default_project_name = "benchy"
    default_img_display_size = 300
    default_print_vol_x = 220.0
    default_print_vol_y = 220.0
    default_n_points = 600
    default_internal_dpmm = 10.0
    _settings = {}

    def __init__(self, root):
        self.normal_quit = False
        self.root = root
        self.root.title("Filament Art Settings")
        self._initialize_ui()

    def _initialize_ui(self):
        """Initialize the user interface."""
        if "nt" == os.name:
            self.root.wm_iconbitmap(Path("./benchy.ico").absolute().as_posix())
        else:
            icon_image = Image.open(Path("./benchy.ico").absolute().as_posix())
            icon_photo = ImageTk.PhotoImage(icon_image)
            self.root.wm_iconphoto(True, icon_photo)

        # Load default images (returns None, if deleted by user or not found)
        self.input_image_path, self.input_image, self.input_image_shape = (
            load_image(
                self.default_input_path,
                display_wrnbox=False,
                size=self.default_img_display_size,
            )
        )
        self.mask_image_path, self.mask_image, self.mask_image_shape = (
            load_image(
                self.default_mask_path,
                display_wrnbox=False,
                size=self.default_img_display_size,
            )
        )

        row_tracker = 0
        self.frame = tk.Frame(self.root)

        # UI Elements
        self.title_label = tk.Label(
            self.frame, text="Filament Art Settings", font=("Arial", 25)
        )
        self.title_label.grid(row=row_tracker, column=0, columnspan=2, pady=15)
        row_tracker += 1

        # Project Name Entry
        self.var_proj_name = self._add_labeled_entry(
            "Project Name:", self.default_project_name, row_tracker
        )
        row_tracker += 1

        # Nozzle Diameter
        self.var_nzzdia, _ = self._add_spinbox(
            "Nozzle diameter (mm):", 0.4, 0.05, 1.0, 0.05, row_tracker
        )
        row_tracker += 1

        # Filament Diameter
        self.var_flmntdia = self._add_radiobuttons(
            "Filament diameter (mm):",
            ["1.75 mm", "3 mm"],
            [1.75, 3.0],
            1.75,
            row_tracker,
        )
        row_tracker += 1

        # Print Volume X and Y
        self.var_print_vol_x, _ = self._add_spinbox(
            "Print volume in x (mm):",
            220,
            50,
            400,
            10,
            row_tracker,
            self._update_max_frame_size,
        )
        row_tracker += 1

        self.var_print_vol_y, _ = self._add_spinbox(
            "Print volume in y (mm):",
            220,
            50,
            400,
            10,
            row_tracker,
            self._update_max_frame_size,
        )
        row_tracker += 1

        # Frame Size, Width, and Height
        self.var_frame_size, self.frame_size_spinbox = self._add_spinbox(
            "Max. printed frame size (mm):",
            200,
            50,
            min(self.default_print_vol_x, self.default_print_vol_y),
            10,
            row_tracker,
        )
        row_tracker += 1

        self.var_frame_width, _ = self._add_spinbox(
            "Frame width (mm):", 5, 1, 20, 1, row_tracker
        )
        row_tracker += 1

        self.var_frame_height, _ = self._add_spinbox(
            "Frame height (mm):", 5, 1, 20, 1, row_tracker
        )
        row_tracker += 1

        # Connection Points and Resolution
        self.var_n_points, _ = self._add_spinbox(
            "Number of connection points:", 600, 10, 2000, 10, row_tracker
        )
        row_tracker += 1

        self.var_internal_dpmm, _ = self._add_spinbox(
            "Set internal resolution (dots per mm):",
            10.0,
            1.0,
            20.0,
            1.0,
            row_tracker,
        )
        row_tracker += 1

        # Browse Input Image
        self._add_button(
            "Browse input picture", self._load_input_image, row_tracker
        )
        self.input_photo_label = self._add_image_label(
            self.input_image, row_tracker
        )
        row_tracker += 1

        # Browse Mask Image
        self._add_button(
            "Browse mask picture (optional)",
            self._load_mask_image,
            row_tracker,
        )
        self._add_button(
            "Unload mask picture", self._unload_mask_image, row_tracker + 1
        )
        self.mask_photo_label = self._add_image_label(
            self.mask_image, row_tracker, rowspan=2
        )
        row_tracker += 2

        # String Thickness
        self.var_string_thickness, _ = self._add_spinbox(
            "Filament string thickness (mm):", 0.3, 0.1, 0.6, 0.05, row_tracker
        )
        row_tracker += 1

        # Start Button
        self._add_button(
            "All done, start the solver!",
            self.get_settings,
            row_tracker,
            columnspan=2,
        )

        self.frame.grid()

    def _add_labeled_entry(self, label, default, row):
        """Adds a labeled entry widget."""
        tk.Label(self.frame, text=label).grid(row=row, column=0)
        var = tk.StringVar(value=default)
        entry = tk.Entry(self.frame, textvariable=var)
        entry.grid(row=row, column=1, padx=5, pady=5)
        return var

    def _add_spinbox(
        self, label, default, min_val, max_val, increment, row, command=None
    ):
        """Adds a labeled spinbox."""
        tk.Label(self.frame, text=label).grid(row=row, column=0)
        var = (
            tk.DoubleVar(value=default)
            if isinstance(default, float)
            else tk.IntVar(value=default)
        )
        spinbox = tk.Spinbox(self.frame, from_=min_val, to=max_val,
                             increment=increment, textvariable=var,
                             command=command)  # type: ignore
        spinbox.grid(row=row, column=1, padx=5, pady=5)
        return var, spinbox

    def _add_radiobuttons(self, label, options, values, default, row):
        """Adds a group of radio buttons."""
        tk.Label(self.frame, text=label).grid(row=row, column=0)
        frame = tk.Frame(self.frame)
        var = tk.DoubleVar(value=default)
        for option, value in zip(options, values):
            tk.Radiobutton(frame, text=option, variable=var, value=value).pack(
                anchor="w"
            )
        frame.grid(row=row, column=1, padx=5, pady=5)
        return var

    def _add_button(self, label, command, row, column=0, columnspan=1):
        """Adds a button."""
        tk.Button(self.frame, text=label, command=command).grid(
            row=row, column=column, columnspan=columnspan, padx=5, pady=5
        )

    def _add_image_label(self, image, row, column=1, rowspan=1):
        """Adds an image label."""
        if image:
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(
                self.frame,
                image=photo,
                width=self.default_img_display_size,
                height=self.default_img_display_size,
            )
            label.image = photo  # type: ignore # Keep a reference
        else:
            label = tk.Label(
                self.frame, text=self.no_img_text, width=0, height=0
            )
        label.grid(row=row, column=column, padx=5, pady=5, rowspan=rowspan)
        return label

    def _refresh_image_label(self, label, image):
        """Refresh the image label."""
        if image:
            photo = ImageTk.PhotoImage(image)
            label.config(image=photo)
            label.image = photo
        else:
            label.config(text=self.no_img_text, image="", width=0, height=0)
            label.image = None

    def _load_input_image(self):
        """Load an input image."""
        self.input_image_path, self.input_image, self.input_image_shape = (
            load_image(size=self.default_img_display_size)
        )
        self._refresh_image_label(self.input_photo_label, self.input_image)

    def _load_mask_image(self):
        """Load a mask image."""
        self.mask_image_path, self.mask_image, self.mask_image_shape = (
            load_image(size=self.default_img_display_size)
        )
        self._refresh_image_label(self.mask_photo_label, self.mask_image)

    def _unload_mask_image(self):
        self.mask_image = None
        self.mask_image_path = None
        self._refresh_image_label(self.mask_photo_label, None)

    def _update_max_frame_size(self, event=None):
        max_size = min(self.var_print_vol_x.get(), self.var_print_vol_y.get())
        if self.var_frame_size.get() > max_size:
            self.var_frame_size.set(max_size)  # type: ignore
        self.frame_size_spinbox.config(to=max_size)

    def validate_settings(self):
        if self.input_image is None:
            messagebox.showinfo(
                "Info",
                "You cannot start the solver without loading an input image!",
            )
            return False

        if self.var_proj_name.get() == "":
            messagebox.showinfo(
                "Info",
                "You cannot start the solver without a project name!",
            )
            return False
        # Regex for a valid filename (adapted for most OS)
        regex = r"^[a-zA-Z0-9._\-\s]+$"
        if not re.match(regex, self.var_proj_name.get()):
            messagebox.showinfo(
                "Info",
                "Invalid project name!",
            )
            return False

        if self.mask_image is not None:
            if self.input_image_shape != self.mask_image_shape:
                messagebox.showinfo(
                    "Info",
                    "Mask image must have the same size as the input image!",
                )
                return False

        # some sanity checks:
        nzzldia = float(self.var_nzzdia.get())
        if nzzldia < 0.2:
            if not messagebox.askyesno(
                "Continue?",
                f"Your selected nozzlediameter of {nzzldia:.2f} mm is quite small.\nDo you want to continue?",
            ):
                return False

        return True

    def get_settings(self, destroy_window=True):
        if not self.validate_settings():
            return None

        self._settings = {
            "project_name": self.var_proj_name.get(),
            "nozzle_diameter": float(self.var_nzzdia.get()),
            "filament_diameter": float(self.var_flmntdia.get()),
            "print_vol_x": self.var_print_vol_x.get(),
            "print_vol_y": self.var_print_vol_y.get(),
            "frame_size": self.var_frame_size.get(),
            "frame_width": self.var_frame_width.get(),
            "frame_height": self.var_frame_height.get(),
            "n_points": self.var_n_points.get(),
            "internal_dpmm": float(self.var_internal_dpmm.get()),
            "input_image_path": Path(self.input_image_path)  # type: ignore
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
        nicer_dict_print(self._settings)
        print()
        self.quit()

    @property
    def settings(self):
        return self._settings

    def quit(self):
        """Quit the application by starting the solver."""
        self.normal_quit = True
        self.root.destroy()

    def close_window_event(self):
        """Quit the application by closing the window."""
        self.normal_quit = False
        self.root.destroy()

    def run(self):
        """Run the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.close_window_event)
        self.root.mainloop()
        if not self.normal_quit:
            return None
        return self.settings


class Solver_GUI(Solver):
    fast_draw_mode = True

    def __init__(
        self,
        name: str,
        contour: "Contour",
        img: img_type,
        img_weights: img_type | None = None,
        weights_importance: float = 1.0,
        mode="greedy_dark",
        dpmm=12.0,
        line_thickness=0.4,
        opacityd=1.0,
        opacitys=1.0,
        n_points=200,
        kink_factor=0.25,
    ):
        super().__init__(
            name,
            contour=contour,
            img=img,
            img_weights=img_weights,
            weights_importance=weights_importance,
            mode=mode,
            dpmm=dpmm,
            line_thickness=line_thickness,
            opacityd=opacityd,
            opacitys=opacitys,
            n_points=n_points,
            overlap_penalty=kink_factor,
        )
        # 1 = black
        # Create figure and subplots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.mpl_og_img = self.axs[0, 0].imshow(
            self.img, cmap="gray", vmin=0, vmax=1
        )
        self.mpl_img = self.axs[0, 1].imshow(
            self._img_solver_target, cmap="gray", vmin=0, vmax=1
        )
        self.mpl_img_canvas = self.axs[1, 0].imshow(
            self._output_canvas, cmap="gray", vmin=0, vmax=1
        )
        titles = ["picture", "solver target", "drawn strings"]
        if self._img_solver_weights is not None:
            self.axs[1, 1].imshow(
                self._img_solver_target * self._img_solver_weights,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            titles.append("mask")
            # self.axs[1, 1].imshow(
            #     self._img_weights, cmap="gray", vmin=0, vmax=1
            # )
        else:
            self.axs[1, 1].axis("off")
        for i, (ax, title) in enumerate(zip(self.axs.flatten(), titles)):
            ax.set_title(title)

        # Adjust layout to make room for the button
        self.fig.subplots_adjust(top=0.88)
        self.fig.subplots_adjust(bottom=0.2)

        # Adding status text
        self.ax_text = plt.axes((0.1, 0.95, 0.8, 0.05))
        self.ax_text.set_axis_off()
        self.status_string = "{n_finished:04d} strings drawn. {n_queue:04d} strings in the queue. Last score: {score:.4f}."
        self.ax_text_text = self.ax_text.text(
            0.5,
            -0.3,
            "Click on any number to add the respective number of strings to your artwork!\nClearing the queue interrupts the drawing process.",
            verticalalignment="bottom",
            horizontalalignment="center",
            transform=self.ax_text.transAxes,
            fontsize=14,
        )

        # Create a button below the subplots
        button_labels = ["1", "50", "300", "600"]
        self._buttons = []
        for i, label in enumerate(button_labels):
            ax_button = plt.axes(
                (0.1 + i * 0.2, 0.1, 0.15, 0.05)
            )  # [left, bottom, width, height]
            button = widgets.Button(ax_button, label)
            button.on_clicked(self._add_frames_to_queue)
            button.ax.set_label(label)
            self._buttons.append(button)
        button_labels = ["clear queue", "save and quit"]
        button_callbacks = [self._clear_queue, self.save_and_quit_with_dialog]
        for i, (label, callback) in enumerate(
            zip(button_labels, button_callbacks)
        ):
            ax_button = plt.axes(
                (0.1 + i * 0.2, 0.02, 0.15, 0.05)
            )  # [left, bottom, width, height]
            button = widgets.Button(ax_button, label)
            button.on_clicked(callback)
            button.ax.set_label(label)
            self._buttons.append(button)
        self._framequeue = 0

    def start_gui(self):
        # plt.ion()
        self.fig.canvas.toolbar.pack_forget()
        plt.show(block=False)
        self._running = True
        return self._mainloop()

    def _mainloop(self):
        """runs mainloop and returns a status variable"""
        # I know that fig.number exists at runtime, the linter does not.
        # Thus type: ignore
        while self._running and plt.fignum_exists(self.fig.number):  # type: ignore
            if self._framequeue > 0:
                self.solve_next()
                if not self.fast_draw_mode:
                    self._update_gui()
                elif (
                    (self.string_count < 10)
                    or (self._framequeue < 20)
                    or (self._framequeue % 10) == 0
                ):
                    self._update_gui()
                self._framequeue -= 1
            self.fig.canvas.start_event_loop(
                0.001
            )  # this makes the mpl window responsive
        return (self.string_count > 1) and (self.save_path is not None)

    def _add_frames_to_queue(self, event):
        if self._framequeue == 0:
            self._update_gui(
                "Solving the first string... This may take a while!"
            )
        n_frames = int(event.inaxes.get_label())
        self._framequeue += n_frames

    def _clear_queue(self, event):
        self._framequeue = 0
        self._update_gui()

    def _update_gui(self, status_string=None):
        status_string = status_string or self.status_string.format(
            n_finished=self.string_count,
            n_queue=self._framequeue,
            score=self.curr_score,
        )
        self.mpl_img_canvas.set_data(self._output_canvas)
        self.mpl_img.set_data(self._img_solver_target)
        self.ax_text_text.set_text(status_string)
        # self.fig.canvas.draw()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
