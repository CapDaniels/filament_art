from pathlib import Path
from tkinter import filedialog, messagebox
from PIL import Image


def nicer_dict_print(d):
    for k, v in d.items():
        print(k, "=", v)


def load_image(filename=None, display_warnbox=True, bw=False, size=None):
    """Loads an image from file, crops it and optioanlly resizes it.

    Args:
        filename (str, optional): The path to the image file. If None, a file
            dialog will be opened.
        display_warnbox (bool, optional): Whether to display warning messages in
            case of errors. Defaults to True.
        bw (bool, optional): Whether to convert the image to grayscale (black
            and white). Defaults to False.
        size (int or tuple, optional): The target size for resizing the image.
            If None, no resizing will be performed. Defaults to None.

    Returns:
        tuple: A tuple containing the filename, the loaded PIL image object and
        the orignale shape, or (None, None, (None, None)) if an error occurs.
    """
    if filename is None:
        filename = filedialog.askopenfilename(
            title="Select an input picture",
            initialdir="./test_images",
            filetypes=(
                ("Picture files", ".jpeg .jpg .png .tiff"),
                ("All files", "*.*"),
            ),
        )

    if filename:
        filename = Path(filename)
        if not (filename.exists() and filename.is_file()):
            if display_warnbox:
                messagebox.showerror("The file does not exist!")
            return None, None, (None, None)

        try:
            image = Image.open(filename)
        except IOError:
            if display_warnbox:
                messagebox.showerror(
                    "Something went wrong while loading the image!"
                )
            return None, None, (None, None)

        if bw:
            image = image.convert("L")

        width, height = image.size
        crop_left, crop_right = 0, width
        crop_top, crop_bottom = 0, height

        # Crop the image to a square based on the smallest dimension
        if width > height:
            crop_left = (width - height) // 2
            crop_right = int(width + height) // 2
        else:
            crop_top = (height - width) // 2
            crop_bottom = (height + width) // 2

        cropped_image = image.crop(
            (crop_left, crop_top, crop_right, crop_bottom)
        )

        if size is not None:
            cropped_image = cropped_image.resize((size, size))

        return filename, cropped_image, (width, height)


    return None, None, (None, None)
