import os
from pathlib import Path
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider




def crop_center(image, crop_size=100):
    """
    Crop a 2D image to its central square region of size crop_size x crop_size.
    """
    h, w = image.shape
    # Ensure crop_size doesn't exceed image dimensions
    crop_size_h = min(crop_size, h)
    crop_size_w = min(crop_size, w)
    
    start_x = (w - crop_size_w) // 2
    start_y = (h - crop_size_h) // 2
    
    return image[start_y:start_y+crop_size_h, start_x:start_x+crop_size_w]


# ---------------------------------------------------------
#  Adaptive windowing — using your histogram tail logic
# ---------------------------------------------------------
def compute_window(image):
    """
    Compute window bounds: clip extremes (1st-99th percentiles), then condense contrast to central 30% of remaining values.
    Returns vmin, vmax as 35th and 65th percentiles of clipped data.
    """
    pixels = image.flatten().astype(np.float32)
    
    # Clip extremes: remove outside 1st-99th percentiles
    vmin_clip = np.percentile(pixels, 60)
    vmax_clip = np.percentile(pixels, 100)
    clipped_pixels = np.clip(pixels, vmin_clip, vmax_clip)
    
    # Condense contrast to central 30% of the clipped pixel values
    vmin = np.percentile(clipped_pixels, 50)
    vmax = np.percentile(clipped_pixels, 100)
    
    return float(vmin), float(vmax)


def rescale_contrast_adaptive(image):
    """
    Apply adaptive windowing (via compute_window) and rescale to [0,1].
    """
    vmin, vmax = compute_window(image)
    img = image.astype(np.float32)

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)

    return img


# ---------------------------------------------------------
#  Load DICOMs
# ---------------------------------------------------------
def load_dicom_images(mainpath: str):
    slice_locations = []
    dicom_set_original = []
    filenames = []

    for root, _, file_list in os.walk(mainpath):
        for filename in file_list:
            if 'dir' in filename.lower() or 'txt' in filename.lower():
                continue

            dcm_path = Path(root, filename)

            try:
                ds = dicom.dcmread(dcm_path, force=True)
                ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
                slice_locations.append(ds.SliceLocation)
            except Exception as e:
                print(f"Couldn't load {dcm_path.name}: {e}")
            else:
                dicom_set_original.append(ds)
                filenames.append(filename)

    if not slice_locations:
        raise ValueError("No valid DICOM slices loaded.")

    slice_locations = np.array(slice_locations)
    sorted_idx = (-slice_locations).argsort()

    sorted_dicoms = [dicom_set_original[i] for i in sorted_idx]
    sorted_filenames = [filenames[i] for i in sorted_idx]
    images = [crop_center(ds.pixel_array.astype(np.float32), crop_size=200)
          for ds in sorted_dicoms]


    return images, sorted_filenames


# ---------------------------------------------------------
# Save slice-filename mapping
# ---------------------------------------------------------
def save_slice_filenames(mainpath, filenames):
    txt_path = os.path.join(mainpath, "slice_filenames.txt")
    with open(txt_path, "w") as f:
        for idx, name in enumerate(filenames):
            f.write(f"{idx} {name}\n")
    print(f"Saved slice-filename mapping to {txt_path}")


# ---------------------------------------------------------
# Slider update callback
# ---------------------------------------------------------
def update(index):
    idx = int(index)
    raw_image = images[idx]

    # Use your adaptive histogram-based windowing
    image = rescale_contrast_adaptive(raw_image)

    ax_img.clear()
    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax_img.set_title(f"Slice {idx + 1} / {len(images)}")
    ax_img.axis("off")

    fig.canvas.draw_idle()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
mainpath =r"C:\TOH Data\CATPHANQA\newdata"#r"C:\TOH Data\XVISampleData"#

images, dicom_files = load_dicom_images(mainpath)
save_slice_filenames(mainpath, dicom_files)

fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)

# Slider
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, "Slice", 0, len(images) - 1, valinit=0, valstep=1)
slider.on_changed(update)

update(0)
plt.show()
