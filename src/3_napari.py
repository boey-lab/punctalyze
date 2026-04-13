"""
Quality control: use napari to validate Cellpose-generated masks.
"""

import os
import numpy as np
from skimage.segmentation import clear_border
from skimage.io import imread
from loguru import logger
import matplotlib.pyplot as plt
import napari
from qtpy.QtWidgets import QApplication

logger.info('import ok')

# configuration
image_folder = 'results/initial_cleanup/'
mask_folder = 'results/cellpose_masking/'
output_folder = 'results/napari_masking/'
mask_filename = 'cellpose_cellmasks.npy'
SATURATION_THRESHOLD = 2**16 - 1  # assuming 16-bit images
SATURATION_FRAC_CUTOFF = 0.05 # saturation tolerance, fraction of pixels in a cell that can be saturated before cell is discarded
NUCLEUS_AREA_THRESHOLD = 800 # minimum area for a nucleus to be considered valid, in pixels
BORDER_BUFFER_SIZE = 10 # number of pixels from the edge of the image to consider as 'border' for removal of border-touching objects
COI = [0, 1] # channel of interest for saturation check 
FLUORO_INTENSITY_THRESHOLD = 500  # threshold for significant fluorescence intensity in COI
FLUORO_FRACTION_CUTOFF = 0.1  # expression tolerance, fraction of pixels in a cell that must be above the fluorescence threshold to be kept


# Setup
def ensure_output_folder(path):
    os.makedirs(path, exist_ok=True)


# IO
def load_images(image_folder):
    return {
        fname.replace('.npy', ''): np.load(os.path.join(image_folder, fname))
        for fname in os.listdir(image_folder) if fname.endswith('.npy')
    }


def load_masks(mask_folder):
    return {
        fname.replace('_sammask.npy', ''): np.load(os.path.join(mask_folder, fname))
        for fname in os.listdir(mask_folder) if fname.endswith('_sammask.npy')
    }


def save_mask(image_name, mask_stack):
    out_path = os.path.join(output_folder, f'{image_name}_mask.npy')
    np.save(out_path, mask_stack)
    logger.info(f'Mask saved: {out_path}')


# Mask Filtering
def remove_saturated_cells(image_stack, mask_stack, COI=COI):
    '''Remove masks for saturated cells based on intensity threshold.'''
    if isinstance(COI, list):
        COI = COI[0]  # for now, just check the first channel of interest

    raw = image_stack[COI, :, :]
    cells = mask_stack[0, :, :]

    valid_labels = []
    for label in np.unique(cells)[1:]:
        pixel_mask = (cells == label)
        pixel_count = np.count_nonzero(pixel_mask)
        saturated = np.count_nonzero(raw[pixel_mask] > SATURATION_THRESHOLD)
        if saturated / pixel_count < SATURATION_FRAC_CUTOFF:
            valid_labels.append(label)

    filtered_cells = np.where(np.isin(cells, valid_labels), cells, 0)
    return filtered_cells


def filter_cells_by_fluoro_expression(image_stack, cells_mask, COI=COI):
    """Keep only cells with significant fluoro signal in all specified channels."""
    
    # if COI is iterable
    if isinstance(COI, int):
        COI = [COI]

    valid_labels = []

    for label in np.unique(cells_mask)[1:]:
        mask = (cells_mask == label)
        pixel_count = np.count_nonzero(mask)

        keep_cell = True  # assume valid unless a channel fails

        for ch in COI:
            fluoro = image_stack[ch, :, :]
            fluoro_pixels = fluoro[mask]
            bright_pixels = np.count_nonzero(
                fluoro_pixels > FLUORO_INTENSITY_THRESHOLD
            )

            if bright_pixels / pixel_count <= FLUORO_FRACTION_CUTOFF:
                keep_cell = False
                break  # fail fast

        if keep_cell:
            valid_labels.append(label)

    filtered_cells = np.where(np.isin(cells_mask, valid_labels), cells_mask, 0)
    return filtered_cells


def remove_border_objects(mask):
    return clear_border(mask, buffer_size=BORDER_BUFFER_SIZE)


def filter_small_nuclei(nuclei_mask):
    new_mask = nuclei_mask.copy()
    for label in np.unique(nuclei_mask)[1:]:
        area = np.count_nonzero(nuclei_mask == label)
        if area < NUCLEUS_AREA_THRESHOLD:
            new_mask[new_mask == label] = 0
    return new_mask


def filter_masks_auto(image_stack, mask_stack, filter_fluoro=False):
    cells, nuclei = mask_stack[0], mask_stack[1]

    cells_filtered = remove_saturated_cells(image_stack, mask_stack)
    cells_filtered = remove_border_objects(cells_filtered)

    if filter_fluoro:
        cells_filtered = filter_cells_by_fluoro_expression(image_stack, cells_filtered)

    intra_nuclei = np.where(cells_filtered > 0, nuclei, 0)
    filtered_nuclei = filter_small_nuclei(nuclei) # note, do not want to remove all nuclei for now

    return np.stack([cells_filtered, filtered_nuclei])


# Manual QC
def validate_with_napari(image_stack, image_name, mask_stack):
    """Launch napari, allow user to edit masks, then save upon exit."""
    app = QApplication.instance()
    if not app:
        app = QApplication([])

    viewer = napari.Viewer()
    viewer.add_image(image_stack, name='image_stack')
    viewer.add_labels(mask_stack[0], name='cells')
    viewer.add_labels(mask_stack[1], name='nuclei')

    # Show and block until window closed
    viewer.window._qt_window.show()
    app.exec_()

    # After closing the window, get edited data
    cells = viewer.layers['cells'].data
    nuclei = viewer.layers['nuclei'].data

    out_stack = np.stack([cells, nuclei])
    save_mask(image_name, out_stack)
    return out_stack


# Main QC Pipeline
def run_qc_pipeline(filter_fluoro=True):
    ensure_output_folder(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]

    # find masks that have already been filtered and saved, to skip these
    already_filtered = {
        fname.replace('_mask.npy', '')
        for fname in os.listdir(output_folder)
        if fname.endswith('_mask.npy')
    }

    for fname in image_files:
        name = fname.replace('.npy', '')

        if name in already_filtered:
            continue

        #  load only what you need
        image = np.load(os.path.join(image_folder, fname))
        mask = np.load(os.path.join(mask_folder, f"{name}_sammask.npy"))

        #  shape check
        if image.shape[1:] != mask.shape[1:]:
            logger.warning(f"Shape mismatch for {name}, skipping")
            continue

        #  process immediately
        filtered_mask = filter_masks_auto(image, mask, filter_fluoro=filter_fluoro)

        #  manual QC
        validate_with_napari(image, name, filtered_mask)

        #  free memory explicitly
        del image, mask, filtered_mask


# Entry Point
if __name__ == '__main__':
    run_qc_pipeline(filter_fluoro=True)
