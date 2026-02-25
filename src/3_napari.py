"""
Quality control: use napari to validate Cellpose-generated masks.
"""

import os
import numpy as np
from skimage.segmentation import clear_border
from skimage.io import imread
from loguru import logger
import napari
from qtpy.QtWidgets import QApplication

logger.info('import ok')

# configuration
image_folder = 'results/initial_cleanup/'
mask_folder = 'results/cellpose_masking/'
output_folder = 'results/napari_masking/'
mask_filename = 'cellpose_cellmasks.npy'
SATURATION_THRESHOLD = 60000
SATURATION_FRAC_CUTOFF = 0.05
NUCLEUS_AREA_THRESHOLD = 8000
BORDER_BUFFER_SIZE = 10
COI = 0 # channel of interest for saturation check (e.g., 1 for channel 2)
FLUORO_INTENSITY_THRESHOLD = 200  # threshold for significant fluorescence intensity in COI
FLUORO_FRACTION_CUTOFF = 0.1  # fraction of pixels in a cell that must be above the threshold to keep it


# Setup
def ensure_output_folder(path):
    os.makedirs(path, exist_ok=True)


# IO
def load_images(image_folder):
    return {
        fname.replace('.npy', ''): np.load(os.path.join(image_folder, fname))
        for fname in os.listdir(image_folder) if fname.endswith('.npy')
    }


def load_masks(mask_path, image_keys):
    all_masks = np.load(mask_path)
    return {
        image_name: all_masks[i, :, :]
        for i, image_name in enumerate(image_keys)
    }


def save_mask(image_name, mask_stack):
    out_path = os.path.join(output_folder, f'{image_name}_mask.npy')
    np.save(out_path, mask_stack)
    logger.info(f'Mask saved: {out_path}')


# Mask Filtering
def remove_saturated_cells(image_stack, mask_stack, COI=COI):
    '''Remove masks for saturated cells based on intensity threshold.'''
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


def filter_cells_by_fluoro_expression(image_stack, cells_mask):
    """Keep only cells with significant fluoro signal."""
    fluoro = image_stack[COI, :, :]
    valid_labels = []

    for label in np.unique(cells_mask)[1:]:
        mask = (cells_mask == label)
        pixel_count = np.count_nonzero(mask)

        fluoro_pixels = fluoro[mask]
        bright_pixels = np.count_nonzero(fluoro_pixels > FLUORO_INTENSITY_THRESHOLD)

        if bright_pixels / pixel_count > FLUORO_FRACTION_CUTOFF:
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
    filtered_nuclei = filter_small_nuclei(intra_nuclei)

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
def run_qc_pipeline(filter_fluoro=False):
    ensure_output_folder(output_folder)

    images = load_images(image_folder)
    masks = load_masks(os.path.join(mask_folder, mask_filename), images.keys())

    logger.info('starting automated mask filtering')
    filtered_masks = {
        name: filter_masks_auto(image, masks[name], filter_fluoro=filter_fluoro)
        for name, image in images.items()
    }

    logger.info('starting manual validation in napari')
    already_filtered = {
        fname.replace('_mask.npy', '')
        for fname in os.listdir(output_folder)
        if fname.endswith('_mask.npy')
    }

    for name, image in images.items():
        name
        if name not in already_filtered:
            _ = validate_with_napari(image, name, filtered_masks[name])


# Entry Point
if __name__ == '__main__':
    run_qc_pipeline(filter_fluoro=True)
