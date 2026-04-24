"""
Import data as numpy array
"""

import os
import numpy as np
from loguru import logger
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from aicspylibczi import CziFile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import bioio_ome_tiff
import bioio_nd2

logger.info('import ok')

# configuration
input_path = 'M:/Olivia'
output_folder = 'results/initial_cleanup/'
image_extensions = ['.czi', '.tif', '.tiff', '.lif', '.nd2']


def squarify(image_stack):
    """
    Pads Y/X to square.
    Supports:
    (C, Y, X) or (C, Z, Y, X)
    """
    if image_stack.ndim not in (3, 4):
        raise ValueError(f"Unsupported shape: {image_stack.shape}")

    # Always operate on last two dims (Y, X)
    y, x = image_stack.shape[-2:]
    max_dim = max(y, x)

    pad_y = max_dim - y
    pad_x = max_dim - x

    pad_width = [(0, 0)] * image_stack.ndim
    pad_width[-2] = (0, pad_y)
    pad_width[-1] = (0, pad_x)

    return np.pad(
        image_stack,
        pad_width,
        mode='constant',
        constant_values=0
    )


def max_project(image):
    """
    Applies max projection if Z exists. Helper for scene_finder and scene_splitter functions.
    Assumes (C, Z, Y, X) or (Z, Y, X)
    """
    if image.ndim == 4:
        return np.max(image, axis=1)  # (C, Z, Y, X) → (C, Y, X)
    elif image.ndim == 3:
        return np.max(image, axis=0)  # (Z, Y, X) → (Y, X)
    else:
        return image

# function for multi-scene data
def scene_splitter(image_path, names_mapped, MIP=False):
    """Split scenes in multi scene acquisitions
    """

    # get a bioimage object
    bio_image = BioImage(image_path)

    # get and save scenes
    for id in bio_image.scenes:
        id
        bio_image.set_scene(id)
        restack = np.stack(bio_image.data[0,:,0,:,:])
        # make scene x and y dimensions the same (sometimes off by one pixel)
        restack = squarify(restack)
        if MIP:
            restack = max_project(restack)
        np.save(f'{output_folder}{name}.npy', restack)        # find matching name
        name = names_mapped[id]
        # save image as numpy array
        np.save(f'{output_folder}{name}.npy', restack)


def normalize_dims(data, dims):
    """
    Normalize to (C, Z, M, Y, X)
    Works with CziFile dims format: [('C', size), ...]
    """

    # Extract just dimension names
    dim_names = [d[0] for d in dims]

    # Build map
    dim_map = {dim: i for i, dim in enumerate(dim_names)}

    # Full axis order (keep everything, just reorder)
    full_order = list(range(len(dim_names)))

    # Move target dims to the front in correct order
    target = ['C', 'Z', 'M', 'Y', 'X']
    front_axes = [dim_map[d] for d in target if d in dim_map]

    # Keep remaining dims (H, S, T, etc.)
    remaining_axes = [i for i in full_order if i not in front_axes]

    new_order = front_axes + remaining_axes

    # Move desired dims to front
    data = np.transpose(data, axes=new_order)

    # Collapse all extra dims (H, S, T, etc.)
    data = data.reshape(
        data.shape[0],  # C
        data.shape[1],  # Z
        data.shape[2],  # M
        data.shape[3],  # Y
        data.shape[4],  # X
    )

    # Remove trailing singleton dims
    data = np.squeeze(data, axis=tuple(range(5, data.ndim)))

    return data


# function for multi-scene data
def scene_finder(image_path, names_mapped, experiment_prefix=True, MIP=False):
    """Find and stitch scenes in multi-scene acquisitions (dimension-safe)"""

    czi = CziFile(image_path)
    data, dims = czi.read_image(return_dims=True)

    # Normalize to (C, Z, M, Y, X)
    data = normalize_dims(data, dims)

    # Unpack dimensions explicitly
    n_channels, n_z, n_tiles, tile_h, tile_w = data.shape

    # Get tile bounding boxes
    bboxes = czi.get_all_mosaic_tile_bounding_boxes()

    # Filter valid tiles
    tile_positions = []
    for tile_info, rect in bboxes.items():
        if tile_info.m_index < n_tiles:
            tile_positions.append((tile_info.m_index, rect))

    # Sort tiles
    tile_positions.sort(key=lambda x: x[0])

    # Normalize coordinates
    xs = [rect.x for _, rect in tile_positions]
    ys = [rect.y for _, rect in tile_positions]

    min_x, min_y = min(xs), min(ys)
    xs = [x - min_x for x in xs]
    ys = [y - min_y for y in ys]

    # Canvas size
    canvas_w = max(xs) + tile_w
    canvas_h = max(ys) + tile_h

    # Initialize stitched array (C, Z, Y, X)
    stitched = np.zeros((n_channels, n_z, canvas_h, canvas_w), dtype=data.dtype)

    # Stitch tiles
    for (m_index, rect), x, y in zip(tile_positions, xs, ys):
        stitched[:, :, y:y+tile_h, x:x+tile_w] = data[:, :, m_index, :, :]

    # Squarify
    stitched = squarify(stitched)

    # Optional MIP
    if MIP:
        stitched = max_project(stitched)

    # Map scene name
    bio_image = BioImage(image_path)
    well_name = bio_image.current_scene
    well_id = names_mapped[well_name]

    if experiment_prefix:
        prefix = image_path.split('\\')[-1].split('/')[0]  # e.g., '2026-03-31'
        well_id = f'{prefix}_{well_id}'

    if any(word in well_id for word in do_not_quantitate):
        logger.info(f'Skipping {well_id} due to do_not_quantitate criteria')
        return

    np.save(f'{output_folder}{well_id}.npy', stitched)


def image_converter(image_path, output_folder, tiff=False, MIP=False, array=True, split_scenes=False, find_scenes=False, name_dict=None, experiment_prefix=False):
    """Stack images from nested .czi files and save for subsequent processing

    Args:
        image_path (str): filepath for the image to be converted
        output_folder (str): filepath for saving the converted images
        tiff (bool, optional): Save tiff. Defaults to False.
        MIP (bool, optional): Save np array as maximum projected image along third to last axis. Defaults to False.
        array (bool, optional): Save np array. Defaults to True.
        split_scenes (bool, optional): Split scenes. Defaults to False.
        find_scenes (bool, optional): Find scenes. Defaults to False.
        names_mapped (dict, optional): Dictionary mapping scene names to desired output names. Required if split_scenes or find_scenes is True. Defaults to None.
        experiment_prefix (bool, optional): Whether to include experiment prefix in output name. Defaults to False.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # check if image exists
    full_path = None
    if os.path.exists(image_path):
        full_path = image_path

    if full_path is None:
        logger.warning(f'File not found for {image_path}')
        return
    
    if split_scenes == True:
        scene_splitter(image_path, name_dict, experiment_prefix=experiment_prefix, MIP=MIP)
        return
    
    if find_scenes == True:
        scene_finder(image_path, name_dict, experiment_prefix=experiment_prefix, MIP=MIP)
        return

    # get a bioimage object
    bio_image = BioImage(full_path)
    image_shape = bio_image.dims

    # import single channel timeseries
    if (image_shape['T'][0] > 1) & (image_shape['C'][0] == 1):
        image = bio_image.get_image_data("TYX", C=0, Z=0)

    # import multichannel timeseries
    if (image_shape['T'][0] > 1) & (image_shape['C'][0] > 1):
        image = bio_image.get_image_data("CTYX", B=0, Z=0, V=0)

    # import multichannel z-stack
    if image_shape['Z'][0] > 1:
        image = bio_image.get_image_data("CZYX", B=0, V=0, T=0)

    # import multichannel single z-slice single timepoint
    if (image_shape['Z'][0] == 1) & (image_shape['T'][0] == 1) & (image_shape['C'][0] > 1):
        image = bio_image.get_image_data("CYX", B=0, Z=0, V=0, T=0)

    # make more human readable name
    short_name = image_path.split('\\')[1].replace('/', '_') # keep experiment name
    short_name = short_name.split('.')[0]  # remove file extension

    if tiff == True:
        # save image as tiff file
        OmeTiffWriter.save(image, f'{output_folder}{short_name}.tif')

    if array == True:
        # save image as numpy array
        np.save(f'{output_folder}{short_name}.npy', image)

    if MIP == True:
        # save image as maximum intensity projection (MIP) numpy array 
        mip_image = max_project(image)
        np.save(f'{output_folder}{short_name}_mip.npy', mip_image)


if __name__ == '__main__':
    
    # --------------- dictionary of sample names ---------------
    name_dict = {
        'B2-B2': 'jetprime_eGFP-co-mCherry_LIP5',
        'B3-B3': 'jetprime_FREE1-co-mCherry_LIP5',
        'B4-B4': 'jetprime_FLOE2-co-mCherry_LIP5',
        'B5-B5': 'jetprime_FLOE3-co-mCherry_LIP5',
        'B6-B6': 'jetprime_SKD1-co-mCherry_LIP5',
        'B7-B7': 'jetprime_ISTL1-co-mCherry_LIP5',
        'B8-B8': 'jetprime_mCherry_LIP5',
        'B9-B9': 'jetprime_noDNA',
    }

    # --------------- initalize file_list ---------------
    if input_path == 'raw_data/': # if data is loaded locally into 'raw_data'
        flat_file_list = [input_path + filename for filename in os.listdir(input_path) if any(sub in filename for sub in image_extensions)]

    else: # if data needs to be pulled from other directories
        # find subdirectories of interest
        experiments = ['240509-Processed']
        # if you want all images from all subdirectories in file path, set experiments to 'walk_list'
        walk_list = [x[0] for x in os.walk(input_path)]
        walk_list = [item for item in walk_list if any(x in item for x in experiments)]

        # read in all image file names
        file_list = [[f'{root}/{filename}' for filename in files]
                    for folder_path in walk_list
                    for root, dirs, files in os.walk(folder_path)]

        # flatten file_list
        flat_file_list = [item for sublist in file_list for item in sublist if any(sub in item for sub in image_extensions)]

    # remove images that do not require analysis (e.g., qualitative controls)
    do_not_quantitate = ['_no-', 'noDNA', 'UT'] 
    image_names = [filename for filename in flat_file_list if not any(word in filename for word in do_not_quantitate)]

    # remove duplicates
    image_names = list(dict.fromkeys(image_names))
    image_names = [name for name in image_names if '(' in name] # keep only images with parentheses in name, which indicates they are from multi-scene acquisitions and need to be split

    # --------------- collect image names and convert ---------------
    # collect and convert images to np arrays
    for name in image_names:
        image_converter(name, output_folder=f'{output_folder}', find_scenes=False, name_dict=None, MIP=False, experiment_prefix=False)

    logger.info('initial cleanup complete :-)')
