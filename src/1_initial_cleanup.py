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
input_path = 'M:/Olivia/2026-02-28'
output_folder = 'results/initial_cleanup/'
image_extensions = ['.czi', '.tif', '.tiff', '.lif', '.nd2']


def squarify(image_stack):
    new_channels = []
    for idx, image in enumerate(image_stack):
        # make sure images are square
        rows, cols = image.shape
        max_dim = max(rows, cols)
        # calculate padding needed for each side
        pad_r = max_dim - rows
        pad_c = max_dim - cols
        # pad at the end (bottom/right)
        padded_image = np.pad(image, ((0, pad_r), (0, pad_c)), 
                                mode='constant', constant_values=0)
        new_channels.append(padded_image)
    image_stack = np.array(new_channels)
    return image_stack


# function for multi-scene data
def scene_splitter(image_path, names_mapped):
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
        # find matching name
        name = names_mapped[id]
        # save image as numpy array
        np.save(f'{output_folder}{name}.npy', restack)


# function for multi-scene data
def scene_finder(image_path, names_mapped):
    """Find scenes in multi scene acquisitions
    """

    czi = CziFile(image_path)
    data, dims = czi.read_image(return_dims=True)

    # squeeze unused dims
    data = np.squeeze(data)  # shape → (4, 16, 2048, 2048)

    # stitching
    bboxes = czi.get_all_mosaic_tile_bounding_boxes()

    # Keep only tiles that exist in pixel data
    tile_positions = []
    for tile_info, rect in bboxes.items():
        if tile_info.m_index < data.shape[1]:   # ensure valid tile index
            tile_positions.append((tile_info.m_index, rect))

    # Sort by M index
    tile_positions.sort(key=lambda x: x[0])
    xs = []
    ys = []
    for m_index, rect in tile_positions:
        xs.append(rect.x)
        ys.append(rect.y)
    min_x, min_y = min(xs), min(ys)
    xs = [x - min_x for x in xs]
    ys = [y - min_y for y in ys]
    tile_h, tile_w = data.shape[2], data.shape[3]
    n_channels = data.shape[0]
    canvas_w = max(xs) + tile_w
    canvas_h = max(ys) + tile_h
    stitched = np.zeros((n_channels, canvas_h, canvas_w), dtype=data.dtype)

    for (m_index, rect), x, y in zip(tile_positions, xs, ys):
        stitched[:, y:y+tile_h, x:x+tile_w] = data[:, m_index, :, :]

    # make scene x and y dimensions the same (sometimes off by one pixel)
    stitched = squarify(stitched)

    # find matching name
    bio_image = BioImage(image_path)
    well_name = bio_image.current_scene
    well_id = names_mapped[well_name]

    # if well_id is in 'do_not_quantitate' list, skip saving
    if any(word in well_id for word in do_not_quantitate):
        logger.info(f'Skipping {well_id} due to do_not_quantitate criteria')
        return

    # save image as numpy array
    np.save(f'{output_folder}{well_id}.npy', stitched)


def image_converter(image_path, output_folder, tiff=False, MIP=False, array=True, split_scenes=False, find_scenes=False, name_dict=None):
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
        scene_splitter(image_path, name_dict)
        return
    
    if find_scenes == True:
        scene_finder(image_path, name_dict)
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
    short_name = os.path.basename(image_path)
    short_name = short_name.split('.')[0]  # remove file extension

    if tiff == True:
        # save image as tiff file
        OmeTiffWriter.save(image, f'{output_folder}{short_name}.tif')

    if array == True:
        # save image as numpy array
        np.save(f'{output_folder}{short_name}.npy', image)

    if MIP == True:
        # save image as maximum intensity projection (MIP) numpy array 
        mip_image = np.max(image, axis=-3) # assuming axis for projection is third from last
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
        'C2-C2': 'jetprime_eGFP-co-mCherry_LIP5',
        'C3-C3': 'jetprime_FREE1-co-mCherry_LIP5',
        'C4-C4': 'jetprime_FLOE2-co-mCherry_LIP5',
        'C5-C5': 'jetprime_FLOE3-co-mCherry_LIP5',
        'C6-C6': 'jetprime_SKD1-co-mCherry_LIP5',
        'C7-C7': 'jetprime_ISTL1-co-mCherry_LIP5',
        'C8-C8': 'jetprime_mCherry_LIP5',
        'C9-C9': 'jetprime_noDNA',
        'E2-E2': 'lipo_eGFP-co-mCherry_LIP5',
        'E3-E3': 'lipo_FREE1-co-mCherry_LIP5',
        'E4-E4': 'lipo_FLOE2-co-mCherry_LIP5',
        'E5-E5': 'lipo_FLOE3-co-mCherry_LIP5',
        'E6-E6': 'lipo_SKD1-co-mCherry_LIP5',
        'E7-E7': 'lipo_ISTL1-co-mCherry_LIP5',
        'E8-E8': 'lipo_mCherry_LIP5',
        'E9-E9': 'lipo_noDNA',
        'F2-F2': 'lipo_eGFP-co-mCherry_LIP5',
        'F3-F3': 'lipo_FREE1-co-mCherry_LIP5',
        'F4-F4': 'lipo_FLOE2-co-mCherry_LIP5',
        'F5-F5': 'lipo_FLOE3-co-mCherry_LIP5',
        'F6-F6': 'lipo_SKD1-co-mCherry_LIP5',
        'F7-F7': 'lipo_ISTL1-co-mCherry_LIP5',
        'F8-F8': 'lipo_mCherry_LIP5',
        'F9-F9': 'lipo_noDNA'
    }

    # --------------- initalize file_list ---------------
    if input_path == 'raw_data/':
        flat_file_list = [input_path + filename for filename in os.listdir(input_path) if any(sub in filename for sub in image_extensions)]

    else:
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
        image_converter(name, output_folder=f'{output_folder}', find_scenes=True, name_dict=name_dict)

    logger.info('initial cleanup complete :-)')
