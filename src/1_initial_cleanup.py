"""
Import data as numpy array
"""

#used bioiamge-fazallab environment 

import os
import numpy as np
from loguru import logger
from bioio import BioImage
from bioio.writers import OmeTiffWriter
import bioio_ome_tiff
import bioio_nd2

logger.info('import ok')

# configuration
input_path = r'P:/Sophie/Analysis for Sophie/A3-B3 Coilin-546 SUMO1-488/'
output_folder = r'C:/Users/u244278/OneDrive - Baylor College of Medicine/Documents/python_projects/ANA_011626_LL-nucleolus-shape/results/initial_cleanup/'
image_extensions = ['.czi', '.tif', '.tiff', '.lif', '.nd2']


def image_converter(image_path, output_folder, tiff=False, MIP=False, array=True):
    """Stack images from nested .czi files and save for subsequent processing

    Args:
        image_path (str): filepath for the image to be converted
        output_folder (str): filepath for saving the converted images
        tiff (bool, optional): Save tiff. Defaults to False.
        MIP (bool, optional): Save np array as maximum projected image along third to last axis. Defaults to False.
        array (bool, optional): Save np array. Defaults to True.
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
    
    # --------------- initalize file_list ---------------
    if input_path == r'P:/Sophie/Analysis for Sophie/A3-B3 Coilin-546 SUMO1-488/':
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
    do_not_quantitate = [] 
    image_names = [filename for filename in flat_file_list if not any(word in filename for word in do_not_quantitate)]

    # remove duplicates
    image_names = list(dict.fromkeys(image_names))

    # --------------- collect image names and convert ---------------
    # collect and convert images to np arrays
    for name in image_names:
        name
        image_converter(name, output_folder=f'{output_folder}', tiff=False, MIP=True, array=False)

    logger.info('initial cleanup complete :-)')
