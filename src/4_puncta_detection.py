"""
Detect and analyze features of puncta per cell
"""

import os
import importlib.util
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import skimage.io
from skimage import measure, segmentation, morphology
from skimage.morphology import remove_small_objects
from scipy import stats
from scipy.stats import skewtest
from loguru import logger
import functools
# special import, path to script
napari_utils_path = 'punctalyze/src/3_napari.py' # adjust as needed

# load the module dynamically due to annoying file name
spec = importlib.util.spec_from_file_location("napari", napari_utils_path)
napari_utils = importlib.util.module_from_spec(spec)
sys.modules["napari_utils"] = napari_utils
spec.loader.exec_module(napari_utils)
remove_saturated_cells = napari_utils.remove_saturated_cells

logger.info('import ok')

# plotting setup
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

# --- configuration ---
STD_THRESHOLD = 3  # number of standard deviations above mean for puncta thresholding
SAT_FRAC_CUTOFF = 0.01  # for consistency with remove_saturated_cells
COI_1 = 1  # channel of interest for saturation check (e.g., 1 for channel 2)
COI_2 = 0  # secondary channel of interest for comparisons
COI_1_name = 'mCherry-LIP5'  # name of the first channel of interest, for plotting
COI_2_name  = 'eGFP'  # name of the second channel of interest, for plotting
MIN_PUNCTA_SIZE = 2**4  # minimum size of puncta
SCALE_PX = (294.67/2720) # size of one pixel in units specified by the next constant
SCALE_UNIT = 'um'  # units for the scale bar
image_folder = 'results/initial_cleanup/'
mask_folder = 'results/napari_masking/'
output_folder = 'results/summary_calculations/'
proofs_folder = 'results/proofs/'

for folder in [output_folder, proofs_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)


def feature_extractor(mask, properties=None):
    if properties is None:
        properties = [
            'area', 'eccentricity', 'label',
            'major_axis_length', 'minor_axis_length',
            'perimeter', 'coords'
        ]
    props = measure.regionprops_table(mask, properties=properties)
    return pd.DataFrame(props)


def load_images(image_folder):
    images = {}
    for fn in os.listdir(image_folder):
        if fn.endswith('.npy'):
            name = fn.removesuffix('.npy')
            images[name] = np.load(f'{image_folder}/{fn}')
    return images


def load_masks(mask_folder):
    masks = {}
    for fn in os.listdir(mask_folder):
        if fn.endswith('_mask.npy'):
            name = fn.removesuffix('_mask.npy')
            masks[name] = np.load(f'{mask_folder}/{fn}', allow_pickle=True)
    return masks


def generate_cytoplasm_masks(masks):
    # logger.info('removing nuclei from cell masks...')
    cyto_masks = {}
    for name, img in masks.items():
        cell_mask, nuc_mask = img[0], img[1]
        cell_bin = (cell_mask > 0).astype(int) # make binary masks
        nuc_bin = (nuc_mask > 0).astype(int)

        single_cyto = []
        labels = np.unique(cell_mask)
        if labels.size > 1:
            for lbl in labels[labels != 0]:
                cyto = np.where(cell_mask == lbl, cell_bin, 0)
                cyto_minus_nuc = cyto & ~nuc_bin
                if np.any(cyto_minus_nuc):
                    single_cyto.append(np.where(cyto_minus_nuc, lbl, 0))
                else:
                    single_cyto.append(np.zeros_like(cell_mask, dtype=int))
        else:
            single_cyto.append(np.zeros_like(cell_mask, dtype=int))

        cyto_masks[name] = sum(single_cyto)
    logger.info('cytoplasm masks created')
    return cyto_masks


def filter_saturated_images(images, masks):
    # logger.info('filtering saturated cells...')
    filtered = {}
    for name, img in images.items():
        # wrangle masks to match the expected input for the saturation check function
        if len(masks[name].shape) == 2:  # if only one mask layer, add a dummy one
            masks_arr = masks[name]
            masks_arr = np.expand_dims(masks_arr, axis=0) 
        else:
            masks_arr = masks[name]             

        # apply imported saturation check function
        cells = remove_saturated_cells(
            image_stack=img,
            mask_stack=masks_arr,
            COI=COI_1
        )

        # build stack for downstream processing as [coi2, coi1, cell_masks]
        filtered[name] = np.stack([img[COI_2], img[COI_1], cells])
    # logger.info('saturated cells filtered')
    return filtered


def collect_features(image_dict, STD_THRESHOLD=STD_THRESHOLD):
    # logger.info('collecting cell and puncta features...')
    results = []
    for name, img in image_dict.items():
        coi2, coi1, mask = img
        unique_cells = np.unique(mask)[1:]
        contours = measure.find_contours((mask > 0).astype(int), 0.8)
        contour = [c for c in contours if len(c) >= 100]

        # if no masks, skip this image
        if len(unique_cells) == 0:
            continue

        for lbl in unique_cells:
            cell_mask = mask == lbl
            coi1_vals = coi1[cell_mask]
            mean_coi1 = coi1_vals.mean()
            std_coi1 = coi1_vals.std()

            threshold = (std_coi1 * STD_THRESHOLD) + mean_coi1
            binary = (coi1 > threshold) & cell_mask
            puncta_labels = morphology.label(binary)
            puncta_labels = remove_small_objects(puncta_labels, min_size=MIN_PUNCTA_SIZE)

            df_p = feature_extractor(puncta_labels).add_prefix('puncta_')
            
            # define column names for the extra stats
            stats_columns = [
                'puncta_cv',
                'puncta_skew',
                'puncta_intensity_mean',
                'puncta_intensity_mean_in_coi2'
            ]

            if df_p.empty:
                # create a single-row df filled with 0s, same columns
                df_stats = pd.DataFrame([[np.nan] * len(stats_columns)], columns=stats_columns)
            else:
                stats_list = []
                for i, row in df_p.iterrows():
                    p_mask = puncta_labels == row['puncta_label']
                    puncta_vals = coi1[p_mask]
                    cv = puncta_vals.std() / puncta_vals.mean() if puncta_vals.mean() != 0 else np.nan
                    skew_stat = skewtest(puncta_vals).statistic if len(puncta_vals) >= 16 else np.nan
                    mean_p = puncta_vals.mean()
                    mean_coi2 = coi2[p_mask].mean()
                    stats_list.append((cv, skew_stat, mean_p, mean_coi2))

                df_stats = pd.DataFrame(stats_list, columns=stats_columns)
            
            df = pd.concat([df_p.reset_index(drop=True), df_stats], axis=1)

            # remove puncta pixels to get only the surrounding cell intensity (dilute) for the partition coefficient calculation
            surrounding_mask = cell_mask & ~binary
            df['cell_coi1_dilute_intensity_mean'] = (coi1[surrounding_mask]).mean()
            df['cell_coi2_dilute_intensity_mean'] = (coi2[surrounding_mask]).mean()

            # add other per-cell features
            df['image_name'], df['cell_number'] = name, lbl
            df['cell_size'] = cell_mask.sum()
            df['cell_std'] = std_coi1
            df['cell_cv'] = std_coi1 / mean_coi1  # coefficient of variation
            df['cell_skew'] = skewtest(coi1_vals).statistic
            df['cell_coi1_intensity_mean'] = mean_coi1
            df['cell_coi2_intensity_mean'] = (coi2[cell_mask]).mean()
            df['cell_coords'] = [contour] * len(df)

            results.append(df)
    
    if not results:
        return pd.DataFrame()   

    # logger.info('feature extraction done')
    return pd.concat(results, ignore_index=True)


def extra_puncta_features(df):
    df = df.copy()  # avoid modifying in place
    df['puncta_aspect_ratio'] = df['puncta_minor_axis_length'] / df['puncta_major_axis_length']
    df['puncta_circularity'] = (4 * np.pi * df['puncta_area']) / (df['puncta_perimeter'] ** 2)
    df['coi1_partition_coeff'] = df['puncta_intensity_mean'] / df['cell_coi1_dilute_intensity_mean']
    df['coi2_partition_coeff'] = df['puncta_intensity_mean_in_coi2'] / df['cell_coi2_dilute_intensity_mean']

    return df


def aggregate_features_by_group(df, group_cols, agg_cols, agg_func='mean'):
    """
    Aggregate multiple columns by group and merge results into a single DataFrame.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        group_cols (list): Columns to group by.
        agg_cols (list): Columns to aggregate.
        agg_func (str or callable): Aggregation function, default is 'mean'.

    Returns:
        pd.DataFrame: Aggregated dataframe with group_cols and agg_cols.
    """
    grouped_dfs = []
    for col in agg_cols:
        agg_df = df.groupby(group_cols)[col].agg(agg_func).reset_index()
        grouped_dfs.append(agg_df)

    merged_df = functools.reduce(
        lambda left, right: left.merge(right, on=group_cols),
        grouped_dfs
    )
    return merged_df.reset_index(drop=True)


# --- Proof Plotting ---
def generate_proofs(df, image_dict):
    # logger.info('generating proof plots...')

    for name, img in image_dict.items():
        contour = df.loc[df['image_name'] == name, 'cell_coords']
        coord_list = df.loc[df['image_name'] == name, 'puncta_coords']

        if contour.empty:
            continue

        coi2, coi1, mask = img
        cell_img = coi1 * (mask > 0)

        h, w = coi1.shape
        aspect = w / h

        # figure size adapts to image aspect
        base_height = 5
        fig_width = base_height * aspect * 3   # three panels
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, base_height), constrained_layout=True)

        # scale linewidth based on image size
        lw = max(0.5, min(h, w) / 2000)
        lw = 0.5

        # --- overlay image ---
        ax1.imshow(coi1, cmap='gray_r')
        ax1.imshow(coi2, cmap='Blues', alpha=0.6)

        # --- coi only ---
        ax3.imshow(coi1, cmap='gray_r')

        # --- cell panel ---
        ax2.imshow(cell_img, cmap='gray_r')

        for line in contour.iloc[0]:
            ax2.plot(line[:, 1], line[:, 0], c='k', lw=lw)

        if len(coord_list) > 1:
            for puncta in coord_list:
                if isinstance(puncta, np.ndarray):
                    ax2.plot(puncta[:, 1], puncta[:, 0], lw=0.1)

        # scalebar stays fractional already
        scalebar = ScaleBar(SCALE_PX, SCALE_UNIT, location='lower right', pad=0.3, sep=2, box_alpha=0, color='gray', length_fraction=0.3)
        ax1.add_artist(scalebar)

        # --- axis-relative text placement ---
        ax1.text(0.02, 0.95, COI_1_name, color='gray', transform=ax1.transAxes, ha='left', va='top')
        ax1.text(0.02, 0.88, COI_2_name, color='steelblue', transform=ax1.transAxes, ha='left', va='top')
        ax3.text(0.02, 0.95, COI_1_name, color='gray', transform=ax3.transAxes, ha='left', va='top')

        for ax in (ax1, ax2, ax3):
            ax.set_axis_off()

        fig.suptitle(name, y=1.05)

        fig.savefig(f'{proofs_folder}{name}_proof.png', dpi=600, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]

    all_features = []

    for fname in image_files:
        name = fname.removesuffix('.npy')

        mask_path = os.path.join(mask_folder, f"{name}_mask.npy")
        if not os.path.exists(mask_path):
            continue

        logger.info(f'processing {name}')

        # --- load only this image ---
        image = np.load(os.path.join(image_folder, fname))
        mask = np.load(mask_path, allow_pickle=True)

        cell_mask = mask[0]

        # --- wrangle channels to match expected input for saturation check ---
        if 'mCh' in name:
            image = np.stack([image[COI_2], image[COI_1]])
        else:
            image = np.stack([image[COI_1], image[COI_2]])

        # --- filter ---
        filtered_img = filter_saturated_images(
            {name: image}, 
            {name: cell_mask}
        )[name]

        # --- features (single image only) ---
        df = collect_features({name: filtered_img})
        if df.empty:
            logger.warning(f"No valid features for {name}, skipping")
            continue
        df = extra_puncta_features(df)

        # --- proof immediately ---
        generate_proofs(df, {name: filtered_img})
        cols_to_drop = [col for col in df.columns if '_coords' in col]
        df = df.drop(columns=cols_to_drop)

        # --- store results ---
        all_features.append(df)

        # free memory
        del image, mask, filtered_img, df

    features = pd.concat(all_features, ignore_index=True)

    # --- data wrangling ---
    logger.info('starting data wrangling and saving...')
    features['tag'] = ['EYFP' for name in features['image_name']]
    features['condition'] = features['image_name'].str.split('-').str[0]
    features['rep'] = features['image_name'].str.split('-').str[-2]

    cols = features.columns.tolist()
    cols = [item for item in cols if '_coords' not in item]
    cols = ['puncta_area', 'puncta_eccentricity', 'puncta_aspect_ratio',
            'puncta_circularity', 'puncta_cv', 'puncta_skew',
            'coi2_partition_coeff', 'coi1_partition_coeff', 'cell_std',
            'cell_cv', 'cell_skew']

    # --- data trimming and saving ---
    # trim off coordinates used for proofs, save the main features dataframe
    cols_to_drop = [col for col in features.columns if '_coords' in col]
    features = features.drop(columns=cols_to_drop)
    features.to_csv(f'{output_folder}puncta_features.csv', index=False)

    # save averages per biological replicate
    rep_df = aggregate_features_by_group(features, ['condition', 'tag', 'rep'], cols)
    rep_df.to_csv(f'{output_folder}puncta_features_reps.csv', index=False)

    # save features normalized to cell intensity of channel of interest
    df_norm = features.copy()
    for col in cols:
        df_norm[col] /= df_norm['cell_coi1_intensity_mean']
    df_norm.to_csv(f'{output_folder}puncta_features_normalized.csv', index=False)

    # save normalized averages per biological replicate
    rep_norm_df = aggregate_features_by_group(df_norm, ['condition', 'tag', 'rep'], cols)
    rep_norm_df.to_csv(f'{output_folder}puncta_features_normalized_reps.csv', index=False)

    logger.info('data wrangling and saving complete.')
    logger.info('pipeline complete.')
