import numpy as np
import pandas as pd
from PIL import Image
import os
from typing import Tuple, List
import slidingwindow
from pathlib import Path
import glob
def compute_windows(numpy_image, patch_size, patch_overlap):
    """Create a sliding window object from a raster tile.

    Args:
        numpy_image (numpy array): Raster object as numpy array to cut into crops

    Returns:
        windows (list): a sliding windows object
    """

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return (windows)


def save_crop(base_dir, image_name, index, tile_pos, crop):
    """Save cropped image tile to disk."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Create filename: originalname_tile_index_xmin_ymin_xmax_ymax.png
    name_parts = os.path.splitext(image_name)
    x_min, y_min, x_max, y_max = tile_pos
    tile_filename = f"{name_parts[0]}_tile_{index}_{x_min}_{y_min}_{x_max}_{y_max}{name_parts[1]}"
    
    image_path = os.path.join(base_dir, tile_filename)
    Image.fromarray(crop).save(image_path)
    
    return image_path


def tile_xy(window_indices):
    """Extract tile position from window indices."""
    (y_slice, x_slice) = window_indices
    return (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)


def split_raster_with_metadata(path_to_raster: str,
                                metadata_df: pd.DataFrame,
                                base_dir: str = "tiled_images",
                                patch_size: int = 1000,
                                patch_overlap: float = 0.5) -> pd.DataFrame:
    """
    Split a drone image into tiles and update associated metadata.
    
    Args:
        path_to_raster: Path to the drone image
        metadata_df: DataFrame containing metadata with bounding boxes
        base_dir: Directory to save tiled images
        patch_size: Size of each tile (e.g., 1000 for 1000x1000)
        patch_overlap: Overlap between tiles (0.5 = 50% overlap, giving 1500x1500 effective coverage)
    
    Returns:
        DataFrame with updated metadata for all tiles from this image
    """
    # Load raster as image
    ##print(f"Opening path:{path_to_raster.name}")
    raster = Image.open(path_to_raster)
    numpy_image = np.array(raster)
    numpy_image = numpy_image[:, :, :3]
    
    # Validate image
    bands = numpy_image.shape[2]
    if bands != 3:
        raise IOError(f"Input file {path_to_raster} has {bands} bands. Expected 3 band RGB.")
    
    height, width = numpy_image.shape[:2]
    if any(np.array([height, width]) < patch_size):
        raise ValueError(f"Patch size {patch_size} is larger than image dimensions {[height, width]}")
    
    # Compute sliding windows
    windows = compute_windows(numpy_image, patch_size, patch_overlap)
    
    # Get image name and filter metadata for this specific image
    image_name = os.path.basename(path_to_raster)
    image_metadata = metadata_df[metadata_df['img_path'] == path_to_raster.name].copy()
    
    print(f"Processing | {image_name}: found {len(image_metadata)} metadata records")
    
    all_tile_metadata = []
    
    for index, window in enumerate(windows):
        # Get tile boundaries in original image coordinates
        tile_pos = tile_xy(window.indices())
        x_min, y_min, x_max, y_max = tile_pos
        
        # Filter metadata: keep records where bounding boxes overlap with this tile
        # A bounding box overlaps if: xmin < tile_xmax AND xmax > tile_xmin AND ymin < tile_ymax AND ymax > tile_ymin
        tile_metadata = image_metadata[
            (image_metadata['xmin'] < x_max) & 
            (image_metadata['xmax'] > x_min) &
            (image_metadata['ymin'] < y_max) & 
            (image_metadata['ymax'] > y_min)
        ].copy()
        
        # Only save tiles that have associated metadata
        if len(tile_metadata) > 0:
            # Crop and save image
            crop = numpy_image[window.indices()]
            tile_image_path = save_crop(base_dir, image_name, index, tile_pos, crop)
            
            # Update metadata for this tile
            tile_metadata['original_img_path'] = path_to_raster
            tile_metadata['new_tile_index'] = index
            tile_metadata['new_tile_xmin'] = x_min
            tile_metadata['new_tile_ymin'] = y_min
            tile_metadata['new_tile_xmax'] = x_max
            tile_metadata['new_tile_ymax'] = y_max
            
            # Adjust bounding box coordinates to be relative to the new tile
            tile_metadata['xmin_tile'] = tile_metadata['xmin'] - x_min
            tile_metadata['ymin_tile'] = tile_metadata['ymin'] - y_min
            tile_metadata['xmax_tile'] = tile_metadata['xmax'] - x_min
            tile_metadata['ymax_tile'] = tile_metadata['ymax'] - y_min
            
            # Adjust center coordinates
            tile_metadata['x_tile'] = tile_metadata['x'] - x_min
            tile_metadata['y_tile'] = tile_metadata['y'] - y_min
            
            # Clip coordinates to tile boundaries (for objects partially outside)
            tile_metadata['xmin_tile'] = tile_metadata['xmin_tile'].clip(lower=0)
            tile_metadata['ymin_tile'] = tile_metadata['ymin_tile'].clip(lower=0)
            tile_metadata['xmax_tile'] = tile_metadata['xmax_tile'].clip(upper=patch_size)
            tile_metadata['ymax_tile'] = tile_metadata['ymax_tile'].clip(upper=patch_size)
            
            # Update img_path to point to new tiled image
            tile_metadata['img_path'] = os.path.basename(tile_image_path)
            
            all_tile_metadata.append(tile_metadata)
            
            print(f"  Tile {index} ({x_min},{y_min} to {x_max},{y_max}): {len(tile_metadata)} objects")
    
    if all_tile_metadata:
        return pd.concat(all_tile_metadata, ignore_index=True)
    else:
        print(f"  Warning: No metadata found for any tiles in {image_name}")
        return pd.DataFrame()

def find_path_files(directory_path:str, globb: str = "png"):
    """
    Find all files inside multiple folders given by a glob string.
    """
    # Create a Path object for the folder
    path = Path(directory_path)
    
    # rglob stands for "recursive glob"
    # It searches for any file ending in .png in all subdirectories
    image_paths = sorted(list(path.rglob(f"*.{globb}")))
    
    ## exclude bad paths
    ## took it from torchgeo 
    # https://github.com/gyrrei/ReforesTree/issues/6
    bad_paths = [
        'Carlos Vera Guevara RGB_15_8425_8305_12425_12305.png',
        'Flora Pluas RGB_3_0_11400_4000_15400.png',
        'Flora Pluas RGB_4_0_11578_4000_15578.png',
        'Flora Pluas RGB_23_12782_11400_16782_15400.png',
        'Flora Pluas RGB_24_12782_11578_16782_15578.png',
    ]
    final_paths = []
    for path in image_paths:
        if os.path.basename(path) not in bad_paths:
            final_paths.append(path)
    return final_paths


def process_all_images(metadata_df: pd.DataFrame,
                       base_dir_tiles: str = "dataset/tiles",
                       base_dir: str = "tiled_images",
                       patch_size: int = 1000,
                       patch_overlap: float = 0.5,
                       output_csv: str = "tiled_metadata.csv") -> pd.DataFrame:
    """
    Process all unique images in the metadata DataFrame.
    
    Args:
        metadata_df: DataFrame with metadata
        base_dir_tiles: Directory containing the dataset with the tiles.
        base_dir: Directory to save tiled images
        patch_size: Size of each tile
        patch_overlap: Overlap fraction between tiles
        output_csv: Path to save the updated metadata CSV
    
    Returns:
        Combined DataFrame with metadata for all tiles
    """
    unique_images = metadata_df['img_path'].unique()
    print(f"Found {len(unique_images)} unique images to process")
    
    all_metadata = []
    
    ## find all files inside the folder 
    list_img_paths = find_path_files(base_dir_tiles, "png")

    ## create a df to store these paths.
    df = pd.DataFrame()
    df['full_path'] = list_img_paths
    df['name_file'] = df['full_path'].apply(lambda x:x.name)

    ## loop over the final_df 
    for img_path in unique_images[:1]:
        try:
            ## find the full path
            full_img_path = df.loc[df["name_file"]==img_path]["full_path"].values[0]

            ## run 
            tile_metadata = split_raster_with_metadata(
                full_img_path, 
                metadata_df, 
                base_dir=base_dir,
                patch_size=patch_size,
                patch_overlap=patch_overlap
            )
            
            if not tile_metadata.empty:
                all_metadata.append(tile_metadata)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if all_metadata:
        final_metadata_df = pd.concat(all_metadata, ignore_index=True)
        final_metadata_df.to_csv(output_csv, index=False)
        print(f"\nSaved tiled metadata to {output_csv}")
        print(f"Total records: {len(final_metadata_df)}")
        return final_metadata_df
    else:
        print("No metadata generated")
        return pd.DataFrame()


# Example usage:
if __name__ == "__main__":
    # Load your metadata
    metadata_df = pd.read_csv("/home/camarada/Documents/CDE/3-semester/DeepLearning/ReforesTree/dataset/mapping/final_dataset.csv")
    
    save_dir = "new_data"
    os.makedirs(save_dir,exist_ok=True)
    # Process all images with 1000x1000 tiles and 50% overlap (effective 1500x1500 coverage)
    tiled_metadata = process_all_images(
        metadata_df,
        base_dir=f"{save_dir}/tiled_images",
        patch_size=1000,
        patch_overlap=0.25,  # 50% overlap gives you the 1250x1250 coverage you want
        output_csv=f"{save_dir}/tiled_metadata.csv"
    )
    
    print(f"\nGenerated {len(tiled_metadata)} metadata records across all tiles")