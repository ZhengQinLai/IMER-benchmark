import os
import cv2
import numpy as np
import pandas as pd
import math
from PIL import Image
from tqdm import tqdm # Used for displaying progress bar
import sys
import re # Added for SAMM filename matching

# --- Base Configuration ---
#
# This script is meant to be shared, so all key paths are configurable via
# environment variables. If not provided, we default to paths under the repo's
# `dataset/` folder.
#
# You can set these variables instead of editing the file:
#   - BASE_DATA_PATH:      where raw datasets are located (casme2/, samm/, mmew/, casme3/, dfme/, ...)
#   - BASE_LANDMARKS_PATH: where landmarks are located (default: <BASE_DATA_PATH>/landmarks)
#   - OUTPUT_BASE_DIR:     where optical-flow PNGs will be written (default: <BASE_DATA_PATH>/mix_me_all)
DEFAULT_BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
BASE_DATA_PATH = os.getenv("BASE_DATA_PATH", DEFAULT_BASE_DATA_PATH)  # Dataset root directory
BASE_LANDMARKS_PATH = os.getenv(
    "BASE_LANDMARKS_PATH",
    os.path.join(BASE_DATA_PATH, "landmarks"),
)  # Landmarks root directory
OUTPUT_BASE_DIR = os.getenv(
    "OUTPUT_BASE_DIR",
    os.path.join(BASE_DATA_PATH, "mix_me_all"),
)  # Root directory for output optical flow images

# --- DFME Configuration (5th dataset) ---
# DFME is optional and uses an Excel coding file (coding.xlsx). We keep its
# paths configurable via environment variables so the script can be shared.
#
# Suggested layout (editable):
#   <BASE_DATA_PATH>/dfme/train_data/
#     - coding.xlsx
#     - <SequenceName>/{00000.png,00001.png,...}
DFME_BASE_DIR = os.getenv("DFME_BASE_DIR", os.path.join(BASE_DATA_PATH, "dfme"))
DFME_FRAMES_ROOT = os.getenv("DFME_FRAMES_ROOT", os.path.join(DFME_BASE_DIR, "train_data"))
DFME_CODING_PATH = os.getenv("DFME_CODING_PATH", os.path.join(DFME_FRAMES_ROOT, "coding.xlsx"))
# Whether to additionally save onset->(apex +/- {1,2,3}) optical flow images.
DFME_SAVE_EXTRA_FLOWS = os.getenv("DFME_SAVE_EXTRA_FLOWS", "1") in ("1", "true", "True", "yes", "Y")


# --- Dataset Specific Configuration ---
# Each dataset entry defines:
#   - coding_file: usually "coding.csv", but DFME uses "coding.xlsx" by default
#   - path builder lambdas for onset/apex frames and landmark files
# Example (CASME2): <BASE_DATA_PATH>/casme2/coding.csv
DATASETS_CONFIG = {
    'casme2': {
        'coding_file': 'coding.csv',
        'labels': ('disgust', 'happiness', 'repression', 'surprise', 'others'),
        'ftype': '.jpg',
        'needs_crop': True, # Assume CASME II needs cropping
        # Lambda function to construct path based on CSV row information
        'landmark_file_func': lambda r: os.path.join(BASE_LANDMARKS_PATH, 'casme2', f'sub{str(r.Subject).zfill(2)}', f'{r.Filename}.npy'),
        'onset_path_func': lambda r, p: os.path.join(p, 'casme2', f'sub{str(r.Subject).zfill(2)}', r.Filename, f'img{int(r.OnsetFrame)}{r["ftype"]}'),
        'apex_path_func': lambda r, p: os.path.join(p, 'casme2', f'sub{str(r.Subject).zfill(2)}', r.Filename, f'img{int(r.ApexFrame)}{r["ftype"]}'),
        'label_col': 'Emotion', # Name of the label column in CSV
        # Output filename format
        'output_filename_func': lambda r, p: f"sub{str(r.Subject).zfill(2)}_{r.Filename}_onset{int(r.OnsetFrame)}_apex{int(r.ApexFrame)}.png"
    },
    'samm': {
        'coding_file': 'coding.csv', # !!! Please confirm the coding filename for SAMM !!!
        'labels': ('Other', 'Anger', 'Contempt', 'Happiness', 'Surprise'),
        'ftype': '.jpg', # !!! Please confirm the image format for SAMM, CK+ was png in the original code !!!
        'needs_crop': True, # Assume SAMM needs cropping
        'landmark_file_func': lambda r: os.path.join(BASE_LANDMARKS_PATH, 'samm', str(r.Subject).zfill(3), f'{r.Filename}.npy'),
        'onset_path_func': lambda r, p: os.path.join(p, 'samm', str(r.Subject).zfill(3), r.Filename, f'{str(r.Subject).zfill(3)}_{str(int(r.OnsetFrame)).zfill(5)}{r["ftype"]}'),
        'apex_path_func': lambda r, p: os.path.join(p, 'samm', str(r.Subject).zfill(3), r.Filename, f'{str(r.Subject).zfill(3)}_{str(int(r.ApexFrame)).zfill(5)}{r["ftype"]}'),
        'label_col': 'Emotion', # !!! Please confirm the label column name in the SAMM coding file !!!
        'output_filename_func': lambda r, p: f"sub{str(r.Subject).zfill(3)}_{r.Filename}_onset{int(r.OnsetFrame)}_apex{int(r.ApexFrame)}.png"
    },
    'mmew': {
        'coding_file': 'coding.csv', # !!! Please confirm the coding filename for MMEW !!!
        'labels': ('others', 'disgust', 'happiness', 'surprise', 'fear'),
        'ftype': '.jpg', # !!! Please confirm the image format for MMEW !!!
        'needs_crop': False, # Assume MMEW does not need cropping
        'landmark_file_func': lambda r: os.path.join(BASE_LANDMARKS_PATH, 'mmew', r.Emotion, r.Filename, f'{r.Filename}.npy'), # !!! Carefully check the MMEW landmark path structure !!!
        'onset_path_func': lambda r, p: os.path.join(p, 'mmew', r.Emotion, r.Filename, f'{int(r.OnsetFrame)}{r["ftype"]}'), # !!! Carefully check the MMEW image path structure !!!
        'apex_path_func': lambda r, p: os.path.join(p, 'mmew', r.Emotion, r.Filename, f'{int(r.ApexFrame)}{r["ftype"]}'), # !!! Carefully check the MMEW image path structure !!!
        'label_col': 'Emotion', # !!! Please confirm the label column name in the MMEW coding file !!!
        'output_filename_func': lambda r, p: f"{r.Emotion}_{r.Filename}_onset{int(r.OnsetFrame)}_apex{int(r.ApexFrame)}.png"
    },
    'casme3': {
        'coding_file': 'coding.csv', # !!! Please confirm the coding filename for CASME3 !!!
        'labels': ('disgust', 'happy', 'surprise', 'others', 'anger', 'sad', 'fear'),
        'ftype': '.jpg', # !!! Please confirm the image format for CASME3 !!!
        'needs_crop': False, # Assume CASME3 does not need cropping
        # !!! The path logic for CASME3 is quite complex, please double-check carefully !!!
        # !!! Pay special attention to landmark filenames and paths, 'color/color.npy' in the original code looks very specific !!!
        'landmark_file_func': lambda r: os.path.join(BASE_LANDMARKS_PATH, 'casme3', f'spNO.{str(r.Subject)}', r.Filename.lower(), 'color', 'color.npy'),
        # !!! Please confirm the column names for onset/apex frames in the CASME3 coding file (is it Onset/Apex or OnsetFrame/ApexFrame?) !!!
        'onset_path_func': lambda r, p: os.path.join(p, 'casme3', r.emotion, f'spNO.{str(r.Subject)}{r.Filename.lower()}{int(r.Apex)}', f'{int(r.Onset)}{r["ftype"]}'),
        'apex_path_func': lambda r, p: os.path.join(p, 'casme3', r.emotion, f'spNO.{str(r.Subject)}{r.Filename.lower()}{int(r.Apex)}', f'{int(r.Apex)}{r["ftype"]}'),
        'label_col': 'emotion', # !!! Please confirm the label column name in the CASME3 coding file !!!
        'output_filename_func': lambda r, p: f"sub{str(r.Subject)}_{r.Filename.lower()}_onset{int(r.Onset)}_apex{int(r.Apex)}.png"
    },
    # 5th dataset: DFME (coding.xlsx + per-sequence frame folders)
    'dfme': {
        # Absolute path is allowed (we set a default via DFME_CODING_PATH).
        'coding_file': DFME_CODING_PATH,
        # Emotions as given by DFME coding.xlsx (recommended to normalize to lowercase).
        'labels': ('anger', 'contempt', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'),
        'ftype': '.png',
        'needs_crop': True,
        # Landmarks are expected at: <BASE_LANDMARKS_PATH>/dfme/<Filename>.npy
        'landmark_file_func': lambda r: os.path.join(BASE_LANDMARKS_PATH, 'dfme', f'{r.Filename}.npy'),
        # Frames are expected at: <DFME_FRAMES_ROOT>/<Filename>/<frame:05d>.png
        'onset_path_func': lambda r, p: os.path.join(
            DFME_FRAMES_ROOT,
            str(r.Filename),
            f"{int(r.OnsetFrame):05d}{r['ftype']}",
        ),
        'apex_path_func': lambda r, p: os.path.join(
            DFME_FRAMES_ROOT,
            str(r.Filename),
            f"{int(r.ApexFrame):05d}{r['ftype']}",
        ),
        'label_col': 'Emotion',
        # Output filename format (Subject can be int or string; keep original value).
        'output_filename_func': lambda r, p: f"{r.Subject}_{r.Filename}_onset{int(r.OnsetFrame)}_apex{int(r.ApexFrame)}.png",
    },
}

# --- Helper Functions ---

def load_image(img_path):
    """Load image using OpenCV (BGR), trying SAMM zfill(4) if zfill(5) fails."""
    image = cv2.imread(img_path)

    if image is None:
        # --- SAMM Specific Fallback Logic ---
        # Check if it's potentially a SAMM path with zfill(5)
        parts = img_path.split(os.sep)
        filename = os.path.basename(img_path)
        # Regex to match SAMM filename pattern: subXXX_frameYYYYY.ext
        samm_pattern_z5 = re.compile(r'^\d{3}_\d{5}\.\w+$')

        # Check if 'samm' is in the path and filename matches the zfill(5) pattern
        if 'samm' in parts and samm_pattern_z5.match(filename):
            try:
                # Construct the zfill(4) alternative path
                base, ext = os.path.splitext(filename)
                subject_part, frame_part = base.split('_')
                # Ensure frame_part is indeed 5 digits before trying zfill(4)
                if len(frame_part) == 5:
                    frame_num_int = int(frame_part) # Convert to int to remove leading zeros
                    alt_frame_part = str(frame_num_int).zfill(4)
                    alt_filename = f"{subject_part}_{alt_frame_part}{ext}"
                    alt_path = os.path.join(os.path.dirname(img_path), alt_filename)
                    # print(f"Trying SAMM alternative path (zfill(4)): {alt_path}") # Keep commented out
                    image = cv2.imread(alt_path)
            except Exception as e:
                # print(f"Error trying SAMM zfill(4) alternative: {e}") # Keep commented out
                pass # Ignore errors during alternative path construction/loading
        # --- End SAMM Specific Fallback Logic ---

        # --- Original Alternative Path Logic (Commented out as potentially redundant/conflicting) ---
        # if image is None: # Check again if SAMM logic didn't find the image
        #     if len(os.path.basename(img_path)) >= 9:
        #          alt_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path)[:-9] + os.path.basename(img_path)[-8:])
        #          # print(f"Trying original alternative path: {alt_path}")
        #          image = cv2.imread(alt_path)
        # --- End Original Alternative Path Logic ---


    if image is None:
         # print(f"Warning: Could not load image: {img_path}") # Keep commented out
         # Added note about checked paths in error message
         raise FileNotFoundError(f"Could not load image: {img_path} (checked zfill(5)/zfill(4) for SAMM if applicable)")

    # Subsequent processing uses RGB, convert to RGB first
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def calculate_optical_flow(onset_img_rgb, apex_img_rgb):
    """Calculate Farneback optical flow and return the flow image in BGR format."""
    if onset_img_rgb is None or apex_img_rgb is None:
        raise ValueError("Input image for optical flow calculation is None.")
    if onset_img_rgb.shape != apex_img_rgb.shape:
        # print(f"Warning: Onset ({onset_img_rgb.shape}) and Apex ({apex_img_rgb.shape}) shapes differ. Attempting to resize Apex.")
        apex_img_rgb = cv2.resize(apex_img_rgb, (onset_img_rgb.shape[1], onset_img_rgb.shape[0]))

    onset_gray = cv2.cvtColor(onset_img_rgb, cv2.COLOR_RGB2GRAY)
    apex_gray = cv2.cvtColor(apex_img_rgb, cv2.COLOR_RGB2GRAY)

    if onset_gray.shape != apex_gray.shape:
         raise ValueError(f"Grayscale image shapes do not match: {onset_gray.shape} vs {apex_gray.shape}")

    hsv = np.zeros_like(onset_img_rgb)
    hsv[..., 1] = 255 # Set saturation to maximum

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(onset_gray, apex_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert Cartesian coordinates to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2 # Hue represents direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Value represents magnitude

    # Convert HSV to BGR for saving
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr_flow


def save_dfme_extra_flows(row, onset_img_rgb, crop_box, output_dir, base_filename_root):
    """
    DFME-only: in addition to the onset->apex flow, optionally save extra flow maps
    between the onset frame and 6 frames around the apex frame:
        ApexFrame+/-1, ApexFrame+/-2, ApexFrame+/-3.

    Each extra flow image is saved as:
        <base_filename_root>_f{t}.png
    where `t` is the target frame index used for onset->t.
    """
    try:
        apex_frame = int(row["ApexFrame"])
    except Exception:
        return

    neighbor_offsets = [-3, -2, -1, 1, 2, 3]
    try:
        seq_dir = os.path.join(DFME_FRAMES_ROOT, str(row["Filename"]))
        ftype = row["ftype"]
    except Exception:
        return

    for off in neighbor_offsets:
        t = apex_frame + off
        if t < 0:
            continue

        target_path = os.path.join(seq_dir, f"{int(t):05d}{ftype}")
        try:
            target_img_rgb = load_image(target_path)
        except Exception:
            # Skip missing frames
            continue

        if crop_box is not None:
            target_img_rgb = crop_image(target_img_rgb, crop_box)
            if target_img_rgb.size == 0 or onset_img_rgb.size == 0:
                continue

        try:
            flow_bgr = calculate_optical_flow(onset_img_rgb, target_img_rgb)
        except Exception:
            continue

        out_name = f"{base_filename_root}_f{int(t)}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, flow_bgr)


def get_landmark_crop_box(landmark_file_path):
    """
    Load landmark file and calculate the cropping box.
    Returns: (ymin, ymax, xmin, xmax) for numpy slicing.
    !!! Please carefully check if the landmark calculation logic and coordinate order match your data !!!
    """
    if not os.path.exists(landmark_file_path):
        raise FileNotFoundError(f"Landmark file not found: {landmark_file_path}")
    try:
        landmark = np.load(landmark_file_path)
        if landmark.shape[0] < 58 or landmark.shape[1] != 2:
             raise ValueError(f"Invalid landmark file format or insufficient points: {landmark_file_path} (shape: {landmark.shape})")

        # Use a dataset-specific rule:
        #   - For DFME we rely on the full landmark bounding box (to cover the whole face);
        #   - For other datasets we keep the original 30/8/19/57 based heuristic.
        norm_path_parts = os.path.normpath(landmark_file_path).split(os.sep)
        is_dfme = "dfme" in norm_path_parts

        if is_dfme:
            # NOTE: DFME landmarks may be stored as (y, x). If your landmarks are
            # (x, y), swap the axes below.
            ys = landmark[:, 0]
            xs = landmark[:, 1]
            ymin_raw, ymax_raw = ys.min(), ys.max()
            xmin_raw, xmax_raw = xs.min(), xs.max()

            cx = (xmin_raw + xmax_raw) / 2.0
            cy = (ymin_raw + ymax_raw) / 2.0
            face_h = ymax_raw - ymin_raw
            face_w = xmax_raw - xmin_raw
            side = max(face_w, face_h)

            # Slight enlargement ratio around the face.
            scale = 1.1
            half = (side * scale) / 2.0

            # Shift the crop centre slightly upwards (towards forehead) so that
            # less neck / below-chin area is included.
            shift_up = 0.10 * face_h
            cy = cy - shift_up

            ymin = math.floor(cy - half)
            ymax = math.floor(cy + half)
            xmin = math.floor(cx - half)
            xmax = math.floor(cx + half)
        else:
            # Calculation method from original code (based on landmarks 30, 8, 19, 57)
            centre_y, centre_x = landmark[30] # Assume landmarks are stored as (y, x)
            y8, y19, y57 = landmark[8][1], landmark[19][1], landmark[57][1]

            # Height calculation in original code: height = (y8 - y19) + (y8 - y57)
            height = (y8 - y19) + (y8 - y57)

            if height <= 0:
                 height = 100 # Provide a fallback value

            half_height = height / 2
            # Calculate cropping box (ymin, ymax, xmin, xmax) for img[ymin:ymax, xmin:xmax]
            ymin = math.floor(centre_y - half_height)
            ymax = math.floor(centre_y + half_height)
            xmin = math.floor(centre_x - half_height) # Assume square cropping
            xmax = math.floor(centre_x + half_height)

        return int(ymin), int(ymax), int(xmin), int(xmax)

    except IndexError:
         raise IndexError(f"Landmark index out of bounds (needs 8, 19, 30, 57): {landmark_file_path}")
    except Exception as e:
        print(f"Error processing landmark file {landmark_file_path}: {e}")
        raise # Re-raise the exception


def crop_image(img_rgb, box):
    """Crop RGB image using (ymin, ymax, xmin, xmax)."""
    ymin, ymax, xmin, xmax = box
    h, w = img_rgb.shape[:2]

    # Boundary check
    ymin = max(0, ymin)
    xmin = max(0, xmin)
    ymax = min(h, ymax)
    xmax = min(w, xmax)

    if ymin >= ymax or xmin >= xmax:
        # print(f"Warning: Invalid crop box [{ymin}:{ymax}, {xmin}:{xmax}] (image size {img_rgb.shape}). Returning original image.")
        return img_rgb # Return the original image if the crop box is invalid

    return img_rgb[ymin:ymax, xmin:xmax, :]

# --- Main Processing Flow ---

print(f"Starting data processing, output will be saved to: {OUTPUT_BASE_DIR}")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

total_processed = 0
total_errors = 0

for dataset_name, config in DATASETS_CONFIG.items():
    print(f"\n--- Starting processing dataset: {dataset_name} ---")

    coding_file = str(config["coding_file"])
    # Allow either:
    #   - a basename like "coding.csv" (resolved under <BASE_DATA_PATH>/<dataset_name>/),
    #   - or a full/relative path (used by DFME by default via DFME_CODING_PATH).
    if os.path.isabs(coding_file) or ("/" in coding_file) or ("\\" in coding_file):
        coding_path = coding_file
    else:
        coding_path = os.path.join(BASE_DATA_PATH, dataset_name, coding_file)
    if not os.path.exists(coding_path):
        print(f"Error: Coding file for {dataset_name} not found: {coding_path}. Skipping this dataset.")
        continue

    try:
        ext = os.path.splitext(coding_path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            data_info = pd.read_excel(coding_path)
        else:
            data_info = pd.read_csv(coding_path)
        # Add dataset-specific info to DataFrame for easier lambda function calls
        data_info['ftype'] = config['ftype']
        data_info['dataset'] = dataset_name # Although not used in lambdas, it might be useful
    except Exception as e:
        print(f"Error: Failed to read coding file {coding_path}: {e}. Skipping this dataset.")
        continue

    dataset_processed = 0
    dataset_errors = 0

    # Use tqdm to display progress bar
    for index, row in tqdm(
        data_info.iterrows(),
        total=data_info.shape[0],
        desc=f"Processing {dataset_name}",
        file=sys.stdout,
    ):
        try:
            label = row[config["label_col"]]
            # DFME labels in Excel are often mixed-case; normalize for matching.
            if dataset_name == "dfme":
                label = str(label).strip().lower()

            # Check if the label is in the defined list
            if label not in config["labels"]:
                continue  # Skip invalid label

            # Get onset and apex image paths
            onset_path = config["onset_path_func"](row, BASE_DATA_PATH)
            apex_path = config["apex_path_func"](row, BASE_DATA_PATH)

            # --- 1. Load Images (RGB) ---
            onset_img_rgb = load_image(onset_path)
            apex_img_rgb = load_image(apex_path)

            crop_box = None
            # --- 2. Crop Images (if needed) ---
            if config["needs_crop"]:
                landmark_file = config["landmark_file_func"](row)
                try:
                    crop_box = get_landmark_crop_box(landmark_file)
                    onset_img_rgb = crop_image(onset_img_rgb, crop_box)
                    apex_img_rgb = crop_image(apex_img_rgb, crop_box)
                    if onset_img_rgb.size == 0 or apex_img_rgb.size == 0:
                        raise ValueError("Image is empty after cropping.")
                except FileNotFoundError:
                    dataset_errors += 1
                    continue
                except Exception:
                    dataset_errors += 1
                    continue

            # --- 3. Calculate Optical Flow (Input RGB, Output BGR) ---
            optical_flow_bgr = calculate_optical_flow(onset_img_rgb, apex_img_rgb)

            # --- 4. Prepare Output Path and Filename ---
            label_foldername = str(label).lower().replace(" ", "_")
            output_dir = os.path.join(OUTPUT_BASE_DIR, f"{dataset_name}_{label_foldername}")
            os.makedirs(output_dir, exist_ok=True)

            output_filename = config["output_filename_func"](row, BASE_DATA_PATH)
            output_path = os.path.join(output_dir, output_filename)

            # --- 5. Save Optical Flow Image (BGR format) ---
            success = cv2.imwrite(output_path, optical_flow_bgr)
            if not success:
                dataset_errors += 1
            else:
                # DFME: optionally save auxiliary flows onset->(apex +/- 1/2/3)
                if dataset_name == "dfme" and DFME_SAVE_EXTRA_FLOWS:
                    base_filename_root, _ = os.path.splitext(output_filename)
                    save_dfme_extra_flows(
                        row=row,
                        onset_img_rgb=onset_img_rgb,
                        crop_box=crop_box,
                        output_dir=output_dir,
                        base_filename_root=base_filename_root,
                    )
                dataset_processed += 1

        # --- Error Handling ---
        except FileNotFoundError:
            dataset_errors += 1
            continue
        except ValueError:
            dataset_errors += 1
            continue
        except Exception as e:
            print(
                f"\nUnknown error occurred while processing index {index} "
                f"in dataset {dataset_name}: {e}"
            )
            dataset_errors += 1
            continue

    print(f"\n--- Finished processing dataset: {dataset_name} ---")
    print(f"Successfully processed: {dataset_processed} items")
    print(f"Errors/Skipped: {dataset_errors} items")
    total_processed += dataset_processed
    total_errors += dataset_errors

print("\n===================================")
print("All dataset processing complete.")
print(f"Total successfully processed: {total_processed} items")
print(f"Total errors/skipped: {total_errors} items")
print(f"Output files saved in: {OUTPUT_BASE_DIR}")
print("===================================")
