# Incremental Learning Benchmark for Micro-Expression Recognition

## Overview

This project provides a framework for benchmarking various incremental learning algorithms specifically tailored for micro-expression recognition tasks. It includes implementations of several main incremental learning methods and supports different backbone architectures like Vision Transformers (ViT), ResNet, and Swin Transformer. The primary dataset used is a combination of micro-expression datasets, referred to as **iMER** within the project (CASME II, SAMM, MMEW, CAS(ME)^3, and DFME as the 5th dataset/session).

## Directory Structure

```
.
├── backbone/         # Contains implementations of different neural network backbones (ViT, ResNet, Swin) and prompting mechanisms (L2P, DualPrompt, VPT, etc.)
├── dataset/          # Data directory, expected to contain the iMER dataset (e.g., mix_me_all/)
|   ├── data_process.py   # Contains preprocess code of dataset
|   └── landmarks.zip     # landmarks of dataset
├── exps/             # Experiment configuration files (.yaml) for different models and settings
├── log/              # Directory for storing log files
├── models/           # Implementations of various incremental learning algorithms (DER, Foster, L2P, DualPrompt, RanPAC, Finetune)
├── utils/            # Utility scripts for data handling, model factories, network definitions, and general tools
├── main.py           # Main script to run experiments
├── trainer.py        # Contains the training and evaluation logic
├── .env              # Environment variables (e.g., dataset paths, model weights paths)
└── README.md         # This file
```

## Implemented Models

The project implements the following incremental learning strategies:

*   Finetune
*   DER
*   Foster
*   L2P
*   DualPrompt
*   RanPAC

## Supported Backbones

*   Vision Transformer (ViT-B/16)
*   ResNet (ResNet-152)
*   Swin Transformer (Swin-T)

## Dataset

### Datasets Used

This project utilizes a combination of several common micro-expression datasets for training and evaluation, specifically:

*   CASME II
*   SAMM
*   MMEW
*   CAS(ME)^3
*   DFME (optional 5th dataset/session)

These datasets are commonly used in Micro-Expression Recognition (MER) research. For detailed information about each dataset, including their collection methodology, annotations, and access policies, please refer to the original research papers associated with them. You will typically need to follow the guidelines provided by the dataset creators to request access.

The preprocessing script (`data_process.py`) is designed to work with these datasets to generate optical flow images used as input for the models.

### Dataset Preprocessing (`data_process.py`)

The `data_process.py` script is crucial for preparing the dataset. It performs the following steps:

1.  Reads coding files (e.g., `coding.csv`, and `coding.xlsx` for DFME) to get information about subjects, filenames, onset/apex frames, and labels.
2.  Loads the onset and apex frame images for each micro-expression sequence.
3.  Optionally crops the images based on facial landmarks (requires corresponding landmark files). The script assumes landmarks are stored in `.npy` files. Cropping is enabled/disabled via the `needs_crop` flag in the `DATASETS_CONFIG` dictionary within the script.
4.  Calculates the Farneback optical flow between the (potentially cropped) onset and apex frames.
5.  Saves the resulting optical flow image (as a BGR PNG file) into a structured output directory (`mix_me_all` by default). The output directory contains subfolders named `{dataset_name}_{label_foldername}`.

**DFME extra optical flows (optional):**

For DFME, `data_process.py` can additionally save 6 auxiliary flow images between the onset frame and frames around the apex:
`ApexFrame +/- {1,2,3}`. These are saved next to the main flow file using the suffix:
`_f<frame_index>.png` (e.g., `..._onset10_apex50_f49.png`).
You can control this via the environment variable `DFME_SAVE_EXTRA_FLOWS` (default: enabled).

### DFME integration details (5th dataset/session)

This repo treats DFME as the **5th incremental session** in iMER.

#### Expected DFME raw data layout

By default, the preprocessing script expects the following layout (you can override with env vars below):

```
<BASE_DATA_PATH>/
  dfme/
    train_data/
      coding.xlsx
      <SequenceName_1>/
        00000.png
        00001.png
        ...
      <SequenceName_2>/
        00000.png
        ...
```

Landmarks (if cropping is enabled) are expected at:

```
<BASE_LANDMARKS_PATH>/
  dfme/
    <SequenceName>.npy
```

#### DFME output (optical flow) layout and naming

After running preprocessing, DFME optical-flow images are saved into:

```
<OUTPUT_BASE_DIR>/
  dfme_anger/
  dfme_contempt/
  ...
```

Each DFME sample produces a **main** flow image:

* `..._onset<OnsetFrame>_apex<ApexFrame>.png`  (onset -> apex)

Optionally (default: enabled), the script also saves **6 auxiliary** flows:

* `..._f<t>.png` where `t` is `ApexFrame +/- {1,2,3}` (onset -> frame `t`)

During training, the dataset loader will:

* treat the main flow image as the actual sample, and
* **skip** `_f*.png` files so they do not become extra training samples.

If you enable *multi-flow loading* (see below), the loader will load the main
flow plus the auxiliary flows as a stack.

#### Environment variables for DFME

`dataset/data_process.py` supports these DFME-related variables:

* `DFME_BASE_DIR` (default: `<BASE_DATA_PATH>/dfme`)
* `DFME_FRAMES_ROOT` (default: `<DFME_BASE_DIR>/train_data`)
* `DFME_CODING_PATH` (default: `<DFME_FRAMES_ROOT>/coding.xlsx`)
* `DFME_SAVE_EXTRA_FLOWS` (default: `1`)

**Before running the script:**

1.  **Download Datasets:** Obtain the CASME II, SAMM, MMEW, and CAS(ME)$^3$ datasets.
2.  **Organize Datasets:** Place the datasets according to the expected structure. The script assumes a base data path (`BASE_DATA_PATH`) containing folders for each dataset (e.g., `casme2`, `samm`, `mmew`, `casme3`). Each dataset folder should contain its respective image sequences and a `coding.csv` file (or similar, adjust `coding_file` in the script if needed).
3.  **Prepare Landmarks (Optional but Recommended for Cropping):** If you intend to use cropping (`needs_crop: True` for a dataset), ensure you have the corresponding facial landmark files (`.npy`) organized under a base landmarks path (`BASE_LANDMARKS_PATH`). The exact expected path structure for landmarks varies per dataset and is defined by the `landmark_file_func` lambda functions in the script.
4.  **Configure Paths:** **Crucially**, modify the following paths at the beginning of `data_process.py` to match your environment:
    *   `BASE_DATA_PATH`: Root directory containing the raw dataset folders (casme2, samm, etc.).
    *   `BASE_LANDMARKS_PATH`: Root directory containing the landmark files.
    *   `OUTPUT_BASE_DIR`: Directory where the processed optical flow images (`mix_me_all`) will be saved.
    *   (DFME only) Optionally set:
        *   `DFME_BASE_DIR`, `DFME_FRAMES_ROOT`, `DFME_CODING_PATH` (via environment variables) if your DFME layout differs from the default.
5.  **Verify Configuration:** Double-check the `DATASETS_CONFIG` dictionary within `data_process.py`. Ensure the `coding_file` names, `ftype` (image extensions), `label_col` names, and especially the path construction logic in the lambda functions (`landmark_file_func`, `onset_path_func`, `apex_path_func`, `output_filename_func`) correctly match the structure and naming conventions of your downloaded datasets and landmarks. The comments marked with `!!!` highlight areas requiring careful verification.

**Running the script:**

```bash
python dataset/data_process.py
```

Recommended (use env vars instead of editing the script):

```bash
export BASE_DATA_PATH=/path/to/raw_datasets_root
export BASE_LANDMARKS_PATH=/path/to/landmarks_root
export OUTPUT_BASE_DIR=/path/to/output/mix_me_all

# DFME (optional)
export DFME_FRAMES_ROOT=/path/to/dfme/train_data
export DFME_CODING_PATH=/path/to/dfme/train_data/coding.xlsx
export DFME_SAVE_EXTRA_FLOWS=1

python dataset/data_process.py
```

The script will process each configured dataset and save the optical flow images to the specified `OUTPUT_BASE_DIR`. The final processed dataset directory (e.g., `dataset/mix_me_all`) should then be pointed to by the `DATASET_DIR` variable in your `.env` file or used as the default location (`./dataset/mix_me_all`).

Example `.env`:

```bash
# Path to the processed optical-flow folder (the folder that contains casme2_*, samm_*, ..., dfme_* subfolders)
DATASET_DIR=/path/to/output/mix_me_all
```

### Using DFME multi-flow inputs (optional)

By default, training uses **single flow images** (onset->apex) for all datasets. If you generated DFME auxiliary flows (the `_f*.png` files) and your model supports 5D inputs `[B, T, C, H, W]`, you can enable stacked DFME loading via:

*   add `use_dfme_multiflow: true` in your experiment YAML, or
*   pass `--use_dfme_multiflow true` via CLI (if you extend the argument parser).

When enabled, DFME samples will be loaded as a stack of flow images:
`[onset->apex, onset->(apex-3), ..., onset->(apex+3)]` (missing auxiliary flows fall back to the central one).

### iMER class count after enabling DFME

With the default iMER dataset order (CASME2, SAMM, MMEW, CASME3, DFME), the total
number of **fine-grained** classes is:

* `5 + 5 + 6 + 7 + 7 = 30`

So most configs set:

* `nb_classes: 30`

### Pretrained Weights

The models (especially ViT, Swin, ResNet backbones) rely on pretrained weights for initialization. You need to download these weights and make them accessible to the project.

1.  **Identify Required Weights:** The `.env` file specifies the expected paths for the weights:
    *   Swin Transformer: [`/mnt/model/I-MER/swinv2_tiny_patch4_window8_256.pth`](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth)
    *   Vision Transformer (ViT): [`/mnt/model/I-MER/vit_base_patch16_224.bin`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/blob/main/pytorch_model.bin)
    *   ResNet: [`/mnt/model/I-MER/resnet152.bin`](https://huggingface.co/timm/resnet152.a1_in1k/blob/main/pytorch_model.bin)

2.  **Configure Paths in `.env`:**
    *   Place the downloaded weight files in a location accessible by the project.
    *   **Update the absolute paths** (`SWIN_WEIGHTS_PATH`, `VIT_WEIGHTS_PATH`, `RESNET_WEIGHTS_PATH`) in the `.env` file to point to the exact locations where you saved the downloaded weights. Alternatively, if using the relative path structure, ensure the `weight/` directory exists and contains the necessary files relative to the project root.

## Configuration

Experiments are configured using YAML files located in the `exps/` directory. Each file defines parameters such as:

*   Model name (`model_name`)
*   Backbone type (`backbone_type`)
*   Dataset details (`dataset`, `nb_classes`, `img_size`)
*   Training parameters (epochs, learning rate, batch size, weight decay)
*   Memory settings for replay-based methods (`memory_size`, `memory_per_class`)
*   Prompting parameters for prompt-based methods
*   Cross-validation settings (`K`, `split_mode`)

## Usage

To run an experiment, use the `main.py` script and specify the configuration file:

```bash
python main.py --config ./exps/your_experiment_config.yaml --device 0 --log_file your_log_name --split_mode subject  # or k_fold
```

*   `--config`: Path to the desired YAML configuration file.
*   `--device`: Specify the GPU ID(s) to use (e.g., '0', '0,1').
*   `--log_file`: A suffix for the log file name.
*   `--split_mode`: Cross-validation strategy.
    *   `'subject'` (alias: `'SLCV'`) for subject-level split within each session
    *   `'k_fold'` (alias: `'ILCV'`) for k-fold split within each session
    *   `'session'` for a trivial split (train/test are the same session indices)

Results, including accuracy curves, will be logged and potentially saved to CSV files (e.g., `SLCVtune.csv`).

## 👍 Acknowledgements
We thank the following repos providing helpful components/functions in our work.

[LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)

[l2p-pytorch](https://github.com/JH-LEE-KR/l2p-pytorch)

[RanPAC](https://github.com/RanPAC/RanPAC/)

## Citation

If you find this benchmark useful in your research, please consider citing our paper:

```bibtex
@article{lai2025benchmark,
  title={A Benchmark for Incremental Micro-expression Recognition},
  author={Lai, Zhengqin and Hong, Xiaopeng and Wang, Yabin and Li, Xiaobai},
  journal={arXiv preprint arXiv:2501.19111},
  year={2025}
}
```
