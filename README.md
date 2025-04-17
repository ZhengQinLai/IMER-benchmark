# Incremental Learning Benchmark for Micro-Expression Recognition

## Overview

This project provides a framework for benchmarking various incremental learning algorithms specifically tailored for micro-expression recognition tasks. It includes implementations of several main incremental learning methods and supports different backbone architectures like Vision Transformers (ViT), ResNet, and Swin Transformer. The primary dataset used appears to be a combination of micro-expression datasets (CASME II, SAMM, MMEW, CASME III), referred to as iMER within the project.

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

*   Finetune (Baseline)
*   DER (Dynamic Expanding Representation)
*   Foster (Feature Boosting and Compression)
*   L2P (Learning to Prompt)
*   DualPrompt
*   RanPAC (Random Projection Incremental Classifier)

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

These datasets are commonly used in Micro-Expression Recognition (MER) research. For detailed information about each dataset, including their collection methodology, annotations, and access policies, please refer to the original research papers associated with them. You will typically need to follow the guidelines provided by the dataset creators to request access.

The preprocessing script (`data_process.py`) is designed to work with these specific datasets to generate optical flow images used as input for the models.

### Dataset Preprocessing (`data_process.py`)

The `data_process.py` script is crucial for preparing the dataset. It performs the following steps:

1.  Reads coding files (e.g., `coding.csv`) for each dataset (CASME II, SAMM, MMEW, CAS(ME)$^3$) to get information about subjects, filenames, onset/apex frames, and labels.
2.  Loads the onset and apex frame images for each micro-expression sequence.
3.  Optionally crops the images based on facial landmarks (requires corresponding landmark files). The script assumes landmarks are stored in `.npy` files. Cropping is enabled/disabled via the `needs_crop` flag in the `DATASETS_CONFIG` dictionary within the script.
4.  Calculates the Farneback optical flow between the (potentially cropped) onset and apex frames.
5.  Saves the resulting optical flow image (as a BGR PNG file) into a structured output directory (`mix_me_all` by default). The output directory contains subfolders named `{dataset_name}_{label_foldername}`.

**Before running the script:**

1.  **Download Datasets:** Obtain the CASME II, SAMM, MMEW, and CAS(ME)$^3$ datasets.
2.  **Organize Datasets:** Place the datasets according to the expected structure. The script assumes a base data path (`BASE_DATA_PATH`) containing folders for each dataset (e.g., `casme2`, `samm`, `mmew`, `casme3`). Each dataset folder should contain its respective image sequences and a `coding.csv` file (or similar, adjust `coding_file` in the script if needed).
3.  **Prepare Landmarks (Optional but Recommended for Cropping):** If you intend to use cropping (`needs_crop: True` for a dataset), ensure you have the corresponding facial landmark files (`.npy`) organized under a base landmarks path (`BASE_LANDMARKS_PATH`). The exact expected path structure for landmarks varies per dataset and is defined by the `landmark_file_func` lambda functions in the script.
4.  **Configure Paths:** **Crucially**, modify the following paths at the beginning of `data_process.py` to match your environment:
    *   `BASE_DATA_PATH`: Root directory containing the raw dataset folders (casme2, samm, etc.).
    *   `BASE_LANDMARKS_PATH`: Root directory containing the landmark files.
    *   `OUTPUT_BASE_DIR`: Directory where the processed optical flow images (`mix_me_all`) will be saved.
5.  **Verify Configuration:** Double-check the `DATASETS_CONFIG` dictionary within `data_process.py`. Ensure the `coding_file` names, `ftype` (image extensions), `label_col` names, and especially the path construction logic in the lambda functions (`landmark_file_func`, `onset_path_func`, `apex_path_func`, `output_filename_func`) correctly match the structure and naming conventions of your downloaded datasets and landmarks. The comments marked with `!!!` highlight areas requiring careful verification.

**Running the script:**

```bash
python data_process.py
```

The script will process each configured dataset and save the optical flow images to the specified `OUTPUT_BASE_DIR`. The final processed dataset directory (e.g., `dataset/mix_me_all`) should then be pointed to by the `DATASET_DIR` variable in your `.env` file or used as the default location (`./dataset/mix_me_all`).

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
python main.py --config ./exps/your_experiment_config.yaml --device 0 --log_file your_log_name --split_mode SLCV # or ILCV
```

*   `--config`: Path to the desired YAML configuration file.
*   `--device`: Specify the GPU ID(s) to use (e.g., '0', '0,1').
*   `--log_file`: A suffix for the log file name.
*   `--split_mode`: Cross-validation strategy ('SLCV' for Subject-Level, 'ILCV' for Instance-Level).

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
