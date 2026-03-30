import logging
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold


from utils.data import iMER, iMER4UP
class IncrementalDataloaderGenerator:
    def __init__(self, batch_size=16, shuffle=True, img_size=224, use_dfme_multiflow=False):
        self.iData = iMER()
        self.iData.download_data()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = img_size
        # Whether to use DFME multi-flow loading (onset↔apex±n frames)
        self.use_dfme_multiflow = use_dfme_multiflow

    def get_dataset(self, indices, extra_data=None):
        data, targets = self._filter_data(indices)

        if extra_data is not None:
            data, targets = self._add_extra_data(data, targets, extra_data)

        dataset = self._create_dataset(data, targets)
        return data, targets, dataset

    def get_dataloader(self, train_indices, test_indices, extra_data=None):
        """
        Returns train and test DataLoaders based on the provided indices and optional extra data.
        """
        train_data, train_targets = self._filter_data(train_indices)
        test_data, test_targets = self._filter_data(test_indices)

        if extra_data is not None:
            train_data, train_targets = self._add_extra_data(train_data, train_targets, extra_data)

        train_dataset = self._create_dataset(train_data, train_targets)
        test_dataset = self._create_dataset(test_data, test_targets)

        if self.use_dfme_multiflow:
            collate_fn = _collate_multi_flow
        else:
            collate_fn = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=32,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=32,
            collate_fn=collate_fn,
        )

        return train_loader, test_loader

    def _filter_data(self, indices):
        """
        Filters data and targets from iData based on the given indices.
        """
        data = self.iData.data[indices]
        targets = self.iData.targets[indices]
        return data, targets

    def _add_extra_data(self, data, targets, extra_data):
        """
        Adds extra data to the existing dataset.
        """
        extra_data_samples, extra_targets = extra_data
        data = np.concatenate([data, extra_data_samples], axis=0)
        targets = np.concatenate([targets, extra_targets], axis=0)
        return data, targets

    def _create_dataset(self, data, targets):
        """
        Creates a Dataset object from the given data and targets.
        """
        return CustomDataset(
            self.iData,
            data,
            targets,
            self.img_size,
            use_dfme_multiflow=self.use_dfme_multiflow,
        )


class IncrementalIndexGenerator:
    def __init__(self, split_flag="session", up=False, k=5, subjects_per_fold=5):
        self.iData = iMER()
        self.iData.download_data()
        self.split_flag = split_flag
        self.up = up
        self.k = k
        self.subjects_per_fold = subjects_per_fold
        self.split_indices = self._generate_splits()


    def _generate_splits(self):
        """
        Precomputes all splits based on the provided split flag.
        """
        splits = []
        sessions = np.unique(self.iData.session)

        for current_session in sessions:
            session_indices = np.where(self.iData.session == current_session)[0]
            session_splits = []

            # Backward-compatible aliases:
            #   - "ILCV"   -> k-fold split
            #   - "SLCV"   -> subject-based split
            split_flag = self.split_flag
            if split_flag == "ILCV":
                split_flag = "k_fold"
            elif split_flag == "SLCV":
                split_flag = "subject"

            if split_flag == "k_fold":
                kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
                for fold_train, fold_test in kf.split(session_indices):
                    session_splits.append((session_indices[fold_train], session_indices[fold_test]))

            elif split_flag == "subject":
                session_subjects = getattr(self.iData, "subject", self.iData.SLCV)[session_indices]
                unique_subjects = np.unique(session_subjects)
                folds = [unique_subjects[i::self.subjects_per_fold] for i in range(self.subjects_per_fold)]

                for test_subjects in folds:
                    fold_test_indices = session_indices[np.isin(session_subjects, test_subjects)]
                    fold_train_indices = session_indices[~np.isin(session_subjects, test_subjects)]
                    session_splits.append((fold_train_indices, fold_test_indices))

            else:  # Default to session-based split
                session_splits.append((session_indices, session_indices))

            splits.append(session_splits)

        return splits

    def get_split(self, session_index, fold_index=0):
        """
        Retrieves precomputed indices for a specific session and fold.
        Combines training data from the specified fold and relevant test data.
        """
        current_session_splits = self.split_indices[session_index]
        train_fold_indices, test_fold_indices = current_session_splits[fold_index]

        # Collect test indices from all test folds of previous sessions
        test_indices = list(test_fold_indices)
        for past_session in range(session_index):
            _, past_test_fold = self.split_indices[past_session][fold_index]
            test_indices.extend(past_test_fold)
        if self.up:
            train_indices = list(train_fold_indices)
            for past_session in range(session_index):
                past_train_fold, _ = self.split_indices[past_session][fold_index]
                train_indices.extend(past_train_fold)
            return np.unique(train_indices), np.unique(test_indices)

        return train_fold_indices, np.unique(test_indices)


def _collate_multi_flow(batch):
    """
    Custom collate function that supports variable-length multi-flow inputs.

    Each sample's `image` can be:
      - [C, H, W]        (single flow)
      - [T, C, H, W]     (multi-flow)

    We:
      1) ensure all become [T, C, H, W] (T>=1),
      2) pad along T to the maximum T in this batch,
      3) stack into [B, Tmax, C, H, W].
    """
    images, targets = zip(*batch)

    processed = []
    max_T = 1
    for img in images:
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [1, C, H, W]
        t = img.size(0)
        max_T = max(max_T, t)
        processed.append(img)

    padded = []
    for img in processed:
        t, c, h, w = img.shape
        if t < max_T:
            pad = torch.zeros(max_T - t, c, h, w, dtype=img.dtype)
            img = torch.cat([img, pad], dim=0)
        padded.append(img)

    batch_images = torch.stack(padded, dim=0)  # [B, Tmax, C, H, W]
    batch_targets = torch.as_tensor(targets, dtype=torch.long)
    return batch_images, batch_targets


class CustomDataset(Dataset):
    def __init__(self, iData, data, targets, img_size, use_dfme_multiflow=False):
        self.data = data
        self.targets = targets
        self.use_dfme_multiflow = use_dfme_multiflow

        transform = [
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
        self.transform = transforms.Compose(transform)        
        self.i2l = iData.i2l
        self.dataset = iData.dataset
        self.dataset_classes = iData.dataset_classes
        self.d2i = iData.d2i
        self.c2l = iData.c2l
        self.incre = iData.incre

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        # Determine dataset name from folder (e.g., "dfme_anger" -> "dfme")
        class_dir = os.path.basename(os.path.dirname(path))
        dataset_name = class_dir.split('_')[0] if '_' in class_dir else class_dir

        # For DFME (when enabled), load the central flow and its auxiliary flows
        # around the apex (pre/post n frames), as a stack of images.
        if self.use_dfme_multiflow and dataset_name == "dfme":
            import re

            base_root, ext = os.path.splitext(path)

            # Parse apex frame index from filename: *_onsetXX_apexYY.png
            fname = os.path.basename(base_root)
            m = re.search(r"_apex(\d+)$", fname)
            apex_frame = int(m.group(1)) if m else None

            img_paths = [path]  # central onset→apex flow

            if apex_frame is not None:
                # Neighbour offsets around apex: [-3, -2, -1, +1, +2, +3]
                neighbor_offsets = [-3, -2, -1, 1, 2, 3]
                for off in neighbor_offsets:
                    t = apex_frame + off
                    if t < 0:
                        # Use central flow as fallback for invalid indices
                        img_paths.append(path)
                        continue
                    aux_path = f"{base_root}_f{int(t)}{ext}"
                    if os.path.exists(aux_path):
                        img_paths.append(aux_path)
                    else:
                        # If auxiliary flow is missing, fall back to central flow
                        img_paths.append(path)

            imgs = [self.transform(pil_loader(p)) for p in img_paths]
            # Shape: [num_flows, C, H, W]
            image = torch.stack(imgs, dim=0)
        else:
            image = self.transform(pil_loader(path))

        return image, self.targets[idx]

def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

# Example Usage
if __name__ == "__main__":
    mix_me = iMER()
    mix_me.download_data()

    # NOTE: split_flag supports: "k_fold"/"ILCV", "subject"/"SLCV", or "session".
    index_gen = IncrementalIndexGenerator(split_flag="subject")
    dataloader_gen = IncrementalDataloaderGenerator()

    print("Precomputed splits:")
    for session_idx in range(len(index_gen.split_indices)):
        for fold_idx in range(len(index_gen.split_indices[session_idx])):
            train_idx, test_idx = index_gen.get_split(session_idx, fold_idx)
            train_loader, test_loader = dataloader_gen.get_dataloader(train_idx, test_idx)
            print(f"Session {session_idx}, Fold {fold_idx}: Train size = {len(train_loader.dataset)}, Test size = {len(test_loader.dataset)}")
