import numpy as np
from torchvision import datasets, transforms
import os
import re

class iData:
    def __init__(self):
        self.train_trsf = []
        self.test_trsf = []
        self.common_trsf = []
        self.class_order = None

class iMER(iData):
    def __init__(self):
        super().__init__()
        self.dataset_dir = os.getenv("DATASET_DIR", './dataset/mix_me_all')
        self.dataset = ["casme2", "samm", "mmew", "casme3"]
        self.dataset_classes = {
            "casme2": ["disgust", "happiness", "others", "repression", "surprise"],
            "samm": ["anger", "contempt", "happiness", "others", "surprise"],
            "mmew": ["disgust", "happiness", "others", "sad", "surprise", "fear"],
            "casme3": ["anger", "disgust", "fear", "happiness", "others", "sad", "surprise"]
        }
        self.c2i = {}
        self.c2l_list = ["disgust", "happiness", "others", "repression", "surprise", "anger", "contempt", "sad", "fear"]
        self.c2l = {emotion: idx for idx, emotion in enumerate(self.c2l_list)}
        self.d2i = {dataset: idx for idx, dataset in enumerate(self.dataset)}
        self.i2l = {}
        self.incre = [5, 5, 6, 7]
        self._generate_mappings()

    def _generate_mappings(self):
        current_index = 0
        for dataset_name in self.dataset:
            if dataset_name in self.dataset_classes:
                for emotion in self.dataset_classes[dataset_name]:
                    key = f"{dataset_name}_{emotion}"
                    self.c2i[key] = current_index
                    if emotion in self.c2l:
                        self.i2l[current_index] = self.c2l[emotion]
                    current_index += 1
        a = 0

    def download_data(self):
        self.data, self.targets, self.SLCV, self.session = self._load_data(self.dataset_dir)

    def _load_data(self, directory):
        data = []
        targets = []
        SLCV = []
        session = []

        for label in sorted(os.listdir(directory)):
            class_path = os.path.join(directory, label)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        try:
                            targets.append(self.c2i[label])
                            session.append(self.d2i[label.split('_')[0]])
                            data.append(img_path)
                            SLCV.append(self._extract_first_number(img_file))
                        except KeyError:
                            pass

        return np.array(data), np.array(targets), np.array(SLCV), np.array(session)

    @staticmethod
    def _extract_first_number(s):
        match = re.search(r'\d+', s)
        return int(match.group(0)) if match else None

class iMER4UP(iData):
    def __init__(self):
        super().__init__()
        self.dataset_dir = os.getenv("DATASET_DIR", './dataset/mix_me_all')
        self.dataset = ["casme2"]  # Modify as needed
        self.dataset_classes = {
            "casme2": ["disgust", "happiness", "others", "repression", "surprise"],
            "samm": ["anger", "contempt", "happiness", "others", "surprise"],
            "mmew": ["disgust", "happiness", "others", "sad", "surprise", "fear"],
            "casme3": ["anger", "disgust", "fear", "happiness", "others", "sad", "surprise"]
        }
        self.all = []
        self._generate_all_classes()
        self.c2i = {}
        self.c2l_list = ["disgust", "happiness", "others", "repression", "surprise", "anger", "contempt", "sad", "fear"]
        self.c2l = {emotion: idx for idx, emotion in enumerate(self.c2l_list)}
        self.d2i = {dataset: 0 for dataset in self.dataset}
        self.i2l = {}
        self.incre = [len(self.dataset_classes["all"])]
        self._generate_mappings()

    def _generate_all_classes(self):
        for data in self.dataset:
            if data in self.dataset_classes:
                prefixed_classes = [f"{data}_{cls}" for cls in self.dataset_classes[data]]
                self.all.extend(prefixed_classes)
        self.dataset_classes["all"] = self.all

    def _generate_mappings(self):
        for index, full_emotion in enumerate(self.dataset_classes["all"]):
            dataset_name, emotion = full_emotion.split('_')
            self.c2i[full_emotion] = index
            if emotion in self.c2l:
                self.i2l[index] = self.c2l[emotion]

    def download_data(self):
        self.data, self.targets, self.SLCV, self.session = self._load_data(self.dataset_dir)

    def _load_data(self, directory):
        data = []
        targets = []
        SLCV = []
        session = []

        for label in sorted(os.listdir(directory)):
            class_path = os.path.join(directory, label)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        try:
                            targets.append(self.c2i[label])
                            session.append(self.d2i[label.split('_')[0]])
                            data.append(img_path)
                            SLCV.append(self._extract_first_number(img_file))
                        except KeyError:
                            pass

        return np.array(data), np.array(targets), np.array(SLCV), np.array(session)

    @staticmethod
    def _extract_first_number(s):
        match = re.search(r'\d+', s)
        return int(match.group(0)) if match else None
