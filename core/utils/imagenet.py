import os
import numpy as np

class ImageNetHelper:

    def __init__(self, config):
        self.config = config
        self.label_mappings = self.build_label_mappings()
        self.valid_blacklist = self.build_valid_blacklist()

    def build_label_mappings(self):
        label_mappings = {}
        # Parse map_clsloc.txt file in devkit
        rows = open(self.config.WORD_IDS).read().strip().split("\n")
        for row in rows:
            id, label, word = row.split(" ")
            label_mappings[id] = int(label) - 1
        return label_mappings

    def build_valid_blacklist(self):
        # Parse ILSVRC2015_clslos_validation_blacklist.txt file in devkit
        rows = open(self.config.VALID_BLACKLIST).read()
        rows = set(rows.strip().split("\n"))
        return rows

    def build_training_set(self):
        img_paths, labels = [], []
        rows = open(self.config.TRAIN_LIST).read().strip().split("\n")
        for row in rows:
            partial_path, _ = row.strip().split(" ")
            id = partial_path.split("/")[0]
            label = self.label_mappings[id]
            img_path = os.path.sep.join([self.config.IMAGES_PATH, "train", f"{partial_path}.JPEG"])
            img_paths.append(img_path)
            labels.append(label)
        return np.array(img_paths), np.array(labels)

    def build_validation_set(self):
        img_paths, labels = [], []
        rows = open(self.config.VALID_LIST).read().strip().split("\n")
        valid_labels = open(self.config.VALID_LABELS).read().strip().split("\n")
        for row, label in zip(rows, valid_labels):
            partial_path, image_id = row.strip().split(" ")
            if image_id in self.valid_blacklist:
                continue
            img_path = os.path.sep.join([self.config.IMAGES_PATH, "val", f"{partial_path}.JPEG"])
            img_paths.append(img_path)
            labels.append(int(label) - 1)
        return np.array(img_paths), np.array(labels)
