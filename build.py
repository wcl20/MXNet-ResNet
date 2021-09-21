import cv2
import json
import numpy as np
import os
import tqdm
from config import config
from core.utils import ImageNetHelper
from sklearn.model_selection import train_test_split


def main():

    imagenet_helper = ImageNetHelper(config)

    print("[INFO] Loading image paths ...")
    train_paths, train_labels = imagenet_helper.build_training_set()
    valid_paths, valid_labels = imagenet_helper.build_validation_set()

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        train_paths, train_labels,
        test_size=config.NUM_TEST_IMAGES,
        stratify=train_labels,
        random_state=42
    )

    datasets = [
        ("train", train_paths, train_labels, config.TRAIN_MX_LIST),
        ("valid", valid_paths, valid_labels, config.VALID_MX_LIST),
        ("test", test_paths, test_labels, config.TEST_MX_LIST)
    ]

    # Dataset mean
    R, G, B = [], [], []

    for type, paths, labels, output_path in datasets:
        print(f"[INFO] Building {output_path} ...")
        file = open(output_path, "w")
        for i, (path, label) in tqdm.tqdm(enumerate(zip(paths, labels))):
            row = "\t".join([str(i), str(label), path])
            file.write(f"{row}\n")
            if type == "train":
                image = cv2.imread(path)
                b, g, r = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)
        file.close()

    os.makedirs("output", exist_ok=True)
    file = open(config.MEAN_PATH, "w")
    file.write(json.dumps({ "R": np.mean(R), "G": np.mean(G), "B": np.mean(B) }))
    file.close()


if __name__ == '__main__':
    main()
