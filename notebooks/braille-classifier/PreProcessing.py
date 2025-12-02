import cv2
import numpy as np
from zipfile import ZipFile
import os
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(
        self,
        dt_path: str,
        dt_name: str,
        extraction_path: str = "/content/",
        formatted_path: str = "/content/formatted_braille_dataset",
        verbose: bool = True,
        train_test_split_ratio: float = 0.7,
    ) -> None:
        self.dt_path = dt_path
        self.dt_name = dt_name
        self.extraction_path = extraction_path
        self.formatted_path = formatted_path
        self.verbose = verbose
        self.train_test_split_ratio = train_test_split_ratio

    def unzip_folder(self):
        with ZipFile(self.dt_path, "r") as zip_object:
            zip_object.extractall(path=self.extraction_path)
        if self.verbose:
            print(f"Extracted dataset to: {self.extraction_path}")

    def encode_label(self, label):
        return ord(label.lower()) - ord("a")

    def read_and_resize(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img

    def add_gaussian_noise(self, img, mean=0, var=10):
        sigma = var**0.5
        noise = np.random.normal(mean, sigma, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img

    def add_threshold(self, img, block_size=15, const=5):
        thresholded_img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, const
        )
        return thresholded_img

    def augment_image(self, img):
        if np.random.rand() < 0.3:
            img = self.add_gaussian_noise(img, var=np.random.randint(5, 20))
        if np.random.rand() < 0.3:
            img - self.add_threshold(img)
        return img

    def pipeline(self):
        unformatted_path = os.path.join(self.extraction_path, self.dt_name)
        formatted_train = os.path.join(self.formatted_path, "train")
        formatted_test = os.path.join(self.formatted_path, "test")
        formatted_val = os.path.join(self.formatted_path, "val")
        os.makedirs(formatted_train, exist_ok=True)
        os.makedirs(formatted_test, exist_ok=True)
        os.makedirs(formatted_val, exist_ok=True)

        X_train = []
        X_val = []
        X_test = []

        y_train = []
        y_val = []
        y_test = []

        for entry in os.scandir(unformatted_path):
            if entry.is_dir():
                label = os.path.basename(entry.path)
                if self.verbose:
                    print(f"Processing class: {label}")

                img_paths = [
                    os.path.join(entry.path, f)
                    for f in os.listdir(entry.path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]

                # split before transformation to avoid leakage
                val_train_size = (self.train_test_split_ratio) + (
                    1 - self.train_test_split_ratio
                ) / 2  # for 0.7 + 0.15 = 0.85
                train_val, test = train_test_split(
                    img_paths, train_size=val_train_size, random_state=42, shuffle=True
                )
                limit = int(
                    len(train_val) * ((self.train_test_split_ratio) / val_train_size)
                )
                train = train_val[:limit]
                val = train_val[limit:]

                os.makedirs(os.path.join(formatted_train, label), exist_ok=True)
                os.makedirs(os.path.join(formatted_test, label), exist_ok=True)
                os.makedirs(os.path.join(formatted_val, label), exist_ok=True)

                encoded_label = self.encode_label(label)

                for path in train:
                    img = self.read_and_resize(path)
                    aug_img = self.augment_image(img)

                    img = np.expand_dims(img, axis=-1)
                    aug_img = np.expand_dims(aug_img, axis=-1)

                    base = os.path.basename(path)
                    X_train.append(img)  # original resized img
                    y_train.append(encoded_label)
                    X_train.append(aug_img)  # transformed img
                    y_train.append(encoded_label)
                    cv2.imwrite(os.path.join(formatted_train, label, base), img)
                    cv2.imwrite(
                        os.path.join(formatted_train, label, f"aug_{base}"), aug_img
                    )

                for path in val:
                    img = self.read_and_resize(path)
                    img = np.expand_dims(img, axis=-1)
                    base = os.path.basename(path)
                    X_val.append(img)
                    y_val.append(encoded_label)
                    cv2.imwrite(os.path.join(formatted_val, label, base), img)

                for path in test:
                    img = self.read_and_resize(path)
                    img = np.expand_dims(img, axis=-1)
                    base = os.path.basename(path)
                    X_test.append(img)
                    y_test.append(encoded_label)
                    cv2.imwrite(os.path.join(formatted_test, label, base), img)

                if self.verbose:
                    print(
                        f"""{len(X_train)} train +
                        validation {len(X_val)} +
                        {len(y_test)} test images processed for '{label}'"""
                    )

        print(len(X_train))
        print(len(y_train))
        print(len(X_val))
        print(len(y_val))
        print(len(X_test))
        print(len(y_test))

        return (
            np.array(X_train),
            np.array(y_train),
            np.array(X_val),
            np.array(y_val),
            np.array(X_test),
            np.array(y_test),
        )
