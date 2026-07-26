import os
import random
import shutil
from math import floor


class DatasetSplitter:
    """
    Utility class for creating train/test splits from an image dataset.
    Assumes a folder-per-class structure.
    """

    def create_train_test_split(
        self,
        dataset_path: str,
        train_path: str,
        test_path: str,
        test_split: float = 0.25,
        extensions=(".jpg", ".jpeg", ".png"),
    ):
        """
        Create a train/test split from a dataset.

        Args:
            dataset_path (str): Root dataset directory.
            train_path (str): Output train directory.
            test_path (str): Output test directory.
            test_split (float): Ratio of test images.
            extensions (tuple): Allowed image extensions.
        """
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(extensions)
            ]

            if not images:
                continue

            random.shuffle(images)

            test_count = floor(len(images) * test_split)
            test_images = images[:test_count]
            train_images = images[test_count:]

            os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

            for img in train_images:
                shutil.copy(
                    os.path.join(class_path, img),
                    os.path.join(train_path, class_name, img),
                )

            for img in test_images:
                shutil.copy(
                    os.path.join(class_path, img),
                    os.path.join(test_path, class_name, img),
                )
