"""Balance image classes by downsampling or augmentation.

Paper reference:
- Experimental setup where partitions are balanced toward target class counts
  after the leakage-safe split.
"""

import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps


class DatasetBalancer:
    """
    Handle class imbalance by downsampling or data augmentation.
    """

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply a random augmentation to an image.
        """
        choice = random.choice(["rotate", "flip", "mirror", "color", "contrast"])

        if choice == "rotate":
            return image.rotate(random.randint(-25, 25))
        if choice == "flip":
            return ImageOps.flip(image)
        if choice == "mirror":
            return ImageOps.mirror(image)
        if choice == "color":
            return ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        if choice == "contrast":
            return ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.3))

        return image

    def balance_dataset(
        self,
        source_path: str,
        output_path: str,
        target: int = 1000,
        extensions=(".jpg", ".jpeg", ".png"),
    ):
        """Balance dataset by downsampling or augmentation.

        Args:
            source_path (str): Path to input dataset.
            output_path (str): Path to balanced output dataset.
            target (int): Target number of images per class.(750 for train and 250 for test)
        """
        # Paper setup: balancing happens after splitting so the reported class
        # counts are reached without introducing cross-partition leakage.
        os.makedirs(output_path, exist_ok=True)

        for class_name in os.listdir(source_path):
            class_path = os.path.join(source_path, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(extensions)
            ]

            if not images:
                continue

            output_class_dir = os.path.join(output_path, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            if len(images) > target:
                selected = random.sample(images, target)
                for img in selected:
                    shutil.copy(
                        os.path.join(class_path, img),
                        os.path.join(output_class_dir, img),
                    )

            elif len(images) < target:
                for img in images:
                    shutil.copy(
                        os.path.join(class_path, img),
                        os.path.join(output_class_dir, img),
                    )

                needed = target - len(images)
                opened_images = [
                    Image.open(os.path.join(class_path, img)) for img in images
                ]

                for i in range(needed):
                    base_img = random.choice(opened_images)
                    augmented = self.augment_image(base_img)

                    # JPEG does not support RGBA
                    if augmented.mode != "RGB":
                        augmented = augmented.convert("RGB")

                    augmented.save(
                        os.path.join(
                            output_class_dir,
                            f"aug_{i}_{random.randint(1000,9999)}.jpg",
                        ),
                        format="JPEG",
                    )


            else:
                for img in images:
                    shutil.copy(
                        os.path.join(class_path, img),
                        os.path.join(output_class_dir, img),
                    )
