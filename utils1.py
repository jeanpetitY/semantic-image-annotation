import os
import numpy as np

def analyze_dataset(root_dir):
    """
    Analyse un dataset structuré avec train/test:
    root_dir/
        train/
            class_1/
            class_2/
        test/
            class_1/
            class_2/

    Retourne statistiques sur le nombre d'images par classe.
    """

    class_counts = {}

    # Parcourir train et test
    for split in ["train", "test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            # Compter uniquement les fichiers images
            image_files = [
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
            ]

            # Ajouter au compteur total (train + test)
            class_counts[class_name] = class_counts.get(class_name, 0) + len(image_files)

    if not class_counts:
        print("No classes found.")
        return

    counts = list(class_counts.values())

    # Statistiques
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)
    mean_images = np.mean(counts)
    std_images = np.std(counts)

    # Affichage
    print("===== Dataset Statistics =====")
    print(f"Total classes: {len(class_counts)}\n")
    print(f"Class with MIN images: {min_class} ({class_counts[min_class]} images)")
    print(f"Class with MAX images: {max_class} ({class_counts[max_class]} images)\n")
    print(f"Average images per class: {mean_images:.2f}")
    print(f"Standard deviation: {std_images:.2f}")

    return {
        "min_class": (min_class, class_counts[min_class]),
        "max_class": (max_class, class_counts[max_class]),
        "mean": mean_images,
        "std": std_images
    }


# ==========================
# UTILISATION
# ==========================

if __name__ == "__main__":
    dataset_path = "dataset/old/AFD"
    analyze_dataset(dataset_path)