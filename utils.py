import os
import numpy as np

def analyze_dataset(root_dir):
    """
    Analyze a dataset structured as:
    root_dir/
        class_1/
            img1.jpg
            img2.jpg
        class_2/
            img1.jpg
            ...

    Returns statistics about image distribution per class.
    """

    class_counts = {}

    # Parcours des sous-dossiers (classes)
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # Compter uniquement les fichiers images
        image_files = [
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
        ]

        class_counts[class_name] = len(image_files)

    if not class_counts:
        print("No classes found.")
        return

    # Conversion en liste
    counts = list(class_counts.values())

    # Statistiques
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)

    mean_images = np.mean(counts)
    std_images = np.std(counts)

    # Résultats
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
    dataset_path = "dataset/old/UECFOOD256"
    analyze_dataset(dataset_path)