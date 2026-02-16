import os


"""
=============== Define labels from food image dataset
"""

fruitveg81_labels = os.listdir("dataset/image/not_merged/fruitveg81/train")
food101_labels = os.listdir("dataset/image/not_merged/food101/train")
uecfood256_labels = os.listdir("dataset/image/not_merged/uecfood256/train")
afd_labels = os.listdir("dataset/image/not_merged/AFD/train")


def get_number_of_initial_classes():
    total_class = len(fruitveg81_labels) + len(food101_labels) + len(uecfood256_labels) + len(afd_labels)
    return {
        "total_class": total_class,
        "details": {
            "fruitveg81": len(fruitveg81_labels),
            "food101": len(food101_labels),
            "uecfood256": len(uecfood256_labels),
            "afd": len(afd_labels)
        }
    }
    

# stats = get_number_of_initial_classes()
# print(stats)
