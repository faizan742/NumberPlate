import os

# Paths to the labels and images directories
LABELS_DIR = "/home/zia/PycharmProjects/numberPlateDetection/Train/labels"  # Directory containing label files
IMAGES_DIR = "/home/zia/PycharmProjects/numberPlateDetection/Train/images"  # Directory containing image files

def find_images_without_labels():
    """Find images that do not have a corresponding label file."""
    images_without_labels = []

    for image_file in os.listdir(IMAGES_DIR):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Process image files
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(LABELS_DIR, label_file)

            if not os.path.exists(label_path):
                images_without_labels.append(image_file)

    return images_without_labels

def main():
    images_without_labels = find_images_without_labels()

    if images_without_labels:
        print("Images without corresponding label files:")
        for image in images_without_labels:
            print(image)
    else:
        print("All images have corresponding label files.")

if __name__ == "__main__":
    main()
