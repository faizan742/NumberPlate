import os

# Paths to the labels and images directories
LABELS_DIR = "/home/zia/PycharmProjects/numberPlateDetection/Train/labels"  # Directory containing label files
IMAGES_DIR = "/home/zia/PycharmProjects/numberPlateDetection/Train/images"  # Directory containing image files

def is_valid_yolo_line(line):
    """Validate if a line follows the YOLO format."""
    parts = line.split()
    if len(parts) != 5:
        return False  # YOLO format requires exactly 5 parts

    try:
        class_id = int(parts[0])  # Class ID must be an integer
        coordinates = list(map(float, parts[1:]))  # Remaining parts must be floats
        return True
    except ValueError:
        return False

def process_label_file(file_path):
    """Check if a label file is empty or invalid, and delete it and its corresponding image if necessary."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Check if the file is empty or contains invalid lines
        if not lines or any(not is_valid_yolo_line(line) for line in lines):
            print(f"Deleting invalid or empty label file: {file_path}")
            os.remove(file_path)

            # Delete the corresponding image file
            image_file = os.path.splitext(os.path.basename(file_path))[0] + ".jpg"  # Assuming images are .jpg
            image_path = os.path.join(IMAGES_DIR, image_file)
            if os.path.exists(image_path):
                print(f"Deleting corresponding image file: {image_path}")
                os.remove(image_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Process each label file in the directory
    for label_file in os.listdir(LABELS_DIR):
        if label_file.endswith('.txt'):  # Ensure it's a label file
            file_path = os.path.join(LABELS_DIR, label_file)
            process_label_file(file_path)

    print("Label and image file validation completed.")

if __name__ == "__main__":
    main()
