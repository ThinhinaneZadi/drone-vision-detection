from pathlib import Path
from PIL import Image


# This script converts VisDrone annotation files into YOLO label files.
#
# VisDrone gives bounding boxes like this:
# x_left, y_top, box_width, box_height, score, category, truncation, occlusion
#
# YOLO needs labels like this:
# class_id x_center_normalized y_center_normalized width_normalized height_normalized
#
# So the main job of this script is to:
# 1. Read each VisDrone annotation file
# 2. Find the matching image size
# 3. Convert each bounding box into YOLO format
# 4. Save the new labels in a labels/ folder


def convert_box_to_yolo(x_left, y_top, box_width, box_height, image_width, image_height):
    """
    Convert one bounding box from VisDrone format to YOLO format.
    """

    # VisDrone gives us the top-left corner of the box.
    # YOLO wants the center of the box.
    x_center = x_left + box_width / 2
    y_center = y_top + box_height / 2

    # YOLO wants values between 0 and 1.
    # So we divide by the full image width and height.
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = box_width / image_width
    height_norm = box_height / image_height

    return x_center_norm, y_center_norm, width_norm, height_norm


def convert_split(split_path):
    """
    Convert one dataset split, for example train or val.
    """

    # Convert the path text into a Path object.
    # This makes file and folder handling easier.
    split_path = Path(split_path)

    # VisDrone already has images/ and annotations/ folders.
    images_dir = split_path / "images"
    annotations_dir = split_path / "annotations"

    # We will create a new labels/ folder for YOLO label files.
    labels_dir = split_path / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Get all annotation files from the annotations folder.
    annotation_files = sorted(annotations_dir.glob("*.txt"))

    print(f"Converting split: {split_path.name}")
    print(f"Number of annotation files: {len(annotation_files)}")

    # Go through each annotation file one by one.
    for annotation_file in annotation_files:

        # Each annotation file should match one image with the same name.
        # Example:
        # annotation: 000001.txt
        # image:      000001.jpg
        image_file = images_dir / (annotation_file.stem + ".jpg")

        # The YOLO label file will have the same file name as the annotation.
        label_file = labels_dir / annotation_file.name

        # If the image is missing, skip this file and show a warning.
        if not image_file.exists():
            print(f"Warning: image not found for {annotation_file.name}")
            continue

        # Open the image only to get its width and height.
        # We need the image size because YOLO labels must be normalized.
        with Image.open(image_file) as img:
            image_width, image_height = img.size

        # This list will store all YOLO label lines for this image.
        yolo_lines = []

        # Read the VisDrone annotation file.
        with open(annotation_file, "r") as file:
            for line in file:

                # Remove extra spaces or new lines.
                line = line.strip()

                # Skip empty lines.
                if not line:
                    continue

                # VisDrone values are separated by commas.
                values = line.split(",")

                # A valid VisDrone line should have at least 8 values.
                if len(values) < 8:
                    continue

                # Read the VisDrone annotation values.
                x_left = float(values[0])
                y_top = float(values[1])
                box_width = float(values[2])
                box_height = float(values[3])
                score = int(values[4])
                category = int(values[5])

                # VisDrone category meaning:
                # 0  = ignored region
                # 1  = pedestrian
                # 2  = people
                # 3  = bicycle
                # 4  = car
                # 5  = van
                # 6  = truck
                # 7  = tricycle
                # 8  = awning-tricycle
                # 9  = bus
                # 10 = motor
                # 11 = others

                # If score is 0, this is usually an ignored region.
                # We do not train YOLO on ignored regions.
                if score == 0:
                    continue

                # Skip category 0 because it is an ignored region.
                # Skip category 11 because "others" is too vague.
                if category == 0 or category == 11:
                    continue

                # Skip invalid boxes.
                # Width and height must be positive numbers.
                if box_width <= 0 or box_height <= 0:
                    continue

                # YOLO class IDs start at 0.
                # VisDrone useful classes start at 1.
                # So we subtract 1.
                #
                # Example:
                # VisDrone category 4 = car
                # YOLO class ID becomes 3
                class_id = category - 1

                # Convert the box coordinates into YOLO format.
                x_center_norm, y_center_norm, width_norm, height_norm = convert_box_to_yolo(
                    x_left,
                    y_top,
                    box_width,
                    box_height,
                    image_width,
                    image_height,
                )

                # Make sure the normalized values are valid.
                # YOLO expects center values between 0 and 1.
                if not (0 <= x_center_norm <= 1):
                    continue

                if not (0 <= y_center_norm <= 1):
                    continue

                # YOLO expects width and height to be greater than 0 and at most 1.
                if not (0 < width_norm <= 1):
                    continue

                if not (0 < height_norm <= 1):
                    continue

                # Create one YOLO label line.
                # Format:
                # class_id x_center y_center width height
                yolo_line = (
                    f"{class_id} "
                    f"{x_center_norm:.6f} "
                    f"{y_center_norm:.6f} "
                    f"{width_norm:.6f} "
                    f"{height_norm:.6f}"
                )

                # Add this object label to the list for this image.
                yolo_lines.append(yolo_line)

        # Save all YOLO labels for this image.
        # If an image has no valid objects, the file will be empty.
        with open(label_file, "w") as file:
            file.write("\n".join(yolo_lines))

    print(f"Finished converting: {split_path.name}")
    print(f"YOLO labels saved in: {labels_dir}")


def main():
    # These are the dataset folders we downloaded.
    train_path = "data/VisDrone-DET/VisDrone2019-DET-train"
    val_path = "data/VisDrone-DET/VisDrone2019-DET-val"

    # Convert the training split.
    convert_split(train_path)

    # Convert the validation split.
    convert_split(val_path)

    print("All VisDrone annotations were converted to YOLO format.")


# This makes sure the script runs only when we execute this file directly.
if __name__ == "__main__":
    main()
