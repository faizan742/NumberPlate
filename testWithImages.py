import cv2
import easyocr
from ultralytics import YOLO
import os
import re

# Load YOLO model (replace with your best.pt path)
model = YOLO("training_output/detect/train/weights/best.pt")

# Load image
image_path = "Test/images/Number_plate_set1_108.jpg"
image = cv2.imread(image_path)

# Run detection
results = model(image_path)[0]
print("🔍 Detecting number plates...")

# Setup EasyOCR
reader = easyocr.Reader(['en'])
plate_counter = 0
os.makedirs("debug", exist_ok=True)

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    plate_img = image[y1:y2, x1:x2]

    if plate_img.size == 0:
        print(f"⚠️ Plate {plate_counter}: Empty region skipped.")
        continue

    # Resize for better OCR
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale and denoise
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR using EasyOCR
    result = reader.readtext(thresh, detail=0)
    print(f"🧪 EasyOCR Output: {result}")

    # Join all detected parts
    raw_text = ' '.join(result).strip()
    cleaned = re.sub(r'[^A-Z0-9 ]', '', raw_text.upper())
    print(f"🧼 Cleaned OCR: {cleaned}")

    match = re.search(r'([A-Z]{2,4})\s?([0-9]{3,5})?', cleaned)
    plate_number = f"{match.group(1)} {match.group(2)}" if match and match.group(2) else match.group(1) if match else ''

    if plate_number:
        print(f"✅ Plate {plate_counter}: {plate_number}")
    else:
        print(f"❌ Plate {plate_counter}: Could not match valid pattern.")

    plate_counter += 1

# cv2.destroyAllWindows()
