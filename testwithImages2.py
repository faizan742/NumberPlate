import cv2
import pytesseract
from ultralytics import YOLO
import re
import os

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Load YOLOv8 model (custom trained for number plates)
model = YOLO("training_output/detect/train/weights/best.pt")

# Input image
image_path = "Test/images/Number_plate_set1_108.jpg"
image = cv2.imread(image_path)

# Run detection
results = model(image_path)[0]
print("🔍 Detecting number plates...")

# Create debug output folder
os.makedirs("debug", exist_ok=True)
plate_counter = 0

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    plate_img = image[y1:y2, x1:x2]

    if plate_img.size == 0:
        print(f"⚠️ Plate {plate_counter}: Empty region skipped.")
        continue

    # Resize to enhance OCR accuracy
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Preprocess (grayscale + denoising)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Threshold version
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug_path = f"debug/plate_{plate_counter}.png"
    cv2.imwrite(debug_path, thresh)
    print(f"🖼️ Debug saved: {debug_path}")

    # OCR both grayscale and threshold version
    ocr_config = '--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text_thresh = pytesseract.image_to_string(thresh, config=ocr_config)
    text_gray = pytesseract.image_to_string(gray, config=ocr_config)

    # Choose better result
    raw_text = text_thresh if len(text_thresh.strip()) >= len(text_gray.strip()) else text_gray
    print(f"🧪 RAW OCR Output: {raw_text!r}")

    # Clean text
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    joined = ' '.join(lines)
    cleaned = re.sub(r'[^A-Z0-9 ]', '', joined)
    print(f"🧼 Cleaned OCR: {cleaned}")

    # Extract pattern like 'LEC 5855 14' or similar
    parts = re.findall(r'[A-Z]{2,4}|\d{2,5}', cleaned)
    plate_number = ' '.join(parts) if parts else ''

    if plate_number:
        print(f"✅ Plate {plate_counter}: {plate_number}")
    else:
        print(f"❌ Plate {plate_counter}: Could not extract full number.")

    plate_counter += 1

cv2.destroyAllWindows()
