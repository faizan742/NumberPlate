Number Plate Detection and OCR System
Overview
This document summarizes the implementation of a number plate detection and recognition system using the YOLOv8 object detection model and OCR techniques. The entire system was tested and run on a Python backend, with model training on Kaggle and deployment in a Docker container locally.

Workflow Breakdown
1. Model Training (YOLOv8)
    • Platform Used: Kaggle
    • Library: Ultralytics YOLOv8
    • Purpose: Train a custom object detection model to localize number plates in vehicle images.
    • Export: The trained weights were exported as a .pt file (e.g., best.pt) from Kaggle to local disk for further use.

2. Local Setup for Inference
    • Development Environment: Python 3.10
    • Virtual Environment: .venv activated
    • Library used for object detection: ultralytics
    • OCR libraries tried:
        ◦ pytesseract (Tesseract OCR engine)
        ◦ easyocr (final working solution due to better accuracy)
Key Python Libraries:
    • ultralytics: For YOLOv8 object detection
    • cv2 (OpenCV): For image processing (cropping, grayscaling, filtering)
    • pytesseract: For OCR using Tesseract engine
    • easyocr: Lightweight and more robust OCR engine
    • re: For regular expression matching and cleanup of OCR results
    • os: For filesystem operations (saving debug crops)

System Flow
    1. Load Image: Input image is read using OpenCV.
    2. Detect Number Plate: The YOLOv8 model identifies bounding boxes around number plates.
    3. Crop the Number Plate: The cropped section is processed using OpenCV filters (grayscale, threshold).
    4. OCR Processing:
        ◦ Initially tried with pytesseract, but OCR results were inconsistent or missing.
        ◦ Switched to easyocr which successfully recognized plate numbers.
    5. Post-processing:
        ◦ Cleaned OCR results (remove unwanted characters, spaces).
        ◦ Regex used to validate patterns (e.g., letters followed by digits).
    6. Output:
        ◦ Plate numbers printed in terminal.
        ◦ Debug image crops optionally saved for inspection.

Docker Consideration
When running in a Docker container:
    • GUI functions like cv2.imshow() or cv2.destroyAllWindows() will fail unless additional GUI support is installed.
    • This was handled by removing GUI-related lines to avoid crashes.
    • The final system was terminal-based, outputting OCR results via print().

Final Working Code Highlights
import easyocr
reader = easyocr.Reader(['en'])
...
ocr_results = reader.readtext(cropped_plate_img, detail=0)
Sample Output:
🧪 EasyOCR Output: ['LEC', '5855']
識 Cleaned OCR: LEC 5855
✅ Plate 0: LEC 5855

Summary
    • YOLOv8 effectively detects number plates.
    • EasyOCR provided the most accurate OCR results.
    • GUI-free setup was essential for Docker compatibility.
    • Final result accurately extracted real-world number plate: LEC 5855
Let me know if you want this system packaged, Dockerized, or extended further.
