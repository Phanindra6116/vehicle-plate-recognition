import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd
import os

# ✅ Initialize OCR Reader with custom model storage
reader = easyocr.Reader(['en'], model_storage_directory="models", user_network_directory="models")

# Page settings
st.set_page_config(page_title="License Plate Recognition", page_icon="🚗", layout="centered")

# Sidebar: About
st.sidebar.title("ℹ️ About this Project")
st.sidebar.write("""
This is a *License Plate Recognition App* built with:
- *Python, OpenCV, EasyOCR, Pandas, Streamlit*  
- Upload a car image, and the system will automatically:
  - Detect the license plate  
  - Extract the text using OCR  
  - Save results into a CSV file  

📌 Potential Use Cases:  
- Automated parking systems  
- Toll booths  
- Traffic surveillance  
- Smart city applications  
""")
st.sidebar.markdown("👨‍💻 Created by *venkata phanindra*")

# Main Title
st.title("🚗 License Plate Recognition App")
st.write("Upload a car image and the system will detect and read the license plate number.")

# File upload
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

RESULTS_FILE = "results.csv"

def detect_and_recognize_plate(image):
    """Detects and extracts license plate text from a car image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:  # rectangle shape
            location = approx
            break

    plate_text, confidence = "", 0.0
    cropped = None
    if location is not None:
        # Crop plate region
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        (x1, y1), (x2, y2) = (np.min(x), np.min(y)), (np.max(x), np.max(y))
        cropped = gray[x1:x2+1, y1:y2+1]

        # OCR on cropped plate
        ocr_result = reader.readtext(cropped)
        if ocr_result:
            plate_text = ocr_result[0][1]
            confidence = round(ocr_result[0][2] * 100, 2)

    # ⚡ Fallback: run OCR on the whole image if no plate found
    if not plate_text:
        ocr_result = reader.readtext(image)
        if ocr_result:
            plate_text = ocr_result[0][1]
            confidence = round(ocr_result[0][2] * 100, 2)

    return plate_text, confidence, cropped


def save_results(filename, plate_number, confidence):
    """Save detected plates to a CSV file."""
    new_data = pd.DataFrame([[filename, plate_number, confidence]], 
                            columns=["Image", "Plate Number", "Confidence (%)"])

    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE)
        updated = pd.concat([existing, new_data], ignore_index=True)
        updated.to_csv(RESULTS_FILE, index=False)
    else:
        new_data.to_csv(RESULTS_FILE, index=False)


# Process uploaded file
if uploaded_file is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="📸 Uploaded Car Image", use_container_width=True)

    with st.spinner("🔍 Detecting license plate..."):
        plate_text, confidence, cropped = detect_and_recognize_plate(image)

    if plate_text:
        st.success(f"✅ Detected Plate Number: *{plate_text}* (Confidence: {confidence}%)")

        if cropped is not None:
            st.image(cropped, caption="🔎 Detected License Plate Region", use_container_width=True)

        # Save result to CSV
        save_results(uploaded_file.name, plate_text, confidence)
        st.info("📁 Result saved to results.csv")

        # Download button
        with open(RESULTS_FILE, "rb") as f:
            st.download_button("⬇️ Download results.csv", f, file_name="results.csv", mime="text/csv")
    else:
        st.error("⚠️ No license plate detected. Try another image.")