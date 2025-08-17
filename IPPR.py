import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Set tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# State mapping dictionary
state_mapping = {
    "M": "Malacca",
    "B": "Selangor",
    "Z": "Military",
    "P": "Penang",
    "T": "Terengganu"
}

def preprocess_image(image):
    """Preprocess image for plate detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blur)
    thresh = cv2.adaptiveThreshold(equalized, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def detect_license_plate(image):
    """Detect plate regions from image"""
    preprocessed = preprocess_image(image)
    edges = cv2.Canny(preprocessed, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6 and 1000 < cv2.contourArea(cnt) < 15000:
            plate_candidates.append((x, y, w, h))
    return plate_candidates

def extract_text(image, plate_region):
    """Extract text from plate region using OCR"""
    x, y, w, h = plate_region
    roi = image[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary_roi, config='--psm 7')
    return text.strip()

def identify_state(plate_text):
    """Identify state by first character"""
    if len(plate_text) > 0:
        return state_mapping.get(plate_text[0].upper(), "Unknown State")
    return "Unknown State"

# ---------------- GUI ---------------- #
def upload_and_detect():
    global panel
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    
    image = cv2.imread(file_path)
    candidates = detect_license_plate(image)

    result_text = "No plate detected"
    state = "Unknown"
    
    for (x, y, w, h) in candidates:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        plate_text = extract_text(image, (x, y, w, h))
        state = identify_state(plate_text)
        result_text = f"Plate: {plate_text} | State: {state}"
        cv2.putText(image, state, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        break  # Take first candidate for now

    # Convert for Tkinter display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.config(image=imgtk)
    panel.image = imgtk

    result_label.config(text=result_text)


# Tkinter Window
root = tk.Tk()
root.title("License Plate Recognition (LPR + SIS)")

btn = Button(root, text="Upload Image", command=upload_and_detect)
btn.pack(pady=10)

panel = Label(root)
panel.pack()

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
