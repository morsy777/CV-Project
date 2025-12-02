import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ======== Categories ========
categories = ["female", "male"]

# ======== Load Pretrained Model ========
svm_model = joblib.load("male_female_hog_svm_opencv.pkl")

# ======== HOG Feature Extractor using OpenCV ========
win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
hog_cv = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

def extract_features(img):
    fd = hog_cv.compute(img)
    return fd.flatten()

# ======== GUI Helper ========
def show_image(img, resize=True):
    img_to_show = img.copy()
    if resize:
        img_to_show = cv2.resize(img_to_show, (400, 400))
    if img_to_show.ndim == 2:
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)

    im_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor="nw", image=imgtk)

# ======== Image Operations ========
current_image = None
original_image_full = None

def open_image():
    global current_image, original_image_full

    path = filedialog.askopenfilename()
    if not path:
        return

    original_image_full = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    current_image = cv2.resize(original_image_full, (400, 400))
    show_image(current_image)

    small_img = cv2.resize(original_image_full, (64, 128))
    vec = extract_features(small_img).reshape(1, -1)

    pred = svm_model.predict(vec)[0]
    prob = svm_model.predict_proba(vec)[0][pred]

    result_label.config(
        text=f"Prediction: {categories[pred].capitalize()} ({prob*100:.2f}%)"
    )

def show_original():
    if original_image_full is None:
        return
    show_image(original_image_full, resize=False)

# ======== Tkinter GUI ========
root = tk.Tk()
root.title("Male vs Female Classifier (HOG OpenCV)")
root.configure(bg="#1B2631")

canvas = tk.Canvas(root, width=400, height=400, bg="#34495E", highlightthickness=0)
canvas.pack(pady=10)

result_label = tk.Label(
    root, text="Prediction: ", font=("Arial", 16), fg="#ECF0F1", bg="#1B2631"
)
result_label.pack(pady=5)

btn_frame = tk.Frame(root, bg="#1B2631")
btn_frame.pack(pady=10)

btn_color = "#0E4D92"
btn_font = ("Arial", 12, "bold")

def add_btn(txt, cmd):
    tk.Button(btn_frame, text=txt, command=cmd, bg=btn_color, fg="white",
              font=btn_font, padx=10, pady=5).pack(side="left", padx=5)

add_btn("Load Image", open_image)
add_btn("Original", show_original)

root.mainloop()
