import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ==================== Categories ====================
categories = ["female", "male"]

# ==================== Load Pretrained Model ====================
svm_model = joblib.load("male_female_orb_svm.pkl")

# ==================== ORB Feature Extractor ====================
orb = cv2.ORB_create(nfeatures=100)

def extract_features(img):
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((1, 32), dtype=np.float32)
    return des.mean(axis=0)

# ==================== GUI Helper ====================
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

# ==================== Image Operations ====================
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

    small_img = cv2.resize(original_image_full, (64, 64))
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

def show_canny():
    if current_image is None:
        return
    edges = cv2.Canny(current_image, 50, 150)
    show_image(edges)

def show_harris():
    if current_image is None:
        return
    gray_f = np.float32(current_image)
    harris = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    img = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
    img[harris > 0.01 * harris.max()] = [255, 0, 0]
    show_image(img)

def show_shi():
    if current_image is None:
        return
    corners = cv2.goodFeaturesToTrack(current_image, 50, 0.01, 5)
    img = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        for c in np.int32(corners):
            x, y = c.ravel()
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    show_image(img)

def show_orb():
    if current_image is None:
        return
    kp, des = orb.detectAndCompute(current_image, None)
    img = cv2.drawKeypoints(
        cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR),
        kp,
        None,
        color=(0, 255, 0),
    )
    show_image(img)

# ==================== Tkinter GUI ====================
root = tk.Tk()
root.title("Male vs Female Classifier")
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
add_btn("Canny", show_canny)
add_btn("Harris", show_harris)
add_btn("Shi-Tomasi", show_shi)
add_btn("ORB", show_orb)

root.mainloop()
