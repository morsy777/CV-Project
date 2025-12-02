import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ==================== Dataset Loading ====================
dataset_dir = r"C:\Users\user\Downloads\Sec\archive\Train\Train" # path to dataset
categories = ["female", "male"]  # الفئات

def load_images(folder):
    X, y = [], []
    for idx, cat in enumerate(categories):
        cat_folder = os.path.join(folder, cat)
        for f in os.listdir(cat_folder):
            path = os.path.join(cat_folder, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))
            X.append(img)
            y.append(idx)
    return X, y

print("Loading dataset...")
X, y = load_images(dataset_dir)
X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} images.")

# ==================== Preprocessing ====================
def blur_image(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def morph_operations(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

# ==================== Feature Extraction ====================
orb = cv2.ORB_create(nfeatures=200)

def extract_features(img):
    img = blur_image(img)
    img = sharpen_image(img)
    img = morph_operations(img)
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((1, 32), dtype=np.float32)
    return des.mean(axis=0)

X_features = np.array([extract_features(img) for img in X])

# ==================== Train/Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# ==================== Train SVM ====================
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# ==================== Save Model ====================
joblib.dump(svm, "male_female_orb_svm.pkl")
print("Model saved as male_female_orb_svm.pkl")

# ==================== GUI ====================
current_image = None
svm_model = joblib.load("male_female_orb_svm.pkl")

def show_image(img, label):
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    label.config(image=imgtk)
    label.image = imgtk

def open_image():
    global current_image
    path = filedialog.askopenfilename()
    if not path:
        return
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    current_image = img_resized
    show_image(img_resized, main_image_label)

    vec = extract_features(img_resized).reshape(1, -1)
    pred = svm_model.predict(vec)[0]
    pred_prob = svm_model.predict_proba(vec)[0][pred]
    result_label.config(text=f"Prediction: {categories[pred].capitalize()} ({pred_prob*100:.2f}%)")

def show_canny():
    if current_image is None:
        return
    edges = cv2.Canny(current_image, 50, 150)
    show_image(edges, vis_label)

def show_harris():
    if current_image is None:
        return
    gray_f = np.float32(current_image)
    harris = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    img_harris = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
    img_harris[harris > 0.01 * harris.max()] = [255, 0, 0]
    show_image(img_harris, vis_label)

def show_shi():
    if current_image is None:
        return
    corners = cv2.goodFeaturesToTrack(current_image, 50, 0.01, 5)
    img_shi = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
    if corners is not None:
        corners = np.int32(corners)
        for c in corners:
            x, y = c.ravel()
            cv2.circle(img_shi, (x, y), 3, (0, 255, 0), -1)
    show_image(img_shi, vis_label)

def show_orb():
    if current_image is None:
        return
    kp, des = orb.detectAndCompute(current_image, None)
    img_orb = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
    img_orb = cv2.drawKeypoints(img_orb, kp, None, color=(0, 255, 0))
    show_image(img_orb, vis_label)

# ==================== Tkinter GUI ====================
root = tk.Tk()
root.title("Male vs Female Classifier")

# Buttons & Labels
tk.Button(root, text="Load Image", command=open_image, width=20).pack(pady=5)
main_image_label = tk.Label(root)
main_image_label.pack()
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=5)

tk.Button(root, text="Show Canny", command=show_canny, width=20).pack(pady=2)
tk.Button(root, text="Show Harris Corners", command=show_harris, width=20).pack(pady=2)
tk.Button(root, text="Show Shi-Tomasi", command=show_shi, width=20).pack(pady=2)
tk.Button(root, text="Show ORB Keypoints", command=show_orb, width=20).pack(pady=2)

vis_label = tk.Label(root)
vis_label.pack()

root.mainloop()
