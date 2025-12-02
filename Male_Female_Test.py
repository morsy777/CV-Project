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
dataset_dir = r"C:\Users\user\Downloads\Sec\archive\Train\Train"
categories = ["female", "male"]

def load_images(folder, limit=None):
    X, y = [], []
    for idx, cat in enumerate(categories):
        cat_folder = os.path.join(folder, cat)
        files = os.listdir(cat_folder)
        if limit:
            files = files[:limit]  # سريع للتجربة
        for f in files:
            path = os.path.join(cat_folder, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            X.append(img)
            y.append(idx)
    return X, y

print("Loading dataset...")
X, y = load_images(dataset_dir, limit=100)
X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} images.")

# ==================== Feature Extraction ====================
orb = cv2.ORB_create(nfeatures=100)

def extract_features(img):
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
joblib.dump(svm, "male_female_orb_svm_fast.pkl")
print("Model saved as male_female_orb_svm_fast.pkl")

# ==================== GUI ====================
current_image = None
original_image_full = None  # الصورة الأصلية كاملة الحجم
svm_model = joblib.load("male_female_orb_svm_fast.pkl")

# ==================== Helper Function ====================
def show_image(img, resize=True):
    img_to_show = img.copy()
    if resize:
        img_to_show = cv2.resize(img_to_show, (400, 400))
    if img_to_show.ndim == 2:
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_GRAY2RGB)
    elif img_to_show.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
    elif img_to_show.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = img_to_show

    im_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor='nw', image=imgtk)

# ==================== Image Operations ====================
def open_image():
    global current_image, original_image_full
    path = filedialog.askopenfilename()
    if not path: return
    # الصورة الأصلية كاملة الحجم بدون أي تعديل
    original_image_full = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # نسخة مصغرة للعرض على canvas
    current_image = cv2.resize(original_image_full, (400, 400))
    show_image(current_image)

    # نسخة صغيرة جداً للتصنيف
    small_img = cv2.resize(original_image_full, (64, 64))
    vec = extract_features(small_img).reshape(1, -1)
    pred = svm_model.predict(vec)[0]
    pred_prob = svm_model.predict_proba(vec)[0][pred]
    result_label.config(text=f"Prediction: {categories[pred].capitalize()} ({pred_prob*100:.2f}%)")

def show_original():
    if original_image_full is None: return
    # عرض الصورة الأصلية كما هي بدون أي تعديل
    show_image(original_image_full, resize=False)

def show_canny():
    if current_image is None: return
    edges = cv2.Canny(current_image, 50, 150)
    show_image(edges)

def show_harris():
    if current_image is None: return
    gray_f = np.float32(current_image)
    harris = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    img_harris = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
    img_harris[harris > 0.01 * harris.max()] = [255, 0, 0]
    show_image(img_harris)

def show_shi():
    if current_image is None: return
    corners = cv2.goodFeaturesToTrack(current_image, 50, 0.01, 5)
    img_shi = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        corners = np.int32(corners)
        for c in corners:
            x, y = c.ravel()
            cv2.circle(img_shi, (x, y), 3, (0, 255, 0), -1)
    show_image(img_shi)

def show_orb():
    if current_image is None: return
    kp, des = orb.detectAndCompute(current_image, None)
    img_orb = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
    img_orb = cv2.drawKeypoints(img_orb, kp, None, color=(0, 255, 0))
    show_image(img_orb)

# ==================== Tkinter GUI ====================
root = tk.Tk()
root.title("Male vs Female Classifier")
root.configure(bg="#1B2631")  # خلفية داكنة هادئة

# Canvas كبير لعرض الصور
canvas = tk.Canvas(root, width=400, height=400, bg="#34495E", highlightthickness=0)
canvas.pack(pady=10)

# Label للنتيجة
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 16), fg="#ECF0F1", bg="#1B2631")
result_label.pack(pady=5)

# Frame للزرار الأفقي
button_frame = tk.Frame(root, bg="#1B2631")
button_frame.pack(pady=10)

btn_color = "#0E4D92"  # كحلي داكن
btn_font = ("Arial", 12, "bold")
btn_padx = 10
btn_pady = 5

tk.Button(button_frame, text="Load Image", command=open_image, bg=btn_color, fg="white", font=btn_font, padx=btn_padx, pady=btn_pady).pack(side='left', padx=5)
tk.Button(button_frame, text="Original", command=show_original, bg=btn_color, fg="white", font=btn_font, padx=btn_padx, pady=btn_pady).pack(side='left', padx=5)
tk.Button(button_frame, text="Canny", command=show_canny, bg=btn_color, fg="white", font=btn_font, padx=btn_padx, pady=btn_pady).pack(side='left', padx=5)
tk.Button(button_frame, text="Harris", command=show_harris, bg=btn_color, fg="white", font=btn_font, padx=btn_padx, pady=btn_pady).pack(side='left', padx=5)
tk.Button(button_frame, text="Shi-Tomasi", command=show_shi, bg=btn_color, fg="white", font=btn_font, padx=btn_padx, pady=btn_pady).pack(side='left', padx=5)
tk.Button(button_frame, text="ORB", command=show_orb, bg=btn_color, fg="white", font=btn_font, padx=btn_padx, pady=btn_pady).pack(side='left', padx=5)

root.mainloop()