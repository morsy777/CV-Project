import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==================== Dataset Path ====================
dataset_dir = r"C:\Users\user\Downloads\Sec\archive\Train\Train"
categories = ["female", "male"]

# ==================== Dataset Loading ====================
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
print('Start Feature Extraction')
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
print('End Feature Extraction')

# ==================== Train/Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

# ==================== Train SVM ====================
print('Start Training')
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# ==================== Save Model ====================
joblib.dump(svm, "male_female_orb_svm.pkl")
print("Model saved as male_female_orb_svm.pkl")
