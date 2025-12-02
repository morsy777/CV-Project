import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ======== Dataset Path ========
dataset_dir = r"C:\Users\user\Downloads\Sec\archive\Train\Train"
categories = ["female", "male"]

# ======== Load Images ========
def load_images(folder):
    X, y = [], []
    for idx, cat in enumerate(categories):
        cat_folder = os.path.join(folder, cat)
        for f in os.listdir(cat_folder):
            path = os.path.join(cat_folder, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 128))  # HOG aspect ratio
            X.append(img)
            y.append(idx)
    return X, y

print("Loading dataset...")
X, y = load_images(dataset_dir)
X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} images.")

# ======== HOG Feature Extraction using OpenCV ========
print("Start Feature Extraction (HOG with OpenCV)...")

win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

hog_cv = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

X_features = []
for i, img in enumerate(X):
    fd = hog_cv.compute(img)
    X_features.append(fd.flatten())
    if i % 500 == 0:
        print(f"Processed {i}/{len(X)} images")
X_features = np.array(X_features)
print("End Feature Extraction")

# ======== Train/Test Split ========
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

# ======== Train SVM ========
print("Start Training SVM...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# ======== Save Model ========
joblib.dump(svm, "male_female_hog_svm_opencv.pkl")
print("Model saved as male_female_hog_svm_opencv.pkl")
