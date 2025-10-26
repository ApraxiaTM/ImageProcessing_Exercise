# Practical Machine Learning and Image Processing — Chapter 5 Exercises

This repository contains five self-contained Jupyter notebooks implementing core topics from H. Singh’s “Practical Machine Learning and Image Processing” (Chapter 5). Each notebook is structured for clarity, reproducibility, and grading.

## 📁 Repository Structure



notebooks/ 01_sift_feature_mapping.ipynb 02_image_registration_ransac.ipynb 03_classification_ann.ipynb 04_classification_cnn.ipynb 05_classification_traditional_ml.ipynb data/ # put your images here (for SIFT/RANSAC) results/ # saved outputs (optional) README.md


## 🚀 Notebooks Overview

### 1) 01_sift_feature_mapping.ipynb
- SIFT keypoints, descriptors, BF/FLANN matching, Lowe’s ratio test
- Inputs: two stored images from `data/`
- Outputs: keypoint/match visualizations, filtered matches

### 2) 02_image_registration_ransac.ipynb
- Robust homography estimation via RANSAC, inlier detection, warping/alignment
- Inputs: an image pair from `data/`
- Outputs: inlier overlay, warped alignment result

### 3) 03_classification_ann.ipynb (MLP)
- Vectorization, normalization, dense layers, early stopping
- Dataset: MNIST (with SSL fallback options)
- Outputs: accuracy, confusion matrix, classification report

### 4) 04_classification_cnn.ipynb
- Conv2D + Pooling + Dropout + Softmax, training curves, evaluation
- Dataset: MNIST (with SSL fallback options)
- Outputs: training curves, confusion matrix, per-class metrics

### 5) 05_classification_traditional_ml.ipynb
- HOG feature extraction, PCA, classifiers: SVM/KNN/RandomForest
- Dataset: MNIST (with SSL fallback options)
- Outputs: model comparison (accuracy, train/pred time), per-class accuracy, disagreement analysis

---

## 🧩 Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab

Install (CPU-only example):


pip install numpy matplotlib seaborn scikit-learn scikit-image opencv-contrib-python tensorflow certifi


Launch:


jupyter lab

or

jupyter notebook


---

## 🔐 MNIST Dataset & SSL Fix

If you see SSL errors when using `keras.datasets.mnist`:

Preferred (macOS):


/Applications/Python\ 3.xx/Install\ Certificates.command


Temporary bypass at top of notebook:
```python
import ssl, os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
```

Manual MNIST download:

Download: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
Save to: ~/.keras/datasets/mnist.npz
📚 How to Run

SIFT (Notebook 1)

Place two images in data/ (e.g., data/img1.jpg, data/img2.jpg)
Update file paths in the first cell
Run all cells to visualize keypoints and matches

RANSAC (Notebook 2)

Use the same or another pair in data/
Run all cells to estimate homography and warp
Inspect inliers and alignment result

ANN (Notebook 3), CNN (Notebook 4), Traditional ML (Notebook 5)

Run dataset loading cell (MNIST) with SSL fallbacks
Train, evaluate, and inspect metrics/plots
🧠 Learning Outcomes
SIFT: Robust local features and matching
RANSAC: Outlier-robust model estimation (e.g., homography)
ANN vs. CNN: Trade-offs for image tasks
Traditional ML: Feature engineering (HOG) + PCA + classic classifiers
🔁 Reproducibility
Each notebook imports dependencies at the top
Random seeds set where applicable
Clear sections: setup → data → preprocessing → model → training/eval → visuals → summary
Some notebooks save artifacts to results/
🛠 Common Issues

OpenCV SIFT unavailable:
pip install opencv-contrib-python
then use: cv2.SIFT_create()

TensorFlow install trouble (CPU):
pip install tensorflow
# for GPU, follow TensorFlow’s official guide for your platform


SSL errors (MNIST): See “MNIST Dataset & SSL Fix” above

✅ Grading Readiness
- Modular, commented code with structured explanations
- Key outputs visualized
- Self-contained notebooks
- Test outputs can be committed for review

📄 License
- Educational use for coursework and demonstrations. Verify licenses for datasets/libraries used.

🙏 Acknowledgements
H. Singh’s “Practical Machine Learning and Image Processing”
Libraries: NumPy, Scikit-Image, OpenCV, Scikit-Learn, TensorFlow/Keras, Matplotlib, Seaborn
