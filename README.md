<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Practical Machine Learning and Image Processing — Chapter 5 Exercises</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --bg:#0f172a;        /* slate-900 */
      --panel:#111827;     /* gray-900 */
      --muted:#94a3b8;     /* slate-400 */
      --text:#e5e7eb;      /* gray-200 */
      --accent:#60a5fa;    /* blue-400 */
      --green:#34d399;     /* emerald-400 */
      --orange:#fb923c;    /* orange-400 */
      --purple:#c084fc;    /* purple-400 */
      --border:#1f2937;    /* gray-800 */
      --code:#0b1220;      /* deep panel */
      --chip:#1e293b;      /* slate-800 */
    }
    html, body { background: var(--bg); color: var(--text); font-family: -apple-system, Segoe UI, Roboto, Inter, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif; margin: 0; padding: 0; line-height: 1.6; }
    .container { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }
    h1, h2, h3 { line-height: 1.2; margin: 0 0 10px; }
    h1 { font-size: 28px; letter-spacing: 0.2px; }
    h2 { font-size: 22px; color: var(--accent); margin-top: 28px; }
    h3 { font-size: 18px; margin-top: 22px; }
    p { color: var(--text); margin: 8px 0 14px; }
    .muted { color: var(--muted); }
    .panel {
      background: linear-gradient(180deg, rgba(17,24,39,0.9), rgba(11,18,32,0.9));
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 18px;
      margin: 16px 0 22px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .grid {
      display: grid; gap: 14px;
    }
    @media (min-width: 720px) {
      .grid.two { grid-template-columns: 1fr 1fr; }
    }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 13px;
    }
    pre {
      background: var(--code);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      overflow: auto;
      margin: 12px 0 18px;
      color: #dbeafe;
    }
    .kbd {
      background: var(--chip);
      border: 1px solid var(--border);
      border-bottom-color: #0b1220;
      padding: 2px 8px;
      border-radius: 8px;
      font-size: 12px;
      color: var(--text);
      display: inline-block;
    }
    ul, ol { margin: 8px 0 16px 18px; }
    li { margin: 4px 0; }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--border);
      background: var(--chip);
      color: var(--text);
      margin: 6px 8px 0 0;
    }
    .note {
      border-left: 3px solid var(--purple);
      padding: 10px 12px;
      background: rgba(192, 132, 252, 0.06);
      border-radius: 8px;
      margin: 12px 0 18px;
    }
    a { color: var(--green); text-decoration: none; }
    a:hover { text-decoration: underline; }
    hr {
      border: 0; border-top: 1px solid var(--border);
      margin: 26px 0;
    }
    .titlebar {
      display: flex; align-items: center; gap: 10px; margin-bottom: 14px;
    }
    .dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--green); box-shadow: 0 0 8px rgba(52, 211, 153, .6);
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="titlebar">
      <div class="dot"></div>
      <h1>Practical Machine Learning and Image Processing — Chapter 5 Exercises</h1>
    </div>
    <p class="muted">This repository contains five self-contained Jupyter notebooks implementing core topics from H. Singh’s “Practical Machine Learning and Image Processing” (Chapter 5). Each notebook is structured for clarity, reproducibility, and grading.</p>

    <h2>📁 Repository Structure</h2>
    <div class="panel">
      <pre><code>notebooks/
  01_sift_feature_mapping.ipynb
  02_image_registration_ransac.ipynb
  03_classification_ann.ipynb
  04_classification_cnn.ipynb
  05_classification_traditional_ml.ipynb
data/                 # put your images here (for SIFT/RANSAC)
results/              # saved outputs (optional)
README.md</code></pre>
    </div>

    <h2>🚀 Notebooks Overview</h2>

    <h3>1) 01_sift_feature_mapping.ipynb</h3>
    <ul>
      <li>SIFT keypoints, descriptors, BF/FLANN matching, Lowe’s ratio test</li>
      <li>Inputs: two stored images from <span class="kbd">data/</span></li>
      <li>Outputs: keypoint/match visualizations, filtered matches</li>
    </ul>

    <h3>2) 02_image_registration_ransac.ipynb</h3>
    <ul>
      <li>Robust homography estimation via RANSAC, inlier detection, warping/alignment</li>
      <li>Inputs: an image pair from <span class="kbd">data/</span></li>
      <li>Outputs: inlier overlay, warped alignment result</li>
    </ul>

    <h3>3) 03_classification_ann.ipynb (MLP)</h3>
    <ul>
      <li>Vectorization, normalization, dense layers, early stopping</li>
      <li>Dataset: MNIST (with SSL fallback options)</li>
      <li>Outputs: accuracy, confusion matrix, classification report</li>
    </ul>

    <h3>4) 04_classification_cnn.ipynb</h3>
    <ul>
      <li>Conv2D + Pooling + Dropout + Softmax, training curves, evaluation</li>
      <li>Dataset: MNIST (with SSL fallback options)</li>
      <li>Outputs: training curves, confusion matrix, per-class metrics</li>
    </ul>

    <h3>5) 05_classification_traditional_ml.ipynb</h3>
    <ul>
      <li>HOG feature extraction, PCA, classifiers: SVM / KNN / RandomForest</li>
      <li>Dataset: MNIST (with SSL fallback options)</li>
      <li>Outputs: model comparison (accuracy, train/pred time), per-class accuracy, disagreement analysis</li>
    </ul>

    <hr />

    <h2>🧩 Requirements</h2>
    <ul>
      <li>Python 3.9+</li>
      <li>Jupyter Notebook or JupyterLab</li>
    </ul>
    <p>Install (CPU-only example):</p>
    <pre><code>pip install numpy matplotlib seaborn scikit-learn scikit-image opencv-contrib-python tensorflow certifi</code></pre>
    <p>Launch:</p>
    <pre><code>jupyter lab
# or
jupyter notebook</code></pre>

    <hr />

    <h2>🔐 MNIST Dataset &amp; SSL Fix</h2>
    <p>If you see SSL errors when using <span class="kbd">keras.datasets.mnist</span>:</p>

    <p><strong>Preferred (macOS):</strong></p>
    <pre><code>/Applications/Python\ 3.xx/Install\ Certificates.command</code></pre>

    <p><strong>Temporary bypass at top of notebook:</strong></p>
    <pre><code>import ssl, os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'</code></pre>

    <p><strong>Manual MNIST download:</strong></p>
    <ul>
      <li>Download: <a href="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz" target="_blank" rel="noopener">https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz</a></li>
      <li>Save to: <span class="kbd">~/.keras/datasets/mnist.npz</span></li>
    </ul>

    <hr />

    <h2>📚 How to Run</h2>

    <h3>SIFT (Notebook 1)</h3>
    <ol>
      <li>Place two images in <span class="kbd">data/</span> (e.g., <span class="kbd">data/img1.jpg</span>, <span class="kbd">data/img2.jpg</span>)</li>
      <li>Update file paths in the first cell</li>
      <li>Run all cells to visualize keypoints and matches</li>
    </ol>

    <h3>RANSAC (Notebook 2)</h3>
    <ol>
      <li>Use the same or another pair in <span class="kbd">data/</span></li>
      <li>Run all cells to estimate homography and warp</li>
      <li>Inspect inliers and alignment result</li>
    </ol>

    <h3>ANN (Notebook 3), CNN (Notebook 4), Traditional ML (Notebook 5)</h3>
    <ol>
      <li>Run dataset loading cell (MNIST) with SSL fallbacks</li>
      <li>Train, evaluate, and inspect metrics/plots</li>
    </ol>

    <hr />

    <h2>🧠 Learning Outcomes</h2>
    <div class="panel grid two">
      <ul>
        <li><span class="badge">SIFT</span> Robust local features and matching</li>
        <li><span class="badge">RANSAC</span> Outlier-robust model estimation (e.g., homography)</li>
      </ul>
      <ul>
        <li><span class="badge">ANN vs. CNN</span> Trade-offs for image tasks</li>
        <li><span class="badge">Traditional ML</span> HOG + PCA + classic classifiers</li>
      </ul>
    </div>

    <h2>🔁 Reproducibility</h2>
    <ul>
      <li>Each notebook imports dependencies at the top</li>
      <li>Random seeds set where applicable</li>
      <li>Clear sections: setup → data → preprocessing → model → training/eval → visuals → summary</li>
      <li>Some notebooks save artifacts to <span class="kbd">results/</span></li>
    </ul>

    <h2>🛠 Common Issues</h2>

    <p><strong>OpenCV SIFT unavailable:</strong></p>
    <pre><code>pip install opencv-contrib-python
# then use: cv2.SIFT_create()</code></pre>

    <p><strong>TensorFlow install trouble (CPU):</strong></p>
    <pre><code>pip install tensorflow
# for GPU, follow TensorFlow’s official guide for your platform</code></pre>

    <p><strong>SSL errors (MNIST):</strong> See “MNIST Dataset &amp; SSL Fix” above.</p>

    <hr />

    <h2>✅ Grading Readiness</h2>
    <ul>
      <li>Modular, commented code with structured explanations</li>
      <li>Key outputs visualized</li>
      <li>Self-contained notebooks</li>
      <li>Test outputs can be committed for review</li>
    </ul>

    <h2>📄 License</h2>
    <p>Educational use for coursework and demonstrations. Verify licenses for datasets/libraries used.</p>

    <h2>🙏 Acknowledgements</h2>
    <ul>
      <li>H. Singh’s “Practical Machine Learning and Image Processing”</li>
      <li>Libraries: NumPy, Scikit-Image, OpenCV, Scikit-Learn, TensorFlow/Keras, Matplotlib, Seaborn</li>
    </ul>