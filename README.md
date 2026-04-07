# AI Pictionary 🎨

An authentic, real-time sketching game powered by **Convolutional Neural Networks (CNNs)**. This project was developed as part of the **Microsoft AI Unlocked Hackathon**, where it reached the **Top 50**. It uses a custom-trained model on the Google QuickDraw dataset to recognize hand-drawn doodles instantly in the browser.

---

## 🚀 Key Features

* **Real-time Inference:** Guesses what you are drawing as you move your mouse/finger.
* **High Accuracy:** Achieves **90%+ validation accuracy** across 50+ distinct classes.
* **Memory Optimized:** Designed to train on over **1.3 million images** using an `uint8` data pipeline to fit within 16GB of system RAM.
* **Smart Preprocessing:** Implements **proportional scaling and bounding box cropping** to ensure the AI "sees" your drawing exactly like the training data.

---

## 💻 Developer Setup

### 1. Environment Setup
The project is optimized for **Ubuntu** and systems with **NVIDIA GPUs** (tested on RTX 4060). It is highly recommended to use a virtual environment.

**Create and activate the environment:**
```bash
python -m venv arc
source arc/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
The model learns from the **Google QuickDraw Dataset**.
1.  Create a `/data` folder in the root directory.
2.  Download `.npy` bitmap files for your chosen categories (e.g., `apple.npy`, `car.npy`).
3.  Place files in the `/data` folder. The script automatically cleans class names (e.g., removing `full_numpy_bitmap_` prefixes).

### 3. Training the Model
To handle large datasets without crashing your system RAM, the trainer uses a `uint8` loading strategy and a `Rescaling` layer in the model architecture.

**Run training with environment flags:**
```bash
TF_USE_LEGACY_KERAS=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python train_model.py --data-dir data --epochs 25
```
*This produces `quickdraw_model.keras` and a corresponding classes JSON.*

### 4. Exporting for Web
Since the frontend uses **TensorFlow.js**, you must convert the Keras model. The export script is configured to bypass Keras 3 metadata issues by saving as a raw graph.

**Run the export script:**
```bash
python export.py
```
*This populates the `/web_model` directory with `model.json` and weight fragments.*

### 5. Running the Game
Because browsers block local file access (CORS), you must serve the project via a local server.

**Start the server:**
```bash
python -m http.server 8000
```
Navigate to `http://localhost:8000`. **Note:** Use `Ctrl+Shift+R` to force a hard refresh if you update your model or classes.

---

## 🛠️ Technical Specifications

| Component | Detail |
| :--- | :--- |
| **Model Architecture** | Sequential CNN with 3 Conv2D layers, Max Pooling, and Dropout |
| **Input Size** | 28x28x1 Grayscale |
| **Optimization** | Adam Optimizer with ReduceLROnPlateau & EarlyStopping |
| **Hardware Used** | NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) |
| **Preprocessing** | 0-255 `uint8` to `float32` Rescaling inside the model graph |

---

## 🏆 Accomplishments
* **Top 50 Finalist** - Microsoft AI Unlocked Hackathon.
* **Top 15 Finalist** - NCIIPC Startup India AI Grand Challenge.