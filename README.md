# 🧠 AI-Powered Crowd Behavior Predictor

An intelligent system for **real-time crowd behavior analysis** using computer vision, reinforcement learning, and deep learning.

This project leverages state-of-the-art AI techniques to monitor and predict crowd behavior from **live feeds** (e.g., drones, CCTV) or **pre-recorded videos**. It combines **YOLOv8** for object and pose detection, **LSTM** for behavior classification, **Q-Learning** for adaptive decision-making, and a rule-based system to deliver **actionable insights**.

---

## 🚀 Key Features

- **Real-Time Monitoring**: Analyze live feeds from webcams or RTSP-enabled cameras.
- **Video Analysis**: Process pre-recorded videos (`.mp4`, `.avi`, `.mov`).
- **Behavior Classification**: Detect crowd behaviors — _Calm_, _Aggressive_, _Dispersing_, _Stampede_.
- **Advanced Tracking**: Monitor crowd **density**, **movement speed**, **direction**, and **pose variance**.
- **Visualization**: Generate **heatmaps**, **movement vectors**, and **trend plots**.
- **Adaptive Thresholds**: Use **Q-Learning** to adjust detection parameters dynamically.
- **Autonomous Decisions**: Suggest response strategies for anomalies.
- **User Interface**: Streamlit dashboard with annotated video, metrics, plots, and a real-time chatbot.

---

## 🧠 Prerequisites

- **Python**: 3.8 or higher
- **Hardware**: GPU recommended (NVIDIA CUDA)
- **Camera**: Webcam or RTSP-compatible device (for real-time)

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-crowd-behavior-predictor.git
cd ai-crowd-behavior-predictor
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Example <code>requirements.txt</code></summary>

```ini
streamlit==1.24.0
ultralytics==8.0.0
opencv-python==4.8.0
supervision==0.15.0
numpy==1.24.0
matplotlib==3.7.0
tensorflow==2.12.0
```

</details>

---

## 🧠 Prepare Models

- **YOLOv8**: Models (`yolov8n.pt`, `yolov8n-pose.pt`) are auto-downloaded on first run.
- **LSTM Model**: Place `lstm_crowd_behavior.h5` in the project root.
- **Label Encoder**: Place `label_encoder_classes.npy` in the root directory.

> ⚠️ If any are missing, refer to the [Training the LSTM Model](#🧪-training-the-lstm-model) section.

---

## ▶️ Usage

### 🔹 Launch the App

```bash
streamlit run app.py
```

### 🔹 Choose Input Mode

- **Real-Time Feed**: Use webcam or enter RTSP URL. Click **Start Real-Time Analysis**.
- **Upload Video**: Analyze pre-recorded video files.

---

## 📊 Output

- 🎥 Annotated video (bounding boxes, pose vectors, heatmaps)
- 📈 Density, speed, and pose variance plots
- 📉 Real-time crowd metrics (e.g., count, density)
- 🤖 Chatbot alerts & strategies (e.g., “Deploy security” for aggressive detection)

---

## 🗂️ Project Structure

```bash
ai-crowd-behavior-predictor/
├── app.py                     # Main Streamlit app
├── analyze_crowd.py           # Real-time analyzer
├── prepare_lstm_data.py       # Feature extractor
├── train_lstm.py              # LSTM model training
├── lstm_crowd_behavior.h5     # Pre-trained LSTM model
├── label_encoder_classes.npy  # Encoded labels
├── crowd_data.csv             # Processed feature data
├── X_train.npy / y_train.npy  # Training data
├── X_test.npy / y_test.npy    # Test data
├── yolov8n.pt                 # YOLOv8 model
├── yolov8n-pose.pt            # YOLOv8 pose model
├── requirements.txt
└── README.md
```

---

## 🧪 Training the LSTM Model

To generate `lstm_crowd_behavior.h5` and `label_encoder_classes.npy`:

1. Collect labeled videos for behaviors (e.g., _Calm_, _Stampede_).
2. Extract features like **density**, **speed**, and **pose variance** using `prepare_lstm_data.py`.
3. Train the LSTM model using `train_lstm.py` with TensorFlow/Keras (sequence length: 10).
4. Save the model and label encoder in the root directory.

> 🧪 Sample training datasets and scripts available on request.

---

## ⚠️ Limitations

- Requires trained LSTM model and encoder for full behavior classification.
- Real-time performance depends on hardware specifications.
- RTSP feed quality is subject to network stability.

---

## 📌 Future Enhancements

- 🔀 Support multiple simultaneous camera feeds
- ☁️ Cloud-based deployment with autoscaling
- 🔊 Integrate IoT & audio sensors for better context
- 🤖 Upgrade chatbot with NLP-based conversational interaction

---

