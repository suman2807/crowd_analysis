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

## 🧰 Prerequisites

- **Python**: 3.8 or higher
- **Hardware**: GPU recommended (NVIDIA CUDA)
- **Camera**: Webcam or RTSP-compatible device (for real-time)

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-crowd-behavior-predictor.git
cd ai-crowd-behavior-predictor
