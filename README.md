
AI-Powered Crowd Behavior Predictor
An intelligent system for real-time crowd behavior analysis using computer vision, reinforcement learning, and deep learning.

This project leverages state-of-the-art AI techniques to monitor and predict crowd behavior from live feeds (e.g., drones, CCTV) or pre-recorded videos. It combines YOLOv8 for object and pose detection, LSTM for behavior classification, Q-Learning for adaptive decision-making, and a rule-based system to deliver actionable insights. The system is deployed via an interactive Streamlit interface, featuring anomaly heatmaps, movement tracking, pose analysis, and a chatbot for real-time alerts and response strategies.

Key Features
Real-Time Monitoring: Analyze live feeds from webcams or RTSP-enabled cameras.
Video Analysis: Process pre-recorded videos (MP4, AVI, MOV).
Behavior Classification: Detect "Calm," "Aggressive," "Dispersing," or "Stampede" behaviors.
Advanced Tracking: Monitor crowd density, movement speed, direction, and pose variance.
Visualization: Display heatmaps, movement vectors, and trend plots.
Adaptive Thresholds: Utilize Q-Learning to dynamically adjust detection parameters.
Autonomous Decisions: Suggest response strategies based on detected anomalies.
User Interface: Streamlit-based dashboard with video output, metrics, and chatbot.
Prerequisites
Python: Version 3.8 or higher
Hardware: GPU recommended for optimal performance (e.g., NVIDIA CUDA support)
Camera: Webcam or RTSP-compatible device (for real-time analysis)
Installation
Clone the Repository

bash

Collapse

Wrap

Copy
git clone https://github.com/your-username/ai-crowd-behavior-predictor.git
cd ai-crowd-behavior-predictor
Create a Virtual Environment (Recommended)

bash

Collapse

Wrap

Copy
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install Dependencies

bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Example requirements.txt:

text

Collapse

Wrap

Copy
streamlit==1.24.0
ultralytics==8.0.0
opencv-python==4.8.0
supervision==0.15.0
numpy==1.24.0
matplotlib==3.7.0
tensorflow==2.12.0
Prepare Models

YOLOv8 Models: Automatically downloaded (yolov8n.pt, yolov8n-pose.pt) on first run.
LSTM Model: Place lstm_crowd_behavior.h5 in the root directory.
Label Encoder: Place label_encoder_classes.npy in the root directory.
Note: If LSTM model or label encoder files are missing, refer to the "Training" section.

Usage
Launch the Application
bash

Collapse

Wrap

Copy
streamlit run app.py
Select Input Source
Real-Time Feed: Use a webcam (default) or enter an RTSP URL. Click "Start Real-Time Analysis."
Pre-Recorded Video: Upload a video file to analyze.
Output
Video Display: Annotated frames with bounding boxes, movement vectors, and heatmaps.
Plots: Trends for density, speed, and pose variance.
Metrics: Real-time statistics (e.g., people count, density).
Chatbot: Alerts and response strategies (e.g., "Deploy security" for aggressive behavior).
Project Structure
text

Collapse

Wrap

Copy
ai-crowd-behavior-predictor/
├── app.py                   # Main application script
├── lstm_crowd_behavior.h5   # Pre-trained LSTM model
├── label_encoder_classes.npy# Encoded behavior labels
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── venv/                    # Virtual environment (optional)
Training the LSTM Model
To generate lstm_crowd_behavior.h5 and label_encoder_classes.npy:

Collect labeled crowd video data (e.g., "Calm," "Stampede").
Extract features (density, speed, pose variance) using the script’s processing logic.
Train an LSTM model with TensorFlow/Keras (sequence length: 10).
Save the model and label encoder files in the root directory.
Contact the repository owner for sample training scripts or datasets.

Limitations
Requires pre-trained LSTM model for full behavior prediction.
Real-time performance varies with hardware capabilities.
RTSP feeds depend on network stability.
Future Enhancements
Multi-camera feed support.
Cloud deployment for scalability.
Integration with audio or IoT sensors.
Enhanced chatbot with NLP capabilities.
