import streamlit as st
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tempfile
import os
import random
import math

# Load models
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")  # Object detection model
    pose_model = YOLO("yolov8n-pose.pt")  # Pose detection model
    try:
        lstm_model = load_model("lstm_crowd_behavior.h5")
    except Exception as e:
        st.error(f"Error: Could not load lstm_crowd_behavior.h5: {e}. Please train the model first.")
        return None, None, None
    return yolo_model, pose_model, lstm_model

yolo_model, pose_model, lstm_model = load_models()
if yolo_model is None or pose_model is None or lstm_model is None:
    st.stop()

# Load label encoder classes
try:
    label_encoder_classes = np.load("label_encoder_classes.npy", allow_pickle=True)
except Exception as e:
    st.error(f"Error: Could not load label_encoder_classes.npy: {e}")
    st.stop()

tracker = sv.ByteTrack()
sequence_length = 10

# Response strategies
RESPONSE_STRATEGIES = {
    "Calm": "Maintain regular monitoring. No immediate action required.",
    "Aggressive": "Deploy additional security personnel. Prepare for potential conflict de-escalation.",
    "Dispersing": "Monitor exits and ensure clear pathways. Consider crowd flow management.",
    "Stampede": "Activate emergency protocols immediately. Coordinate with local authorities and medical teams."
}

# Q-Learning setup
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = {}
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def get_state(self, density, speed, variance, movement_uniformity, pose_variance):
        density_bin = int(min(density * 1000, 50) // 10)
        speed_bin = int(min(speed, 50) // 10)
        variance_bin = int(min(variance, 100) // 20)
        movement_bin = int(min(movement_uniformity * 100, 100) // 20)
        pose_bin = int(min(pose_variance * 100, 100) // 20)  # New pose variance bin
        return (density_bin, speed_bin, variance_bin, movement_bin, pose_bin)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.get_q_value(state, a) for a in self.actions]
            return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state, action)] = new_q

# Autonomous Decision-Making Agent
class DecisionAgent:
    def __init__(self):
        self.alert_level = "Normal"
        self.last_escalation_frame = 0
        self.escalation_cooldown = 300  # Frames (e.g., 10 seconds at 30 FPS)

    def decide_action(self, rule_behavior, lstm_behavior, frame_count):
        current_behavior = lstm_behavior if lstm_behavior != "Unknown" else rule_behavior
        if frame_count - self.last_escalation_frame < self.escalation_cooldown:
            return None  # Prevent frequent escalations

        if current_behavior == "Stampede":
            self.alert_level = "Critical"
            self.last_escalation_frame = frame_count
            return "Activate emergency protocols: Notify authorities, activate sirens, and dispatch medical teams."
        elif current_behavior == "Aggressive":
            self.alert_level = "High"
            self.last_escalation_frame = frame_count
            return "Escalate alert: Notify security personnel and prepare for de-escalation."
        elif current_behavior == "Dispersing":
            self.alert_level = "Moderate"
            self.last_escalation_frame = frame_count
            return "Monitor situation: Ensure clear pathways and adjust crowd flow."
        else:
            self.alert_level = "Normal"
            return None

# Initialize agents
actions = ["increase_density", "decrease_density", "increase_speed", "decrease_speed", 
           "increase_variance", "decrease_variance", "increase_pose", "decrease_pose"]
rl_agent = QLearningAgent(actions)
decision_agent = DecisionAgent()

# Initial thresholds
thresholds = {"density": 0.015, "speed": 25, "variance": 50, "pose_variance": 0.5}

# Streamlit UI
st.title("AI-Powered Crowd Behavior Predictor with Movement and Pose Detection")
st.write("Analyze crowd behavior with real-time anomaly heatmaps, overlays, movement, and pose detection.")

# Chatbot section
st.subheader("Crowd Management Chatbot")
chat_placeholder = st.empty()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ["Chatbot: Welcome to Crowd Management Assistant with Movement and Pose Detection."]

# Input selection
input_option = st.radio("Select Input Source:", ("Real-Time Drone/CCTV Feed", "Pre-Recorded Video"))

# Function to generate heatmap
def generate_heatmap(frame, centroids, intensity_factor=50):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for x, y in centroids:
        cv2.circle(heatmap, (x, y), 30, intensity_factor, -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

# Function to process frame with pose detection
def process_frame(frame, prev_positions, density_history, speed_history, pose_history, time_history, frame_count, rl_agent, thresholds):
    # Object detection
    results = yolo_model(frame)
    filtered_boxes = [box for box in results[0].boxes if int(box.cls) == 0]  # Class 0 is "person"

    if len(filtered_boxes) == 0:
        detections = sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=np.int32)
        )
    else:
        detections = sv.Detections(
            xyxy=np.array([box.xyxy[0].cpu().numpy() for box in filtered_boxes]),
            confidence=np.array([box.conf[0].cpu().numpy() for box in filtered_boxes]),
            class_id=np.array([0] * len(filtered_boxes))
        )

    tracked_detections = tracker.update_with_detections(detections)
    annotated_frame = frame.copy()

    # Pose detection
    pose_results = pose_model(frame)
    pose_keypoints = pose_results[0].keypoints.xy.cpu().numpy() if pose_results[0].keypoints is not None else []

    num_people = len(tracked_detections)
    frame_area = frame.shape[0] * frame.shape[1]
    density = num_people / frame_area * 10000 if frame_area > 0 else 0

    speeds = []
    centroids = []
    movement_vectors = []
    pose_orientations = []  # Store pose orientation angles

    # Process tracked detections
    for box, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
        x1, y1, x2, y2 = map(int, box)
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        centroids.append(centroid)
        if prev_positions[track_id] is not None:
            prev_x, prev_y = prev_positions[track_id]
            dx = centroid[0] - prev_x
            dy = centroid[1] - prev_y
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
            angle = math.atan2(dy, dx) if speed > 0 else 0
            movement_vectors.append((speed, angle))
            arrow_length = min(int(speed * 2), 50)
            end_x = int(centroid[0] + arrow_length * math.cos(angle))
            end_y = int(centroid[1] + arrow_length * math.sin(angle))
            cv2.arrowedLine(annotated_frame, centroid, (end_x, end_y), (255, 255, 0), 2, tipLength=0.3)
        prev_positions[track_id] = centroid
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Process pose keypoints
    for keypoints in pose_keypoints:
        if len(keypoints) >= 17:  # Ensure all keypoints are present (YOLOv8 has 17 keypoints)
            # Use shoulder (5, 6) and hip (11, 12) to estimate torso orientation
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            left_hip = keypoints[11][:2]
            right_hip = keypoints[12][:2]
            if all(np.all(kp != 0) for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                torso_dx = (right_shoulder[0] + right_hip[0]) / 2 - (left_shoulder[0] + left_hip[0]) / 2
                torso_dy = (right_shoulder[1] + right_hip[1]) / 2 - (left_shoulder[1] + left_hip[1]) / 2
                orientation = math.atan2(torso_dy, torso_dx)
                pose_orientations.append(orientation)
                # Draw keypoints
                for kp in keypoints:
                    if np.all(kp != 0):
                        cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 5, (255, 0, 255), -1)

    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    pose_variance = np.var(pose_orientations) if pose_orientations else 0

    # Movement uniformity
    if movement_vectors:
        angles = [vec[1] for vec in movement_vectors]
        angle_variance = np.var([math.cos(a) for a in angles]) + np.var([math.sin(a) for a in angles])
        movement_uniformity = max(0, 1 - angle_variance)
    else:
        movement_uniformity = 0

    # RL: Adjust thresholds
    state = rl_agent.get_state(density, avg_speed, speed_variance, movement_uniformity, pose_variance)
    action = rl_agent.choose_action(state)
    if action == "increase_density":
        thresholds["density"] += 0.001
    elif action == "decrease_density":
        thresholds["density"] = max(0.001, thresholds["density"] - 0.001)
    elif action == "increase_speed":
        thresholds["speed"] += 1
    elif action == "decrease_speed":
        thresholds["speed"] = max(5, thresholds["speed"] - 1)
    elif action == "increase_variance":
        thresholds["variance"] += 5
    elif action == "decrease_variance":
        thresholds["variance"] = max(10, thresholds["variance"] - 5)
    elif action == "increase_pose":
        thresholds["pose_variance"] += 0.05
    elif action == "decrease_pose":
        thresholds["pose_variance"] = max(0.1, thresholds["pose_variance"] - 0.05)

    # Rule-based behavior with pose
    if (density > thresholds["density"] and avg_speed > thresholds["speed"] and 
        speed_variance > thresholds["variance"] and movement_uniformity > 0.8 and pose_variance < thresholds["pose_variance"]):
        rule_behavior = "Stampede"  # Uniform movement and pose suggest coordinated rush
    elif (density > thresholds["density"] * 0.66 and avg_speed > thresholds["speed"] * 0.8 and 
          movement_uniformity < 0.4 and pose_variance > thresholds["pose_variance"]):
        rule_behavior = "Aggressive"  # Erratic movement and diverse poses suggest conflict
    elif density < thresholds["density"] * 0.33 and avg_speed > thresholds["speed"] * 0.4:
        rule_behavior = "Dispersing"
    else:
        rule_behavior = "Calm"

    # LSTM prediction
    lstm_behavior = "Unknown"
    if len(density_history) >= sequence_length - 1:
        density_history.append(density)
        speed_history.append(avg_speed)
        pose_history.append(pose_variance)
        sequence = np.array(list(zip(density_history[-sequence_length:], speed_history[-sequence_length:])))
        sequence = sequence.reshape(1, sequence_length, 2)
        pred = lstm_model.predict(sequence, verbose=0)
        lstm_behavior = label_encoder_classes[np.argmax(pred)]
    else:
        density_history.append(density)
        speed_history.append(avg_speed)
        pose_history.append(pose_variance)

    frame_count += 1
    time_history.append(frame_count)

    # RL feedback
    reward = 1 if lstm_behavior == rule_behavior and lstm_behavior != "Unknown" else -1
    next_state = rl_agent.get_state(density, avg_speed, speed_variance, movement_uniformity, pose_variance)
    rl_agent.update_q_table(state, action, reward, next_state)

    # Generate heatmap
    intensity = min(50 + int(density * 1000 + avg_speed), 255)
    heatmap = generate_heatmap(frame, centroids, intensity)
    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap, 0.3, 0)

    # Dynamic annotations for anomalies
    if rule_behavior in ["Aggressive", "Stampede"] or lstm_behavior in ["Aggressive", "Stampede"]:
        for x, y in centroids:
            cv2.circle(annotated_frame, (x, y), 40, (0, 0, 255), 3)

    # Autonomous decision-making
    autonomous_action = decision_agent.decide_action(rule_behavior, lstm_behavior, frame_count)
    if autonomous_action:
        st.session_state.chat_history.append(f"Chatbot: AUTONOMOUS ACTION - {autonomous_action}")
        if decision_agent.alert_level == "Critical":
            st.error(f"CRITICAL ALERT: {autonomous_action}")
        elif decision_agent.alert_level == "High":
            st.warning(f"HIGH ALERT: {autonomous_action}")
        elif decision_agent.alert_level == "Moderate":
            st.info(f"MODERATE ALERT: {autonomous_action}")

    # Chatbot alert
    detected_behavior = lstm_behavior if lstm_behavior != "Unknown" else rule_behavior
    if frame_count % 30 == 0:
        alert_msg = (f"Chatbot: ALERT - Detected {detected_behavior} (People: {num_people}, "
                     f"Density: {density:.4f}, Uniformity: {movement_uniformity:.2f}, Pose Variance: {pose_variance:.2f})")
        strategy_msg = f"Chatbot: Strategy - {RESPONSE_STRATEGIES[detected_behavior]}"
        st.session_state.chat_history.extend([alert_msg, strategy_msg])
    chat_placeholder.text("\n".join(st.session_state.chat_history[-5:]))

    # Annotate frame
    cv2.putText(annotated_frame, f"Rule-Based: {rule_behavior}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"LSTM Pred: {lstm_behavior}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(annotated_frame, f"People: {num_people}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Density: {density:.4f}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Speed Variance: {speed_variance:.2f}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Move Uniformity: {movement_uniformity:.2f}", (10, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Pose Variance: {pose_variance:.2f}", (10, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Thresholds: D:{thresholds['density']:.4f}, S:{thresholds['speed']}, "
                f"V:{thresholds['variance']}, P:{thresholds['pose_variance']:.2f}", 
                (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
        cv2.putText(annotated_frame, "ALERT: Aggressive Behavior Detected", (10, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
        cv2.putText(annotated_frame, "ALERT: Stampede Detected!", (10, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle('Crowd Analysis Trends')
    ax1.plot(time_history, density_history, 'b-', label='Density')
    ax1.set_title('Density Over Time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('People/10k pixels')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_history, speed_history, 'r-', label='Speed')
    ax2.set_title('Average Speed Over Time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Pixels/Frame')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(time_history, pose_history, 'g-', label='Pose Variance')
    ax3.set_title('Pose Variance Over Time')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Variance')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()

    return annotated_frame, num_people, density, avg_speed, pose_variance, rule_behavior, lstm_behavior, fig, frame_count

# Real-Time Drone/CCTV Feed
if input_option == "Real-Time Drone/CCTV Feed":
    st.write("Using real-time feed from camera (default: webcam). Enter an RTSP URL for drone/CCTV if needed.")
    rtsp_url = st.text_input("RTSP URL (leave blank for webcam)", "")
    video_source = rtsp_url if rtsp_url else 0

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Error: Could not open real-time feed. Check your camera or RTSP URL.")
        st.stop()

    prev_positions = defaultdict(lambda: None)
    density_history = []
    speed_history = []
    pose_history = []
    time_history = []
    frame_count = 0

    video_placeholder = st.empty()
    plot_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if 'running' not in st.session_state:
        st.session_state.running = False

    if st.button("Start Real-Time Analysis"):
        st.session_state.running = True
    if st.button("Stop Real-Time Analysis"):
        st.session_state.running = False

    if st.session_state.running:
        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("End of feed or error reading frame.")
                break

            annotated_frame, num_people, density, avg_speed, pose_variance, rule_behavior, lstm_behavior, fig, frame_count = process_frame(
                frame, prev_positions, density_history, speed_history, pose_history, time_history, frame_count, rl_agent, thresholds
            )

            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, caption=f"Frame {frame_count}", use_container_width=True)
            plot_placeholder.pyplot(fig)
            metrics_placeholder.write(f"Frame {frame_count}: People: {num_people}, Density: {density:.4f}, "
                                     f"Avg Speed: {avg_speed:.2f}, Pose Variance: {pose_variance:.2f}, "
                                     f"Rule-Based: {rule_behavior}, LSTM: {lstm_behavior}")

            if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
                st.warning("Aggressive behavior detected!")
            if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
                st.error("Stampede detected! Immediate action recommended.")

            plt.close(fig)

    cap.release()

# Pre-Recorded Video
elif input_option == "Pre-Recorded Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video.")
            st.stop()

        prev_positions = defaultdict(lambda: None)
        density_history = []
        speed_history = []
        pose_history = []
        time_history = []
        frame_count = 0

        video_placeholder = st.empty()
        plot_placeholder = st.empty()
        metrics_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, num_people, density, avg_speed, pose_variance, rule_behavior, lstm_behavior, fig, frame_count = process_frame(
                frame, prev_positions, density_history, speed_history, pose_history, time_history, frame_count, rl_agent, thresholds
            )

            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, caption=f"Frame {frame_count}", use_container_width=True)
            plot_placeholder.pyplot(fig)
            metrics_placeholder.write(f"Frame {frame_count}: People: {num_people}, Density: {density:.4f}, "
                                     f"Avg Speed: {avg_speed:.2f}, Pose Variance: {pose_variance:.2f}, "
                                     f"Rule-Based: {rule_behavior}, LSTM: {lstm_behavior}")

            if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
                st.warning("Aggressive behavior detected!")
            if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
                st.error("Stampede detected! Immediate action recommended.")

            plt.close(fig)

        cap.release()
        os.unlink(video_path)
        st.success("Video processing completed!")
    else:
        st.write("Please upload a video file to begin analysis.")