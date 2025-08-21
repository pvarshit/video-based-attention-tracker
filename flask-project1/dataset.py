import cv2
import mediapipe as mp
import numpy as np
import csv
from collections import Counter


# Initialize MediaPipe Pose & Face Mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()

# Open video file
video_path = "oc3_i1_4.mp4"
cap = cv2.VideoCapture(video_path)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_count = 0
labeled_data = []

def classify_posture(pose_landmarks, face_landmarks, frame):
    """Classify attention state based on improved gaze detection and body tilt."""
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Estimate person's distance using nose Z-coordinate
    person_distance = abs(nose.z) if nose.z else 1.0  # Avoid division by zero

    # Normalized Head Tilt Calculation
    raw_head_tilt = abs(nose.x - (left_shoulder.x + right_shoulder.x) / 2)
    head_tilt = raw_head_tilt / person_distance  # Normalize by distance

    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
        # Hips are visible → Calculate body tilt normally
        raw_body_tilt = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
    else:
        # Hips are missing → Fallback to shoulder tilt instead
        raw_body_tilt = abs(left_shoulder.y - right_shoulder.y)
    
    body_tilt = raw_body_tilt / person_distance

    # Enhanced Gaze Detection using Additional Landmarks (Normalized)
    gaze_deviation = 0.0
    downward_gaze = False

    if face_landmarks:
        left_eye_upper = face_landmarks.landmark[159].y
        left_eye_lower = face_landmarks.landmark[145].y
        right_eye_upper = face_landmarks.landmark[386].y
        right_eye_lower = face_landmarks.landmark[374].y

        downward_gaze = (left_eye_lower - left_eye_upper > 0.0275) and (right_eye_lower - right_eye_upper > 0.0275)

        left_eye = np.mean([
            face_landmarks.landmark[33].x,
            face_landmarks.landmark[159].x,
            face_landmarks.landmark[145].x,
            face_landmarks.landmark[133].x
        ])

        right_eye = np.mean([
            face_landmarks.landmark[263].x,
            face_landmarks.landmark[386].x,
            face_landmarks.landmark[374].x,
            face_landmarks.landmark[362].x
        ])

        raw_gaze_deviation = abs(left_eye - right_eye)
        gaze_deviation = raw_gaze_deviation / person_distance  # Normalize by distance

        gaze_label = "Distraction" if downward_gaze or gaze_deviation >= 0.17 else "Attention"
    else:
        gaze_label = "Unknown"
    print(head_tilt)
    # print(left_eye_lower - left_eye_upper)
    # print(right_eye_lower - right_eye_upper)
    # Final Classification with Normalized Thresholds
    if head_tilt < 0.07 and body_tilt < 0.15 and gaze_label == "Attention":
        return "Attention", gaze_deviation
    elif head_tilt > 0.11 or body_tilt > 0.25 or gaze_label == "Distraction":
        return "Distraction", gaze_deviation
    else:
        return "Neutral", gaze_deviation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)
    
    label = "Unknown"
    gaze_deviation = 0.0

    if pose_results.pose_landmarks:
        label, gaze_deviation = classify_posture(
            pose_results.pose_landmarks, 
            face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None, 
            frame
        )
    
    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display gaze deviation
    cv2.putText(frame, f"Gaze Dev: {gaze_deviation:.3f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Save frame label every second
    if frame_count % frame_rate == 0:
        labeled_data.append([frame_count // frame_rate, label])
    
    # Display frame with label
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Attention Tracking", frame)
    
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





# Calculate percentages
label_counts = Counter([row[1] for row in labeled_data])  # Extract labels
total_frames = sum(label_counts.values())

attention_percent = (label_counts.get("Attention", 0) / total_frames) * 100 if total_frames else 0
neutral_percent = (label_counts.get("Neutral", 0) / total_frames) * 100 if total_frames else 0
distraction_percent = (label_counts.get("Distraction", 0) / total_frames) * 100 if total_frames else 0

# Define CSV file
summary_file = "video_summary.csv"
write_header = False

# Check if the file exists to avoid rewriting headers
try:
    with open(summary_file, "r") as file:
        if file.readline():
            write_header = False
except FileNotFoundError:
    write_header = True  # File doesn't exist, write headers

# Write to CSV
with open(summary_file, "a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["videoname", "attention", "neutral", "distraction"])
    writer.writerow([video_path, attention_percent, neutral_percent, distraction_percent])

print(f"Summary saved to {summary_file}")
