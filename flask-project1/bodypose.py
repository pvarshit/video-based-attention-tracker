import cv2
import mediapipe as mp
import numpy as np
import csv

# Initialize MediaPipe Pose & Face Mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()

# Open video file
video_path = "oc6_i5_1.mp4"
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

    # 1️⃣ Normalized Head Tilt Calculation
    raw_head_tilt = abs(nose.x - (left_shoulder.x + right_shoulder.x) / 2)
    head_tilt = raw_head_tilt / person_distance  # Normalize by distance
    # print(head_tilt)
    
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:  
        # ✅ Hips are visible → Calculate body tilt normally
        raw_body_tilt = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
        body_tilt = raw_body_tilt / person_distance 
    else:
        # ❌ Hips are missing → Fallback to shoulder tilt instead
        raw_body_tilt = abs(left_shoulder.y - right_shoulder.y)
        body_tilt = raw_body_tilt / person_distance 
    


    # 3️⃣ Enhanced Gaze Detection using Additional Landmarks (Normalized)
    gaze_deviation = 0.0
    downward_gaze = False
    if face_landmarks:
        left_eye_upper = face_landmarks.landmark[159].y
        left_eye_lower = face_landmarks.landmark[145].y
        right_eye_upper = face_landmarks.landmark[386].y
        right_eye_lower = face_landmarks.landmark[374].y
        
        downward_gaze = (left_eye_lower - left_eye_upper > 0.023) and (right_eye_lower - right_eye_upper > 0.023)
        print("left eye:",left_eye_lower - left_eye_upper)
        print("right eye:",right_eye_lower - right_eye_upper)
        left_eye = np.mean([
            face_landmarks.landmark[33].x,  # Right eye (mirrored index)
            face_landmarks.landmark[159].x,
            face_landmarks.landmark[145].x,
            face_landmarks.landmark[133].x  # Outer eye corner
        ])
        right_eye = np.mean([
            face_landmarks.landmark[263].x,  # Left eye
            face_landmarks.landmark[386].x,
            face_landmarks.landmark[374].x,
            face_landmarks.landmark[362].x  # Outer eye corner
        ])
        
        raw_gaze_deviation = abs(left_eye - right_eye)
        gaze_deviation = raw_gaze_deviation / person_distance  # Normalize by distance
        
        # Draw gaze landmarks
        for idx in [33, 159, 145, 133, 263, 386, 374, 362]:
            lm = face_landmarks.landmark[idx]
            h, w, _ = frame.shape
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
        
        # More relaxed gaze deviation threshold
        if downward_gaze:
            gaze_label = "Distraction"
        elif gaze_deviation < 0.15:
            gaze_label = "Attention"
        else:
            gaze_label = "Distraction"
    else:
        gaze_label = "Unknown"

    # 🏆 Final Classification with Normalized Thresholds
    if head_tilt < 0.07 and body_tilt<0.15 and gaze_label=="Attention":
        return "Attention", gaze_deviation
    elif head_tilt > 0.095 or body_tilt>0.25 or gaze_label=="Distraction":
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
        label, gaze_deviation = classify_posture(pose_results.pose_landmarks, face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None, frame)
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

# Save labeled data to CSV
with open("frame_labels.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Second", "Label"])
    writer.writerows(labeled_data)

print("Frame labeling completed! Check 'frame_labels.csv'.")