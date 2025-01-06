# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import math
# import pygame
# import screen_brightness_control as sbc
#
# # Suppress TensorFlow warnings
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # Initialize sound system
# pygame.mixer.init()
# sound_file = "beep-warning-6387.mp3"  # Replace with the correct file path
#
#
# def play_sound(file_path):
#     try:
#         pygame.mixer.music.load(file_path)
#         pygame.mixer.music.play()
#     except Exception as e:
#         print(f"Error playing sound: {e}")
#
#
# def calculate_angle(a, b, c):
#     ba = (a[0] - b[0], a[1] - b[1])
#     bc = (c[0] - b[0], c[1] - b[1])
#     cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
#             math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2)
#     )
#     angle = math.degrees(math.acos(cos_angle))
#     return angle
#
#
# def draw_angle(frame, p1, p2, p3, angle, color):
#     cv2.line(frame, p1, p2, color, 2)
#     cv2.line(frame, p2, p3, color, 2)
#     cv2.putText(frame, f"{int(angle)}\u00b0", (p2[0] + 10, p2[1] - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
#
#
# # Initialize MediaPipe solutions
# mp_hands = mp.solutions.hands
# mp_face_detection = mp.solutions.face_detection
# mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils
#
# # Initialize video capture
# cap = cv2.VideoCapture(0)
#
# # Initialize hand tracking, face detection, and pose detection
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# # Load YOLOv4-tiny model for water bottle detection
# net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
#
# layer_names = net.getLayerNames()
# try:
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# except:
#     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
# # Initialize variables for hydration tracking
# sip_count = 0
# last_sip_time = 0
# sip_cooldown = 2  # seconds between sips to avoid double-counting
# sip_threshold = 0.1  # vertical wrist movement threshold for sip detection
# cup_ratio = 10  # sips per cup
# bottle_detected = False  # Flag to track if a water bottle is detected
#
# # Initialize variables for posture analysis
# posture_score = 0
# posture_feedback = "Good Posture"
#
# # Initialize variables for mouth and wrist detection
# mouth_open_threshold = 0.03  # Distance threshold for mouth open
# wrist_bend_threshold = 150  # Angle threshold for wrist bending
#
# # Virtual keyboard setup
# keys = [
#     ['1', '2', '3'],
#     ['4', '5', '6'],
#     ['7', '8', '9'],
#     ['Switch', '0', '10']
# ]
#
# # Initialize variables for brightness adjustment
# mode = "Auto"
# current_brightness = sbc.get_brightness()[0]
# last_adjustment_time = 0
# cooldown_period = 0.3  # Reduced cooldown for faster adjustments
#
# # Brightness thresholds and buffer range
# OPTIMAL_BRIGHTNESS_RANGE_FACE = (120, 180)  # LAB L channel values
# OPTIMAL_BRIGHTNESS_RANGE_BG = (90, 150)  # LAB L channel values
# BUFFER_RANGE = 5
# MIN_BRIGHTNESS = 15  # Increased minimum for better visibility
# MAX_BRIGHTNESS = 100
#
# # Constants for brightness calculation
# FACE_WEIGHT = 0.7
# BG_WEIGHT = 0.3
# IDEAL_FACE_LIGHTNESS = 150  # Mid-bright for good visibility
# IDEAL_BG_LIGHTNESS = 120  # Slightly darker than face
# SCALING_FACTOR = 0.3  # Increased scaling for more aggressive adjustments
# SMOOTHING_FACTOR = 0.2  # Reduced smoothing for quicker responses
# ADJUSTMENT_THRESHOLD = 1.0  # Lowered threshold for more sensitivity
#
# # Calibration variables for posture correction
# is_calibrated = False
# calibration_frames = 0
# calibration_shoulder_angles = []
# calibration_neck_angles = []
# calibration_shoulder_distances = []  # For shoulder line
# calibration_chest_chin_distances = []  # New: For chest-chin distance
# shoulder_threshold = 0  # Placeholder; will be set after calibration
# neck_threshold = 0  # Placeholder; will be set after calibration
# calibrated_shoulder_distance = 0  # Calibrated shoulder distance
# calibrated_chest_chin_distance = 0  # New: Calibrated chest-chin distance
#
# # Alert system variables for posture correction
# alert_cooldown = 5  # Time in seconds to wait between alerts
# last_alert_time = 0
#
#
# def draw_keyboard(frame, selected_key=None):
#     h, w, _ = frame.shape
#     start_x = 10
#     start_y = h // 2 - 150
#     key_size = 60
#     padding = 10
#
#     for i, row in enumerate(keys):
#         for j, key in enumerate(row):
#             x = start_x + j * (key_size + padding)
#             y = start_y + i * (key_size + padding)
#
#             if key == selected_key:
#                 cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), (0, 255, 0), -1)
#             else:
#                 cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), (255, 255, 255), 2)
#
#             cv2.putText(frame, key, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#
# def get_key_at_position(x, y):
#     h, w, _ = cap.read()[1].shape
#     start_x = 10
#     start_y = h // 2 - 150
#     key_size = 60
#     padding = 10
#
#     for i, row in enumerate(keys):
#         for j, key in enumerate(row):
#             key_x = start_x + j * (key_size + padding)
#             key_y = start_y + i * (key_size + padding)
#             if key_x < x < key_x + key_size and key_y < y < key_y + key_size:
#                 return key
#     return None
#
#
# def preprocess_region(region):
#     lab_region = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
#     l_channel = lab_region[:, :, 0]  # L channel represents lightness
#
#     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
#     enhanced_l = clahe.apply(l_channel)
#
#     sobel_x = cv2.Sobel(enhanced_l, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(enhanced_l, cv2.CV_64F, 0, 1, ksize=3)
#     magnitude = cv2.magnitude(sobel_x, sobel_y)
#
#     normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#     return normalized.astype(np.uint8), np.mean(l_channel)
#
#
# def calculate_optimal_brightness(face_lightness, bg_lightness, current_brightness):
#     face_adjustment = (IDEAL_FACE_LIGHTNESS - face_lightness) * FACE_WEIGHT
#     bg_adjustment = (IDEAL_BG_LIGHTNESS - bg_lightness) * BG_WEIGHT
#
#     total_adjustment = face_adjustment + bg_adjustment
#
#     new_brightness = current_brightness + (total_adjustment * SCALING_FACTOR)
#
#     smoothed_brightness = (SMOOTHING_FACTOR * new_brightness +
#                            (1 - SMOOTHING_FACTOR) * current_brightness)
#
#     return max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, smoothed_brightness))
#
#
# def adjust_brightness(face_region, background_region):
#     global current_brightness, last_adjustment_time
#
#     if time.time() - last_adjustment_time < cooldown_period:
#         return current_brightness, None, None
#
#     try:
#         face_features, face_lightness = preprocess_region(face_region)
#         bg_features, bg_lightness = preprocess_region(background_region)
#
#         target_brightness = calculate_optimal_brightness(
#             face_lightness,
#             bg_lightness,
#             current_brightness
#         )
#
#         if abs(target_brightness - current_brightness) >= ADJUSTMENT_THRESHOLD:
#             current_brightness = target_brightness
#             sbc.set_brightness(int(current_brightness))
#             last_adjustment_time = time.time()
#
#         print(
#             f"Face Lightness: {face_lightness:.2f}, BG Lightness: {bg_lightness:.2f}, Brightness: {current_brightness}%")
#
#         return current_brightness, face_lightness, bg_lightness
#
#     except Exception as e:
#         print(f"Error in brightness adjustment: {e}")
#         return current_brightness, None, None
#
#
# def preprocess_frame(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized_frame = clahe.apply(gray_frame)
#
#     sobel_x = cv2.Sobel(equalized_frame, cv2.CV_64F, 1, 0, ksize=5)
#     sobel_y = cv2.Sobel(equalized_frame, cv2.CV_64F, 0, 1, ksize=5)
#     sobel_frame = cv2.magnitude(sobel_x, sobel_y)
#     sobel_frame = cv2.normalize(sobel_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#
#     return equalized_frame, sobel_frame
#
#
# def detect_bottle(frame):
#     global bottle_detected
#     height, width, channels = frame.shape
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     bottle_detected = False
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and classes[class_id] == "bottle":
#                 bottle_detected = True
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, "Water Bottle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return frame
#
#
# def check_posture(landmarks):
#     global posture_score, posture_feedback
#
#     left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#     right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#     left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#     right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#     left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
#     right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
#
#     shoulder_angle = calculate_angle(left_shoulder, right_shoulder, right_hip)
#     neck_angle = calculate_angle(left_shoulder, left_ear, left_hip)
#
#     if shoulder_angle > 100 or neck_angle > 20:
#         posture_feedback = "Straighten your back"
#         posture_score = max(0, posture_score - 1)
#     else:
#         posture_feedback = "Good Posture"
#         posture_score = min(100, posture_score + 1)
#
#
# def detect_sip(landmarks):
#     global sip_count, last_sip_time
#
#     if not bottle_detected:
#         return
#
#     left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#     right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
#     mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
#
#     if (time.time() - last_sip_time) > sip_cooldown:
#         if abs(left_wrist[1] - mouth[1]) < sip_threshold or abs(right_wrist[1] - mouth[1]) < sip_threshold:
#             sip_count += 1
#             last_sip_time = time.time()
#
#
# def is_mouth_open(face_landmarks):
#     upper_lip = face_landmarks.landmark[13]  # Upper lip landmark
#     lower_lip = face_landmarks.landmark[14]  # Lower lip landmark
#     distance = np.sqrt((upper_lip.x - lower_lip.x) ** 2 + (upper_lip.y - lower_lip.y) ** 2)
#     return distance > mouth_open_threshold
#
#
# def is_wrist_bent(landmarks):
#     wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
#     elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
#     shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#
#     angle = calculate_angle([wrist.x, wrist.y], [elbow.x, elbow.y], [shoulder.x, shoulder.y])
#     return angle < wrist_bend_threshold
#
#
# # Main loop
# selected_key = None
# button_pressed = False
# press_start_time = 0
# press_duration = 0.5  # Duration in seconds to consider a press
#
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Error: Could not read frame from camera.")
#         break
#
#     image = cv2.flip(image, 1)
#
#     equalized_frame, sobel_frame = preprocess_frame(image)
#
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     hand_results = hands.process(rgb_image)
#     face_results = face_detection.process(rgb_image)
#     pose_results = pose.process(rgb_image)
#     face_mesh_results = face_mesh.process(rgb_image)
#
#     draw_keyboard(image, selected_key)
#
#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#             h, w, _ = image.shape
#             index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
#             thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
#
#             key = get_key_at_position(index_x, index_y)
#             if key:
#                 selected_key = key
#
#             distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
#             if distance < 20:
#                 if not button_pressed:
#                     press_start_time = time.time()
#                     button_pressed = True
#                 elif time.time() - press_start_time >= press_duration:
#                     if selected_key:
#                         if selected_key == 'Switch':
#                             mode = "Manual" if mode == "Auto" else "Auto"
#                         elif selected_key == '10':
#                             mode = "Manual"
#                             current_brightness = 100
#                             sbc.set_brightness(current_brightness)
#                         elif selected_key.isdigit():
#                             mode = "Manual"
#                             current_brightness = int(selected_key) * 10
#                             sbc.set_brightness(current_brightness)
#                     button_pressed = False
#             else:
#                 button_pressed = False
#
#     face_brightness = None
#     bg_brightness = None
#
#     if face_results.detections:
#         for detection in face_results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = image.shape
#             x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
#
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             face_region = image[y:y + h, x:x + w]
#             background_region = np.copy(image)
#             cv2.rectangle(background_region, (x, y), (x + w, y + h), (0, 0, 0), -1)
#
#             if mode == "Auto":
#                 current_brightness, face_brightness, bg_brightness = adjust_brightness(face_region, background_region)
#
#             if face_brightness is not None and bg_brightness is not None:
#                 cv2.putText(image, f"Face: {face_brightness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
#                             2)
#                 cv2.putText(image, f"BG: {bg_brightness:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#     if pose_results.pose_landmarks:
#         landmarks = pose_results.pose_landmarks.landmark
#
#         left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1]),
#                          int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0]))
#         right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image.shape[1]),
#                           int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image.shape[0]))
#         left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * image.shape[1]),
#                     int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * image.shape[0]))
#         chin = (int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * image.shape[1]),
#                 int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * image.shape[0]))
#
#         shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
#         neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
#
#         shoulder_distance = math.hypot(right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1])
#
#         chest_midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
#
#         chest_chin_distance = math.hypot(chin[0] - chest_midpoint[0], chin[1] - chest_midpoint[1])
#
#         if not is_calibrated and calibration_frames < 30:
#             calibration_shoulder_angles.append(shoulder_angle)
#             calibration_neck_angles.append(neck_angle)
#             calibration_shoulder_distances.append(shoulder_distance)
#             calibration_chest_chin_distances.append(chest_chin_distance)
#             calibration_frames += 1
#             cv2.putText(image, f"Calibrating... {calibration_frames}/30", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#         elif not is_calibrated:
#             shoulder_threshold = np.mean(calibration_shoulder_angles) - 10
#             neck_threshold = np.mean(calibration_neck_angles) - 10
#             calibrated_shoulder_distance = np.mean(calibration_shoulder_distances)
#             calibrated_chest_chin_distance = np.mean(calibration_chest_chin_distances)
#             is_calibrated = True
#             print(
#                 f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")
#             print(f"Calibrated shoulder distance: {calibrated_shoulder_distance:.2f}")
#             print(f"Calibrated chest-chin distance: {calibrated_chest_chin_distance:.2f}")
#
#         mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
#         draw_angle(image, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
#         draw_angle(image, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))
#
#         shoulder_line_color = (0, 255, 0)
#         if is_calibrated and shoulder_distance < calibrated_shoulder_distance * 0.9:
#             shoulder_line_color = (0, 0, 255)
#         cv2.line(image, left_shoulder, right_shoulder, shoulder_line_color, 2)
#
#         chest_chin_line_color = (0, 255, 0)
#         if is_calibrated and chest_chin_distance < calibrated_chest_chin_distance * 0.9:
#             chest_chin_line_color = (0, 0, 255)
#         cv2.line(image, chest_midpoint, chin, chest_chin_line_color, 2)
#
#         if is_calibrated:
#             current_time = time.time()
#             if (shoulder_angle < shoulder_threshold or neck_angle < neck_threshold or
#                     shoulder_distance < calibrated_shoulder_distance * 0.9 or
#                     chest_chin_distance < calibrated_chest_chin_distance * 0.9):
#                 status = "Poor Posture"
#                 color = (0, 0, 255)
#                 if current_time - last_alert_time > alert_cooldown:
#                     print("Poor posture detected! Please sit up straight.")
#                     play_sound(sound_file)
#                     last_alert_time = current_time
#             else:
#                 status = "Good Posture"
#                 color = (0, 255, 0)
#
#             cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
#             cv2.putText(image, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
#             cv2.putText(image, f"Neck Angle: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
#             cv2.putText(image, f"Chest-Chin Distance: {chest_chin_distance:.1f}/{calibrated_chest_chin_distance:.1f}",
#                         (10, 120),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
#
#     cv2.putText(image, f"Brightness: {current_brightness}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.putText(image, f"Mode: {mode}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#     cv2.imshow('Original Frame', image)
#     cv2.imshow("Equalized Frame", equalized_frame)
#     cv2.imshow("Sobel Edge Detection", sobel_frame)
#
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()