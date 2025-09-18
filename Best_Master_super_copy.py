import cv2
import pprint
import json
import numpy as np
import mediapipe as mp
from collections import deque
import argparse
import time
import matplotlib.pyplot as plt
import math
from ratio_range_body import analyze_movement
from ratio_range_body import get_top_keypoints


MASTER_VIDEO_PATH = "/home/sidharth/Documents/Siddharth_office/gym_assistant /voice_enable_ai_trainer-main/leg/Raj_ka_jalwa_Test/5.mp4" 

MASTER_JSON_PATH  = "master_reference.json"
WEIGHTS_JSON_PATH = "joint_weights.json"
BODY_WEIGHTS_PATH = "body_weights.json"

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def draw_custom_skeleton(image, landmarks,visibility_threshold=0.5):
    h, w = image.shape[:2]
    CUSTOM_CONNECTIONS = [
                                (11, 13), (13, 15),   # Left shoulder to elbow to wrist
                                (12, 14), (14, 16),   # Right shoulder to elbow to wrist
                                (11, 23), (12, 24),   # Shoulder to hip
                                (23, 25), (25, 27),   # Left hip to knee to ankle
                                (24, 26), (26, 28),   # Right hip to knee to ankle
                                (27, 31), (28, 32),   # Left/right ankle to foot
                                (11, 12),                  # Connect both shoulders
                                (23, 24),                  # Connect both hips
                            ]
    connections = CUSTOM_CONNECTIONS
    for idx1, idx2 in connections:
        if landmarks[idx1].visibility > visibility_threshold and landmarks[idx2].visibility > visibility_threshold:
            x1, y1 = int(landmarks[idx1].x * w), int(landmarks[idx1].y * h)
            x2, y2 = int(landmarks[idx2].x * w), int(landmarks[idx2].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(image, (x1, y1), 4, (255, 255, 255), -1)
            cv2.circle(image, (x2, y2), 4, (255, 255, 255), -1)



def get_xy(landmarks, idx):
    return [landmarks[idx].x, landmarks[idx].y]

def normalize_point(point, reference_length):
    return [coord / reference_length for coord in point]

def get_shoulder_width(landmarks):
    left = get_xy(landmarks, 11)
    right = get_xy(landmarks, 12)
    return np.linalg.norm(np.array(left) - np.array(right))

def get_relevant_indices():
    return [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def extract_key_angles(landmarks):
    def get_point(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    angles = {}
    # Elbow
    angles["left_elbow"] = calculate_angle(get_point(11), get_point(13), get_point(15))
    angles["right_elbow"] = calculate_angle(get_point(12), get_point(14), get_point(16))

    # Shoulder
    angles["left_shoulder"] = calculate_angle(get_point(13), get_point(11), get_point(23))
    angles["right_shoulder"] = calculate_angle(get_point(14), get_point(12), get_point(24))

    # Hip
    angles["left_hip"] = calculate_angle(get_point(11), get_point(23), get_point(25))
    angles["right_hip"] = calculate_angle(get_point(12), get_point(24), get_point(26))

    # Knee
    angles["left_knee"] = calculate_angle(get_point(23), get_point(25), get_point(27))
    angles["right_knee"] = calculate_angle(get_point(24), get_point(26), get_point(28))

    # Ankle
    angles["left_ankle"] = calculate_angle(get_point(25), get_point(27), get_point(31))
    angles["right_ankle"] = calculate_angle(get_point(26), get_point(28), get_point(32))

    return angles


BODY_PARTS = {
    "nose"    : [0],
    "wrist"   : [15, 16],
    "elbow"   : [13, 14],
    "shoulder": [11, 12],
    "hip"     : [23, 24],
    "knee"    : [25, 26],
    "ankle"   : [27, 28]
}

def get_relevant_indices():
    return [0,11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def extract_keypoints(landmarks, width, height):
    return {str(idx): [landmarks[idx].x * width, landmarks[idx].y * height] for idx in get_relevant_indices()}



def preprocess_master_video():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(MASTER_VIDEO_PATH)
    frame_data = []

    joint_indices = get_relevant_indices()
    joint_paths = {idx: [] for idx in joint_indices}

    joint_angle_names = [
        "left_elbow", "right_elbow",
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]
    joint_angle_ranges = {name: {"min": float("inf"), "max": float("-inf")} for name in joint_angle_names}

    # Define 12 body keypoints to track for movement-based weights
    point_names = {
        0 : 'nose',
        15: "left_wrist", 16: "right_wrist",
        13: "left_elbow", 14: "right_elbow",
        11: "left_shoulder", 12: "right_shoulder",
        23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle"
    }
    point_paths = {name: [] for name in point_names.values()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Save normalized keypoints for frame
            norm_coords = {
                str(idx): [landmarks[idx].x, landmarks[idx].y]
                for idx in joint_indices
            }

            for idx in joint_indices:
                joint_paths[idx].append(norm_coords[str(idx)])

            for idx, name in point_names.items():
                point_paths[name].append([landmarks[idx].x, landmarks[idx].y])

            # Save joint angles and update min-max ranges
            angles = extract_key_angles(landmarks)
            for joint, angle in angles.items():
                joint_angle_ranges[joint]["min"] = min(joint_angle_ranges[joint]["min"], angle)
                joint_angle_ranges[joint]["max"] = max(joint_angle_ranges[joint]["max"], angle)

            frame_data.append({
                "keypoints": norm_coords,
                "angles": angles
            })

    cap.release()

    # --- ANGLE-BASED WEIGHTS ---
    angle_threshold = 10
    total_range = sum(
        r["max"] - r["min"]
        for r in joint_angle_ranges.values()
        if (r["max"] - r["min"] >= angle_threshold)
    )
    joint_weights = {}
    for joint, r in joint_angle_ranges.items():
        movement = r["max"] - r["min"]
        joint_weights[joint] = movement / total_range if movement >= angle_threshold else 0

    # --- MOTION-BASED WEIGHTS FOR BODY POINTS ---
    point_motions = {}
    for name, path in point_paths.items():
        if len(path) > 1:
            arr = np.array(path)
            motion = np.linalg.norm(arr.max(axis=0) - arr.min(axis=0))
            point_motions[name] = motion
        else:
            point_motions[name] = 0

    total_motion = sum(point_motions.values())
    body_point_weights = {
        name: motion / total_motion if total_motion > 0 else 0
        for name, motion in point_motions.items()
    }

    # Save weights and frames
    with open(MASTER_JSON_PATH, "w") as f:
        json.dump({
            "frames": frame_data,
            "active_joints": list(point_motions.keys())
        }, f, indent=2)

    with open(WEIGHTS_JSON_PATH, "w") as f:
        json.dump({
            "joint_weights": joint_weights,
            "body_point_weights": body_point_weights
        }, f, indent=2)

    print(f"Saved: {MASTER_JSON_PATH}, {WEIGHTS_JSON_PATH}")

def extract_keypoints(landmarks, width, height, visibility_thresh=0.5):
    keypoints = {}
    for idx in get_relevant_indices():
        lm = landmarks[idx]
        if lm.visibility >= visibility_thresh:
            keypoints[idx] = [lm.x * width, lm.y * height]

    for required in [11, 12, 23, 24]:
        if required not in keypoints:
            print(f"[DEBUG] Keypoint {required} missing or low confidence (vis < {visibility_thresh})")
    return keypoints


def normalize_by_image_size(kps, image_width, image_height):
    return {
        k: [x / image_width, y / image_height]
        for k, (x, y) in kps.items()
    }

########################################################################################################################################################
def normalize_keypoints_by_body_scale(raw_kps):
    try:
        ls = np.array(raw_kps[11])
        rs = np.array(raw_kps[12])
        lh = np.array(raw_kps[23])
        rh = np.array(raw_kps[24])
    except KeyError as e:
        print("[WARN] Missing keypoint for normalization:", e)
        return None

    shoulder_center = (ls + rs) / 2
    hip_center = (lh + rh) / 2
    body_scale = np.linalg.norm(shoulder_center - hip_center)
    
    print(body_scale)
  

    if body_scale < 1e-5:
        print("[WARN] Body scale too small.")
        return None

    normalized = {}
    for k, (x, y) in raw_kps.items():
        rel = abs(np.array([x, y]) - shoulder_center)
        normalized[k] = (rel / body_scale).tolist()
    print(normalized)
 
    return normalized



def get_x_diff_from_nose(live_kps):
    nose_x = live_kps['0'][0]
    x_differences = {}

    for k, (x, y) in live_kps.items():
        x_diff = x - nose_x  # difference in x direction
        x_differences[k] = [x_diff, y]  # y remains same

    return x_differences

def get_y_diff_from_nose(live_kps):
    nose_y = live_kps['0'][1]
    y_differences = {}

    for k, (x, y) in live_kps.items():
        y_diff = y - nose_y 
        y_differences[k] = [x, y_diff] 

    return y_differences

########################################################################################################################################################
def live_compare_to_master():
    with open(MASTER_JSON_PATH, "r") as f:
        master_reference = json.load(f)

    with open(WEIGHTS_JSON_PATH, "r") as f:
        weights_data = json.load(f)

    joint_weights = weights_data.get("joint_weights", {})
    body_point_weights = weights_data.get("body_point_weights", {})

    cap_live = cv2.VideoCapture(0)#MASTER_VIDEO_PATH)
    cap_master = cv2.VideoCapture(MASTER_VIDEO_PATH)

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

    fps_live = cap_live.get(cv2.CAP_PROP_FPS)
    fps_master = cap_master.get(cv2.CAP_PROP_FPS)

    if fps_live == 0 or fps_master == 0:
        return

    skip_ratio = fps_live / (fps_master * 0.20)
    print(f"[INFO] FPS Live: {fps_live:.2f}, FPS Master: {fps_master:.2f}, Skip Ratio: {skip_ratio:.2f}") 

    frame_accumulator = 0.0
    frame_idx = 0

    accuracy_history = deque(maxlen=100)
    rep_count = 0
    rep_state = "down"


    angle_threshold = 40
    keypoint_threshold = 0.1  # Normalized distance threshold

    ret_master, frame_master = cap_master.read()  # Initial master frame
    ####################################for distance####################################################
    initial_positions = {}
    ################################Rep counting#######################################################
 

    UP_THRESHOLD = 0.85
    DOWN_THRESHOLD = 0.15

    # ✅ initialize state variables
    state = "DOWN"
    rep_count = 0
    rep_started = False   # <-- important

    ####################################################################################################
    while cap_live.isOpened() and cap_master.isOpened():
        ret_live, frame_live = cap_live.read()
        if not ret_live:
            break
        valid_pose_detected = False  # ← Add this here
        frame_accumulator += 1

        if frame_accumulator >= skip_ratio:
            ret_master, frame_master = cap_master.read()
            if not ret_master:
                cap_master.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                ret_master, frame_master = cap_master.read()

            frame_accumulator -= skip_ratio
            frame_idx += 1

 
        if frame_idx >= len(master_reference["frames"]):
            frame_idx = 0  
            cap_master.set(cv2.CAP_PROP_POS_FRAMES, 0)  


        frame_rgb = cv2.cvtColor(frame_live, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame_display = frame_live.copy()

        if results.pose_landmarks:
            draw_custom_skeleton(frame_display, results.pose_landmarks.landmark)
            live_angles = extract_key_angles(results.pose_landmarks.landmark)

            image_height, image_width = frame_display.shape[:2]

            raw_kps = extract_keypoints(results.pose_landmarks.landmark, image_width, image_height)
            live_kps = {
                str(k): [x / image_width, y / image_height] for k, (x, y) in raw_kps.items()
            }

            if live_kps is None:
                cv2.putText(frame_display, "⚠️ Stand fully in camera view", (50, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                combined_frame = np.hstack((
                    cv2.resize(frame_master, (frame_display.shape[1], frame_display.shape[0])),
                    frame_display
                ))
                combined_frame = cv2.resize(combined_frame, (1600, 960))
                cv2.imshow("Master (Left) | Live + Feedback (Right)", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            valid_pose_detected = True 
            ##########################################Distance 12 to 24 for live point ######################################
            dist_12_to_24_liv = None
            if '12' in live_kps and '24' in live_kps:
                x1, y1 = live_kps['12']
                x2, y2 = live_kps['24']
                
                # Euclidean distance between (x1, y1) and (x2, y2)
                dist_12_to_24_liv = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            #####################################################################################################

            master_frame = master_reference["frames"][frame_idx]
            master_angles = master_frame.get("angles", {})
            ##################################################### Angle accuracy ################################
            weighted_correct = 0
            total_weight = 0
            for joint in live_angles:
                master_val = master_angles.get(joint)
                if master_val is None:
                    continue
                weight = joint_weights.get(joint, 0)
                if weight == 0:
                    continue
                diff = abs(live_angles[joint] - master_val)
                score = max(0, 1 - (diff / angle_threshold))
                weighted_correct += score * weight
                total_weight += weight

            angle_accuracy = weighted_correct / total_weight if total_weight > 0 else 0
            #####################################################################################################
            
            master_kps = {str(k): v for k, v in master_frame.get("keypoints", {}).items()}
            #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            x_shift = master_kps['0'][0] - 0.5
            for kpid, (x, y) in master_kps.items():
                master_kps[kpid] = [x - x_shift, y]

            x_shifted_kps = get_x_diff_from_nose(live_kps)

            #print(x_shifted_kps)
            for k, (x_diff, _) in x_shifted_kps.items():
                if k in master_kps:
                    original_x, original_y = master_kps[k]
                    new_x = master_kps['0'][0] + x_diff
                    master_kps[k] = [new_x, original_y]


            #print(master_kps)
            y_shifted_kps = get_y_diff_from_nose(live_kps)
            #print("y_shifted_kps")
            #print(y_shifted_kps)
            for k, (_, y_diff) in y_shifted_kps.items():
                if k in master_kps:
                    original_x, original_y = master_kps[k]
                    new_y = master_kps['0'][1] + y_diff
                    master_kps[k] = [original_x, new_y]
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            ############################ AURA ##############################################
            #--- Draw master "aura" over the live display for alignment guidance ---
            for k_str, (mx, my) in master_kps.items():
                x = int(mx * frame_display.shape[1])
                y = int(my * frame_display.shape[0])
                cv2.circle(frame_display, (x, y), 6, (0, 0, 255), 2)  # Light magenta aura point
                

            # Optional: Draw connections like skeleton
            master_connections = [
                (11, 13), (13, 15),  # left arm
                (12, 14), (14, 16),  # right arm
                (11, 12),            # shoulders
                (11, 23), (12, 24),  # upper body
                (23, 24),            # hips
                (23, 25), (25, 27),  # left leg
                (24, 26), (26, 28),  # right leg
            ]

            for idx1, idx2 in master_connections:
                str1, str2 = str(idx1), str(idx2)
                if str1 in master_kps and str2 in master_kps:
                    x1, y1 = int(master_kps[str1][0] * frame_display.shape[1]), int(master_kps[str1][1] * frame_display.shape[0])
                    x2, y2 = int(master_kps[str2][0] * frame_display.shape[1]), int(master_kps[str2][1] * frame_display.shape[0])
                    cv2.line(frame_display, (x1, y1), (x2, y2), (0, 255, 2), 3, cv2.LINE_AA)  # aura line
            ############################################# Distance calculation #######################################################
            joint_weights_path = WEIGHTS_JSON_PATH
            reference_json_path = MASTER_JSON_PATH 
            top_kps = get_top_keypoints(joint_weights_path, top_n=2)

            major_point = analyze_movement(reference_json_path,top_kps)
            pprint.pprint(major_point)
            results = []

            for k, v in major_point.items():
                if "index" in v:
                    results.append((v["index"], v["max_distance"]))
                elif k == "shoulder_hip_distance":
                    results.append((f'{v["from"]}_{v["to"]}_avg_master', v["12_24_avg_distance_master"]))

            pprint.pprint(results)
            pprint.pprint(dist_12_to_24_liv)


            
            data = results
            

            
            master_avg = next(v for k, v in data if k == "12_24_avg_master")

            # Build dictionary with scaled values
            distance_to_cover = {
                k: (v * dist_12_to_24_liv) / master_avg
                for k, v in data if isinstance(k, int)
            }
            print(distance_to_cover)
            
            max_distances = distance_to_cover  # normalized distances like 0.58
            h, w = frame_display.shape[:2] 
            accuracies = []
            for kp_id in max_distances.keys():
                kp_str = str(kp_id).replace("_liv", "")

                if kp_str in live_kps:
                    x, y = live_kps[kp_str]  # normalized coords from Mediapipe

                    # If no initial position, set it now
                    if kp_id not in initial_positions:
                        initial_positions[kp_id] = (x, y)

                    x0, y0 = initial_positions[kp_id]
                    px    = int(x0*w)
                    py    = int(y0*h)
                    print("*********************************************************************************")
                    print(x0,y0)
                    cv2.circle(frame_display, (px, py), 19, (0, 0, 255), -1)
                    print(x,y)
                    print("*********************************************************************************")
                    # Euclidean distance in normalized coordinates
                    current_dist = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)

                    # Calculate progress percentage
                    progress = (current_dist / max_distances[kp_id]) * 100
                    progress = max(0, min(progress, 100))
                    accuracies.append(progress)

                    
                    # Convert normalized coords to pixel coords
                    px, py = int(x * image_width), int(y * image_height)

                    # Show progress on image
                    cv2.putText(
                        frame_display,
                        f"{kp_str}: {progress:.2f}%",
                        (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    print(f"Keypoint {kp_str} progress: {progress:.2f}%")
            

            # ---- Average accuracy calculation ----
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies) / 100.0  # scale to 0–1
                # inside your loop
                if avg_accuracy < DOWN_THRESHOLD and not rep_started:
                    # ✅ went fully down, mark as start of rep
                    rep_started = True  
                    cv2.putText(frame_display, f"DOWN detected | rep_started={rep_started}", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)

                elif avg_accuracy > UP_THRESHOLD and rep_started:
                    # ✅ only count when it goes fully up after full down
                    rep_count += 1
                    rep_started = False
                    cv2.putText(frame_display, f"DOWN detected | rep_started={rep_started}", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.putText(frame_display, f"Reps: {rep_count}", 
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0), 2, cv2.LINE_AA)

                # Bar position & size
                bar_x = 20
                bar_y = 50
                bar_width = 30
                bar_height = 200

                # Draw bar outline
                cv2.rectangle(frame_display, (bar_x, bar_y),
                            (bar_x + bar_width, bar_y + bar_height),
                            (200, 200, 200), 2)

                # Fill height based on average accuracy
                filled_height = int(bar_height * avg_accuracy)
                fill_top = bar_y + bar_height - filled_height

                # Color based on accuracy
                bar_color = (0, 255, 0) if avg_accuracy >= 0.8 else (0, 0, 255)

                # Draw filled bar
                cv2.rectangle(frame_display, (bar_x, fill_top),
                            (bar_x + bar_width, bar_y + bar_height),
                            bar_color, -1)

                # Label
                cv2.putText(frame_display, f"{avg_accuracy*100:.1f}%",
                            (bar_x - 5, bar_y + bar_height + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            if angle_accuracy < 0.5:
                cv2.putText(
                    frame_display,
                    "Please perform as per master video",
                    (50, 100),  # (x,y) position
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,            # font scale
                    (0, 0, 255),    # red color
                    3,              # thickness
                    cv2.LINE_AA
                )

            
            cv2.putText(frame_display, f"Skip_ratio: {skip_ratio}", (500, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 9, 255), 1)

            cv2.putText(frame_display, f"angle_accuracy: {angle_accuracy}", (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 9, 2), 3)
            cv2.line(frame_display, (image_width // 2, 0), (image_width // 2, image_height), (255, 255, 0), 2)

            
        # Show Master & Live Frames Side-by-Side
        frame_master_resized = cv2.resize(frame_master, (frame_display.shape[1], frame_display.shape[0]))

        combined_frame = np.hstack((frame_master_resized, frame_display))
        combined_frame = cv2.resize(combined_frame,(1600,960))
        cv2.imshow("Master (Left) | Live + Feedback (Right)", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_live.release()
    cap_master.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preprocess", "compare"], required=True, help="Mode to run: preprocess or compare")
    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess_master_video()
    elif args.mode == "compare":
        live_compare_to_master()