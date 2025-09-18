
# import json

# # MediaPipe keypoint index mapping
# KEYPOINT_INDEX = {
#     "nose": 0,
#     "left_eye_inner": 1,
#     "left_eye": 2,
#     "left_eye_outer": 3,
#     "right_eye_inner": 4,
#     "right_eye": 5,
#     "right_eye_outer": 6,
#     "left_ear": 7,
#     "right_ear": 8,
#     "mouth_left": 9,
#     "mouth_right": 10,
#     "left_shoulder": 11,
#     "right_shoulder": 12,
#     "left_elbow": 13,
#     "right_elbow": 14,
#     "left_wrist": 15,
#     "right_wrist": 16,
#     "left_pinky": 17,
#     "right_pinky": 18,
#     "left_index": 19,
#     "right_index": 20,
#     "left_thumb": 21,
#     "right_thumb": 22,
#     "left_hip": 23,
#     "right_hip": 24,
#     "left_knee": 25,
#     "right_knee": 26,
#     "left_ankle": 27,
#     "right_ankle": 28,
#     "left_heel": 29,
#     "right_heel": 30,
#     "left_foot_index": 31,
#     "right_foot_index": 32,
# }

# def get_top_weighted_keypoints(joint_weights_path, top_n=4):
#     with open(joint_weights_path, 'r') as f:
#         weights_data = json.load(f)

#     body_weights = weights_data["body_point_weights"]

#     # Sort and get top N keypoints
#     top_points = sorted(body_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]

#     # Return keypoint name and corresponding index
#     result = {name: KEYPOINT_INDEX[name] for name, _ in top_points}
#     return result


# # Usage
# result = get_top_weighted_keypoints("joint_weights.json")
# print(result)



import json
import math


# MediaPipe keypoint index mapping
KEYPOINT_INDEX = {
    "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20, "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_top_keypoints(joint_weights_path, top_n=4):
    with open(joint_weights_path, 'r') as f:
        weights = json.load(f)
    body_weights = weights["body_point_weights"]
    top = sorted(body_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {name: KEYPOINT_INDEX[name] for name, _ in top}


# def analyze_movement(reference_json_path, top_kp_dict):
#     with open(reference_json_path, 'r') as f:
#         reference_data = json.load(f)

#     frames = reference_data['frames']
#     movement_stats = {}

#     for kp_name, kp_index in top_kp_dict.items():
#         key = str(kp_index)
#         coords = []

#         for frame in frames:
#             if key in frame['keypoints']:
#                 coords.append(tuple(frame['keypoints'][key]))

#         max_dist = -1
#         point1 = point2 = ()

#         for i in range(len(coords)):
#             for j in range(i+1, len(coords)):
#                 dist = euclidean(coords[i], coords[j])
#                 if dist > max_dist:
#                     max_dist = dist
#                     point1, point2 = coords[i], coords[j]

#         movement_stats[kp_name] = {
#             "index": kp_index,
#             "frame_1_coord": point1,
#             "frame_2_coord": point2,
#             "max_distance": max_dist
#         }

#     return movement_stats
def analyze_movement(reference_json_path, top_kp_dict):
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)

    frames = reference_data['frames']
    movement_stats = {}

    for kp_name, kp_index in top_kp_dict.items():
        key = str(kp_index)
        coords = []

        for frame in frames:
            if key in frame['keypoints']:
                coords.append(tuple(frame['keypoints'][key]))

        max_dist = -1
        point1 = point2 = ()

        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = euclidean(coords[i], coords[j])
                if dist > max_dist:
                    max_dist = dist
                    point1, point2 = coords[i], coords[j]

        movement_stats[kp_name] = {
            "index": kp_index,
            "frame_1_coord": point1,
            "frame_2_coord": point2,
            "max_distance": max_dist
        }

    # 👉 Additional computation: shoulder (12) to hip (24) distance across frames
    shoulder_hip_distances = []
    for frame in frames:
        if "12" in frame['keypoints'] and "24" in frame['keypoints']:
            p1 = tuple(frame['keypoints']["12"])
            p2 = tuple(frame['keypoints']["24"])
            dist = euclidean(p1, p2)
            shoulder_hip_distances.append(dist)

    if shoulder_hip_distances:
        movement_stats["shoulder_hip_distance"] = {
            "from": 12,
            "to": 24,
            "min_distance": min(shoulder_hip_distances),
            "max_distance": max(shoulder_hip_distances),
            "12_24_avg_distance_master": sum(shoulder_hip_distances) / len(shoulder_hip_distances)
        }

    return movement_stats
# --- USAGE ---
joint_weights_path = "/home/sidharth/Documents/Siddharth_office/gym_assistant /voice_enable_ai_trainer-main/leg/joint_weights.json"
reference_json_path = "/home/sidharth/Documents/Siddharth_office/gym_assistant /voice_enable_ai_trainer-main/leg/master_reference.json"

top_kps = get_top_keypoints(joint_weights_path, top_n=4)
#print(top_kps)
result = analyze_movement(reference_json_path, top_kps)

# Display nicely
import pprint
pprint.pprint(result)