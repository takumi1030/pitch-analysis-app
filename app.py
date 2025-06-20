# --- app.py ---
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.signal import butter, filtfilt
import tempfile

# ------------------------------
# Helper Functions
# ------------------------------
def normalize_curve(data_series, num_points=101):
    if len(data_series) < 2: return np.zeros(num_points)
    current_x = np.linspace(0, 100, len(data_series))
    new_x = np.linspace(0, 100, num_points)
    return np.interp(new_x, current_x, data_series)

def lowpass_filter(data, cutoff, fs, order=4):
    data_np = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1 or len(data_np) <= 3 * (order + 1): return data_np
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data_np)

def calculate_plane_angular_velocity(p1_prev, p2_prev, p1_curr, p2_curr, dt):
    v_prev = np.array([p2_prev[0] - p1_prev[0], p2_prev[2] - p1_prev[2]])
    v_curr = np.array([p2_curr[0] - p1_curr[0], p2_curr[2] - p1_curr[2]])
    angle_prev = np.arctan2(v_prev[1], v_prev[0])
    angle_curr = np.arctan2(v_curr[1], v_curr[0])
    delta_angle = angle_curr - angle_prev
    if delta_angle > np.pi: delta_angle -= 2 * np.pi
    elif delta_angle < -np.pi: delta_angle += 2 * np.pi
    return np.rad2deg(delta_angle) / dt

def calculate_elbow_extension_velocity(shoulder_coords, elbow_coords, wrist_coords, dt):
    v1 = shoulder_coords - elbow_coords
    v2 = wrist_coords - elbow_coords
    dot_product = np.einsum('ij,ij->j', v1, v2)
    norm_v1 = np.linalg.norm(v1, axis=0)
    norm_v2 = np.linalg.norm(v2, axis=0)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    angles_deg = np.rad2deg(angles_rad)
    angular_velocity = np.diff(angles_deg) / dt
    return np.abs(angular_velocity)

# ------------------------------
# Streamlit App
# ------------------------------
st.title("🎥 投球動作解析アプリ (Streamlit版)")
st.write("スマホ動画をアップロードして、骨盤・胸郭・肩・肘の角速度を自動解析します。")

side = st.radio("投球手の利き手を選んでください:", ('右投げ (R)', '左投げ (L)'))
side = 'R' if 'R' in side else 'L'

uploaded_file = st.file_uploader("投球動画をアップロードしてください (MP4推奨)", type=['mp4', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    filename = tfile.name

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1 / fps if fps > 0 else 1/30.0

    all_landmarks_data = []
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_world_landmarks:
            all_landmarks_data.append(results.pose_world_landmarks.landmark)
    cap.release()
    pose.close()

    if len(all_landmarks_data) > 40:
        num_landmarks = len(all_landmarks_data[0])
        raw_coords = np.array([[[lm.x, lm.y, lm.z] for lm in frame] for frame in all_landmarks_data]).transpose((1,2,0))
        filtered_coords = np.zeros_like(raw_coords)
        for i in range(num_landmarks):
            for j in range(3):
                filtered_coords[i,j,:] = lowpass_filter(raw_coords[i,j,:], 10, fps)

        stride_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE if side == "R" else mp_pose.PoseLandmark.RIGHT_KNEE
        throwing_hand_idx = mp_pose.PoseLandmark.RIGHT_INDEX if side == "R" else mp_pose.PoseLandmark.LEFT_INDEX
        start_frame = np.argmin(filtered_coords[stride_knee_idx, 1, :])
        hand_velocity = np.diff(filtered_coords[throwing_hand_idx, 0, :]) / dt
        release_frame = start_frame + np.argmax(np.abs(hand_velocity[start_frame:]))

        segmented_coords = filtered_coords[:,:,start_frame:release_frame]
        num_segmented_frames = segmented_coords.shape[2]
        indices = {
            'pelvis_l': mp_pose.PoseLandmark.LEFT_HIP,
            'pelvis_r': mp_pose.PoseLandmark.RIGHT_HIP,
            'thorax_l': mp_pose.PoseLandmark.LEFT_SHOULDER,
            'thorax_r': mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER if side=="R" else mp_pose.PoseLandmark.LEFT_SHOULDER,
            'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW if side=="R" else mp_pose.PoseLandmark.LEFT_ELBOW,
            'wrist': mp_pose.PoseLandmark.RIGHT_WRIST if side=="R" else mp_pose.PoseLandmark.LEFT_WRIST
        }

        pelvis_vel = [calculate_plane_angular_velocity(segmented_coords[indices['pelvis_l'],:,i-1], segmented_coords[indices['pelvis_r'],:,i-1],
                                                       segmented_coords[indices['pelvis_l'],:,i], segmented_coords[indices['pelvis_r'],:,i], dt)
                      for i in range(1, num_segmented_frames)]
        thorax_vel = [calculate_plane_angular_velocity(segmented_coords[indices['thorax_l'],:,i-1], segmented_coords[indices['thorax_r'],:,i-1],
                                                       segmented_coords[indices['thorax_l'],:,i], segmented_coords[indices['thorax_r'],:,i], dt)
                      for i in range(1, num_segmented_frames)]
        shoulder_vel = [calculate_plane_angular_velocity(segmented_coords[indices['shoulder'],:,i-1], segmented_coords[indices['elbow'],:,i-1],
                                                         segmented_coords[indices['shoulder'],:,i], segmented_coords[indices['elbow'],:,i], dt)
                        for i in range(1, num_segmented_frames)]
        elbow_vel = calculate_elbow_extension_velocity(segmented_coords[indices['shoulder']],
                                                       segmented_coords[indices['elbow']],
                                                       segmented_coords[indices['wrist']], dt)

        normalized_time = np.linspace(0, 100, 101)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(normalized_time, normalize_curve(np.abs(pelvis_vel)), label='骨盤 角速度')
        ax.plot(normalized_time, normalize_curve(np.abs(thorax_vel)), label='胸郭 角速度')
        ax.plot(normalized_time, normalize_curve(np.abs(shoulder_vel)), label='肩(上腕) 角速度')
        ax.plot(normalized_time, normalize_curve(np.abs(elbow_vel)), label='肘(前腕) 伸展速度')
        ax.set_title("スマホ映像から自動生成した運動連鎖グラフ")
        ax.set_xlabel("正規化時間 (%)")
        ax.set_ylabel("角速度 (deg/s)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("十分なフレームが検出できませんでした。")
