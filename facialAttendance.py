import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import pickle
import mediapipe as mp
import time

SKIP_BLINK_FRAMES = 2
SKIP_RECOG_FRAMES = 10
PROCESS_SCALE = 0.5

TRAINING_SKIP = 10

MAX_SAMPLES = 20

mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

svm_model = SVC(kernel='linear', probability=True, C=1.0)
scaler = StandardScaler()

try:
    lbph_model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=5, grid_y=5)
except AttributeError:
    print("[ERROR] Install 'opencv-contrib-python'!")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


Combined_Features = []
Labels = []
Name_map = {}
current_label_id = 0
is_trained = False
marked_present = set()

blink_state = {"has_blinked": False, "last_face_id": -1}
frame_count = 0
cache = {
    "name": "Initializing...",
    "color": (255, 0, 0),
    "id": -1,
    "box": None,
    "landmarks": None
}


def save_system():
    with open('hybrid_system.pkl', 'wb') as f:
        pickle.dump({
            'Features': Combined_Features,
            'Labels': Labels,
            'Map': Name_map,
            'SVM': svm_model,
            'Scaler': scaler,
            'Id': current_label_id
        }, f)
    lbph_model.write('lbph_brain.yml')
    print("[SYSTEM] Saved successfully.")


def load_system():
    global Combined_Features, Labels, Name_map, svm_model, current_label_id, is_trained, scaler
    if os.path.exists('hybrid_system.pkl'):
        try:
            with open('hybrid_system.pkl', 'rb') as f:
                data = pickle.load(f)
                Combined_Features = data['Features']
                Labels = data['Labels']
                Name_map = data['Map']
                svm_model = data['SVM']
                scaler = data['Scaler']
                current_label_id = data['Id']
        except:
            pass

    if os.path.exists('lbph_brain.yml'):
        lbph_model.read('lbph_brain.yml')

    if len(Labels) > 0:
        print(f"[SYSTEM] Loaded: {len(Name_map)} people ({len(Labels)} samples).")
        is_trained = True


def load_today_attendance():
    if not os.path.exists("Attendance.csv"): return
    today_str = datetime.now().strftime("%d/%m/%Y")
    with open("Attendance.csv", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[1] == today_str:
                marked_present.add(parts[0])


def get_landmarks_only(image_rgb):
    mesh_results = face_mesh_detector.process(image_rgb)
    if mesh_results.multi_face_landmarks:
        return mesh_results.multi_face_landmarks[0]
    return None


def get_features_from_landmarks(image_gray, landmarks_obj):
    img_eq = cv2.equalizeHist(image_gray)
    resized_img = cv2.resize(img_eq, (64, 128))

    hog_feat = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')

    landmark_vec = []
    if landmarks_obj:
        for lm in landmarks_obj.landmark:
            landmark_vec.extend([lm.x, lm.y, lm.z])
    else:
        landmark_vec = [0] * (468 * 3)

    return np.hstack([hog_feat, np.array(landmark_vec)])


def mark_attendance(name):
    if name in marked_present: return
    now = datetime.now()
    Time = now.strftime("%H:%M:%S")
    Date = now.strftime("%d/%m/%Y")
    with open("Attendance.csv", "a") as csvfile:
        csvfile.write(f'{name},{Date},{Time}\n')
    marked_present.add(name)
    print(f"[LOGGED] {name}")


def train_hybrid(new_raw_faces, new_labels, new_features):
    global is_trained, svm_model, scaler, Combined_Features, Labels
    print(f"[TRAINING] Processing {len(new_labels)} new samples...")

    lbph_model.update(new_raw_faces, np.array(new_labels))
    Combined_Features.extend(new_features)
    Labels.extend(new_labels)

    X_train = np.array(Combined_Features)
    Y_train = np.array(Labels)

    if len(set(Labels)) == 1:
        feature_len = len(X_train[0])
        noise_vectors = [np.random.rand(feature_len) for _ in range(5)]
        X_combined = np.vstack([X_train, noise_vectors])
        Y_combined = np.concatenate([Y_train, [-1] * 5])
        X_scaled = scaler.fit_transform(X_combined)
        svm_model.fit(X_scaled, Y_combined)
    else:
        X_scaled = scaler.fit_transform(X_train)
        svm_model.fit(X_scaled, Y_train)

    is_trained = True
    save_system()
    print("[SYSTEM] Training Complete.")


def check_blink(landmarks):
    if landmarks is None: return False
    left_dist = abs(landmarks.landmark[145].y - landmarks.landmark[159].y)
    right_dist = abs(landmarks.landmark[374].y - landmarks.landmark[386].y)
    if left_dist < 0.012 and right_dist < 0.012: return True
    return False

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not os.path.exists('Attendance.csv'):
    with open('Attendance.csv', 'w') as f: f.write('Name,Date,Time\n')

load_system()
load_today_attendance()

print("--- OPTIMIZED HYBRID SYSTEM ---")
print("1. Detection Mode")
print("2. Press 'r' to Register")

if len(Name_map) == 0:
    current_name_input = input("Enter Name for First Person: ")
else:
    current_name_input = "Unknown"
    if len(Name_map) > 0: current_label_id = max(Name_map.keys())

while True:
    success, img = cap.read()
    if not success: break
    frame_count += 1

    small_img = cv2.resize(img, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
    small_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    small_faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=5)

    if len(small_faces) == 0:
        blink_state["has_blinked"] = False
        blink_state["last_face_id"] = -1
        cache["box"] = None
        cache["landmarks"] = None

    for (sx, sy, sw, sh) in small_faces:
        scale_factor = 1 / PROCESS_SCALE
        x = int(sx * scale_factor)
        y = int(sy * scale_factor)
        w = int(sw * scale_factor)
        h = int(sh * scale_factor)
        cache["box"] = (x, y, w, h)

        p_small = int(20 * PROCESS_SCALE)
        sx_b, sy_b = max(0, sx - p_small), max(0, sy - p_small)
        sw_b, sh_b = min(small_img.shape[1] - sx_b, sw + 2 * p_small), min(small_img.shape[0] - sy_b, sh + 2 * p_small)

        face_roi_small = small_gray[sy_b:sy_b + sh_b, sx_b:sx_b + sw_b]
        face_roi_rgb_small = cv2.cvtColor(small_img[sy_b:sy_b + sh_b, sx_b:sx_b + sw_b], cv2.COLOR_BGR2RGB)

        if face_roi_small.size > 0:
            if frame_count % SKIP_BLINK_FRAMES == 0:
                landmarks = get_landmarks_only(face_roi_rgb_small)
                cache["landmarks"] = landmarks
                if check_blink(landmarks):
                    blink_state["has_blinked"] = True
                    cache["color"] = (0, 255, 0) if cache["name"] != "Unknown" else (0, 0, 255)

            if is_trained and (frame_count % SKIP_RECOG_FRAMES == 0 or blink_state["last_face_id"] == -1):
                lm_to_use = cache["landmarks"] if cache["landmarks"] else get_landmarks_only(face_roi_rgb_small)
                features = get_features_from_landmarks(face_roi_small, lm_to_use)

                try:
                    feat_scaled = scaler.transform(features.reshape(1, -1))
                    svm_id = svm_model.predict(feat_scaled)[0]
                except:
                    svm_id = -1

                lbph_id, lbph_dist = lbph_model.predict(cv2.equalizeHist(face_roi_small))
                final_id = svm_id if (svm_id == lbph_id and svm_id != -1) else lbph_id

                if final_id != blink_state["last_face_id"]:
                    blink_state["has_blinked"] = False
                    blink_state["last_face_id"] = final_id

                if final_id != -1 and lbph_dist < 80:
                    name = Name_map.get(final_id, "Unknown")
                    if blink_state["has_blinked"]:
                        cache["name"] = f"GRANTED: {name}"
                        cache["color"] = (0, 255, 0)
                        mark_attendance(name)
                    else:
                        cache["name"] = "BLINK NOW"
                        cache["color"] = (0, 255, 255)
                else:
                    cache["name"] = "Unknown"
                    cache["color"] = (0, 0, 255)

    if cache["box"] is not None:
        (x, y, w, h) = cache["box"]
        p = 40
        cv2.rectangle(img, (x - p, y - p), (x + w + p, y + h + p), cache["color"], 2)
        cv2.putText(img, cache["name"], (x - p, y - p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cache["color"], 2)

    cv2.imshow("System", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        session_raw_faces = []
        session_labels = []
        session_features = []
        start_time = time.time()
        capture_frame_idx = 0

        print(f"[CAPTURE] Starting for {current_name_input}...")

        while (time.time() - start_time) < 10 and len(session_raw_faces) < MAX_SAMPLES:
            success, raw_img = cap.read()
            if not success: break
            capture_frame_idx += 1

            s_img = cv2.resize(raw_img, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
            s_gray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(s_gray, 1.2, 5)

            if len(faces) == 1:
                (sx, sy, sw, sh) = faces[0]
                p = int(20 * PROCESS_SCALE)
                sx_b, sy_b = max(0, sx - p), max(0, sy - p)
                sw_b, sh_b = min(s_img.shape[1] - sx_b, sw + 2 * p), min(s_img.shape[0] - sy_b, sh + 2 * p)

                # [FILE SIZE FIX] Only save 1 out of 10 frames
                if capture_frame_idx % TRAINING_SKIP == 0:
                    roi = s_gray[sy_b:sy_b + sh_b, sx_b:sx_b + sw_b]
                    roi_rgb = cv2.cvtColor(s_img[sy_b:sy_b + sh_b, sx_b:sx_b + sw_b], cv2.COLOR_BGR2RGB)

                    if roi.size > 0:
                        session_raw_faces.append(roi)
                        session_labels.append(current_label_id)
                        lm = get_landmarks_only(roi_rgb)
                        feats = get_features_from_landmarks(roi, lm)
                        session_features.append(feats)
                        cv2.rectangle(raw_img, (0, 0), (640, 480), (255, 255, 255), 5)

                x, y, w, h = int(sx / PROCESS_SCALE), int(sy / PROCESS_SCALE), int(sw / PROCESS_SCALE), int(
                    sh / PROCESS_SCALE)
                cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

                cv2.putText(raw_img, f"Captured: {len(session_raw_faces)}/{MAX_SAMPLES}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("System", raw_img)
                cv2.waitKey(30)
            else:
                cv2.imshow("System", raw_img)
                cv2.waitKey(1)

        Name_map[current_label_id] = current_name_input
        train_hybrid(session_raw_faces, session_labels, session_features)

    if key == ord('n'):
        current_label_id += 1
        cv2.destroyAllWindows()
        current_name_input = input(f"Enter Name for Person ID {current_label_id}: ")
        print(f"Ready. Press 'r' to register.")

    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()
