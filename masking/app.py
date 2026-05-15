import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
from streamlit_image_coordinates import streamlit_image_coordinates
import threading

# -----------------------------
# CONFIG
# -----------------------------
CONF_THRESH = 0.5        
IOU_THRESH = 0.5         
CAMERA_MOVE_THRESH = 50  
FRAME_WIDTH = 640        
FRAME_HEIGHT = 360       
HIST_MATCH_THRESH = 0.45 

# Set page config for mobile friendly
st.set_page_config(page_title="AI Masking App", page_icon="🕵️", layout="centered")

# Initialize YOLO model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n-seg.pt")
    import torch
    if torch.cuda.is_available(): 
        model.to("cuda")
    else:
        model.to("cpu")
    return model

model = load_model()

# -----------------------------
# SESSION STATE MEMORY
# -----------------------------
if "manual_mask_profiles" not in st.session_state:
    st.session_state.manual_mask_profiles = []
if "manual_unmask_profiles" not in st.session_state:
    st.session_state.manual_unmask_profiles = []
if "snapshot_frame" not in st.session_state:
    st.session_state.snapshot_frame = None
if "snapshot_persons" not in st.session_state:
    st.session_state.snapshot_persons = None
if "snapshot_main_person" not in st.session_state:
    st.session_state.snapshot_main_person = None

# -----------------------------
# HELPERS (From Primitive Code)
# -----------------------------
def camera_moved(prev_frame, curr_frame, threshold=CAMERA_MOVE_THRESH):
    if prev_frame is None or curr_frame is None: return False
    if prev_frame.shape != curr_frame.shape:
        curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
    
    border_size = 10
    p_b = np.concatenate([prev_frame[:border_size, :].flatten(), prev_frame[-border_size:, :].flatten(), prev_frame[:, :border_size].flatten(), prev_frame[:, -border_size:].flatten()])
    c_b = np.concatenate([curr_frame[:border_size, :].flatten(), curr_frame[-border_size:, :].flatten(), curr_frame[:, :border_size].flatten(), curr_frame[:, -border_size:].flatten()])
    return np.mean(np.abs(p_b.astype(np.int16) - c_b.astype(np.int16))) > threshold

def generate_background(frame, mask):
    return cv2.inpaint(frame, (mask * 255).astype(np.uint8), 3, cv2.INPAINT_NS)

def extract_histogram(frame, mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    calc_mask = (mask * 255).astype(np.uint8)
    if cv2.countNonZero(calc_mask) == 0: return None
    hist = cv2.calcHist([hsv], [0, 1], calc_mask, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def match_profile(hist, profiles, threshold=HIST_MATCH_THRESH):
    if hist is None or not profiles: return -1
    best_score, best_idx = float('inf'), -1
    for i, p in enumerate(profiles):
        score = cv2.compareHist(hist, p, cv2.HISTCMP_BHATTACHARYYA)
        if score < best_score: best_score, best_idx = score, i
    return best_idx if best_score < threshold else -1

def detect_persons_and_masks(frame, model):
    results = model.predict(frame, conf=CONF_THRESH, classes=[0], imgsz=320, verbose=False)[0]
    persons = []
    if results.masks is None: return persons
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    masks = results.masks.data.cpu().numpy()
    h, w = frame.shape[:2]
    
    for box, mask in zip(boxes, masks):
        mask_bin = cv2.dilate((cv2.resize(mask, (w, h)) > 0.5).astype(np.uint8), np.ones((11,11), np.uint8), iterations=2)
        persons.append(((int(box[0]), int(box[1]), int(box[2]), int(box[3])), mask_bin))
    return persons

# -----------------------------
# VIDEO PROCESSOR
# -----------------------------
class MaskingProcessor(VideoTransformerBase):
    def __init__(self):
        self.prev_frame = None
        self.background = None
        self.frame_count = 0
        self.cached_persons = []
        
        # We'll copy state here so it can be accessed in the thread
        self.manual_mask_profiles = []
        self.manual_unmask_profiles = []
        self.latest_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        original_shape = img.shape
        proc_frame = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Save latest frame for snapshot capture
        self.latest_frame = proc_frame.copy()
        
        if self.prev_frame is None:
            self.prev_frame = proc_frame.copy()
            self.background = proc_frame.copy()

        # 1. Run YOLO (skip frames for cloud smoothness)
        self.frame_count += 1
        if self.frame_count % 3 == 0 or not self.cached_persons:
            self.cached_persons = detect_persons_and_masks(proc_frame, model)
        
        persons_now = self.cached_persons

        # 2. Build current all mask
        current_all_mask = np.zeros(proc_frame.shape[:2], dtype=np.uint8)
        for p in persons_now: 
            current_all_mask = cv2.bitwise_or(current_all_mask, cv2.dilate(p[1], np.ones((15, 15), np.uint8), iterations=1))

        # 3. Background Update
        if camera_moved(self.prev_frame, proc_frame):
            self.background = generate_background(proc_frame, current_all_mask) if np.any(current_all_mask) else proc_frame.copy()
        else:
            update_mask = (current_all_mask == 0)[..., None]
            self.background = np.where(update_mask, proc_frame, self.background)

        self.prev_frame = proc_frame.copy()

        # 4. Identify Main Person
        main_person = None
        if persons_now:
            main_person = max(persons_now, key=lambda p: (p[0][2]-p[0][0]) * (p[0][3]-p[0][1]))

        # 5. Apply Masks Based on Memory (Primitive Code Logic)
        unknown_mask = np.zeros(proc_frame.shape[:2], dtype=np.uint8)

        for person in persons_now:
            hist = extract_histogram(proc_frame, person[1])
            is_main = (main_person and person[0] == main_person[0])
            should_mask = False
            
            if is_main:
                if match_profile(hist, self.manual_mask_profiles) != -1:
                    should_mask = True
            else:
                if match_profile(hist, self.manual_unmask_profiles) == -1:
                    should_mask = True

            if should_mask:
                unknown_mask = cv2.bitwise_or(unknown_mask, person[1])
            elif hist is not None:
                # Update histogram memory slightly to handle real-time lighting changes
                target_list = self.manual_mask_profiles if is_main else self.manual_unmask_profiles
                idx = match_profile(hist, target_list)
                if idx != -1:
                    cv2.addWeighted(target_list[idx], 0.9, hist, 0.1, 0, target_list[idx])

        # 6. Smooth Blending
        if np.any(unknown_mask):
            alpha = np.expand_dims(cv2.GaussianBlur(unknown_mask * 255, (21,21), 0).astype(float) / 255.0, axis=2)
            proc_frame = (proc_frame * (1 - alpha) + self.background * alpha).astype(np.uint8)

        output_frame = cv2.resize(proc_frame, (original_shape[1], original_shape[0]))
        return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🕵️ Real-Time AI Masking App")
st.markdown("Your original memory logic is active! By default, background people are masked. Take a snapshot below to click on a person and toggle their mask state.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="masking-app",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=MaskingProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480, "max": 640},
            "height": {"ideal": 270, "max": 360},
            "frameRate": {"ideal": 15, "max": 30}
        },
        "audio": False
    },
    async_processing=True,
)

# Update WebRTC memory profiles from session state
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.manual_mask_profiles = st.session_state.manual_mask_profiles
    webrtc_ctx.video_processor.manual_unmask_profiles = st.session_state.manual_unmask_profiles

st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("📸 Take Snapshot to Click Person", use_container_width=True):
        if webrtc_ctx.video_processor and webrtc_ctx.video_processor.latest_frame is not None:
            frame = webrtc_ctx.video_processor.latest_frame
            st.session_state.snapshot_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Re-run YOLO for the snapshot strictly to get accurate click bounds
            persons = detect_persons_and_masks(frame, model)
            st.session_state.snapshot_persons = persons
            if persons:
                st.session_state.snapshot_main_person = max(persons, key=lambda p: (p[0][2]-p[0][0]) * (p[0][3]-p[0][1]))
            else:
                st.session_state.snapshot_main_person = None

with col2:
    if st.button("🗑️ Clear Memory", use_container_width=True):
        st.session_state.manual_mask_profiles = []
        st.session_state.manual_unmask_profiles = []
        st.session_state.snapshot_frame = None
        st.success("Memory cleared! Reverted to default behavior.")

# Handle the click interaction on the snapshot
if st.session_state.snapshot_frame is not None:
    st.write("👆 **Click on a person below to toggle their mask status.**")
    value = streamlit_image_coordinates(st.session_state.snapshot_frame, key="snapshot_click")
    
    if value is not None:
        cx, cy = value["x"], value["y"]
        frame_bgr = cv2.cvtColor(st.session_state.snapshot_frame, cv2.COLOR_RGB2BGR)
        
        # Iterate over persons in snapshot to see who was clicked
        clicked_someone = False
        for p in st.session_state.snapshot_persons:
            if cy < p[1].shape[0] and cx < p[1].shape[1] and p[1][cy, cx] > 0:
                clicked_someone = True
                hist = extract_histogram(frame_bgr, p[1])
                if hist is not None:
                    is_main = (st.session_state.snapshot_main_person and p[0] == st.session_state.snapshot_main_person[0])
                    
                    if is_main:
                        idx = match_profile(hist, st.session_state.manual_mask_profiles)
                        if idx != -1: 
                            st.session_state.manual_mask_profiles.pop(idx) # Unmask them
                            st.success("Unmasked Main Person!")
                        else: 
                            st.session_state.manual_mask_profiles.append(hist) # Mask them
                            st.success("Masked Main Person!")
                    else:
                        idx = match_profile(hist, st.session_state.manual_unmask_profiles)
                        if idx != -1: 
                            st.session_state.manual_unmask_profiles.pop(idx) # Mask them
                            st.success("Masked Background Person!")
                        else: 
                            st.session_state.manual_unmask_profiles.append(hist) # Unmask them
                            st.success("Unmasked Background Person!")
                break
        
        if not clicked_someone:
            st.warning("You clicked outside a person's bounds. Try again!")

st.info("💡 **Cloud Optimized:** The model runs at a reduced resolution and skips frames to maintain smoothness on free cloud CPUs.")
