
import streamlit as st
import cv2
import numpy as np
import hmac
import os
from dotenv import load_dotenv
import math
from skimage.filters import frangi
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# Load the .env
load_dotenv()

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        # If you use .streamlit/secrets.toml, replace os.environ.get with st.secrets["STREAMLIT_PASSWORD"]
        if hmac.compare_digest(st.session_state["password"], os.environ.get("STREAMLIT_PASSWORD", "")):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # st.info(f"Password: {os.environ.get('STREAMLIT_PASSWORD', '')}")
    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

st.set_page_config(layout="wide")

st.title("Fiber Detection and Analysis App")
st.write("Upload an image to detect and analyze fibers.")

# --- CORE IMAGE PROCESSING ENGINE ---
def process_fiber_image(image_bytes):
    # Load Image
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # 1. Enhancement
    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(3,3))
    enhanced = clahe.apply(img_original.astype(np.uint8))
    
    # 2. Frangi Filter
    fr = frangi(enhanced, sigmas=np.arange(0.5, 4, 0.5), alpha=200, beta=1, gamma=10, black_ridges=False)
    fr_norm = cv2.normalize(fr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. Skeletonization
    fiber_core = (fr_norm > 20).astype(np.uint8) * 255
    dist = cv2.distanceTransform(fiber_core, cv2.DIST_L2, 3)
    skeleton = skeletonize(fiber_core > 0)
    skeleton_clean = (skeleton & (dist <= 10)).astype(np.uint8) * 255
    
    # 4. Filter small objects
    labels = label(skeleton_clean)
    skeleton_connected = np.zeros_like(skeleton_clean, dtype=bool)
    for r in regionprops(labels):
        if r.area >= 10:
            skeleton_connected[labels == r.label] = 1
    skeleton_connected = skeleton_connected.astype(np.uint8) * 255

    # 5. Blob identification (to remove junctions at noise points)
    blob_threshold = 21
    _, blob_core = cv2.threshold(img_original, blob_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    opened = cv2.morphologyEx(blob_core, cv2.MORPH_OPEN, kernel)
    opened = cv2.dilate(opened, kernel)
    inv_opened = cv2.bitwise_not(opened)
    
    # 6. Final Skeleton
    final_image = cv2.bitwise_and(skeleton_connected, inv_opened)
    skeleton_final = (skeletonize(final_image > 0).astype(np.uint8)) * 255
    new_final = skeleton_final.copy()

    # 7. Neighbor Counting & Endpoints
    skel_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D((skeleton_final > 0).astype(np.uint8), -1, skel_kernel)
    endpoints = np.logical_and(skeleton_final > 0, neighbor_count == 1)
    
    # 8. Path Extraction
    h, w = skeleton_final.shape
    visited = np.zeros_like(skeleton_final, dtype=bool)
    fibers = []
    
    def get_neighbors(y, x):
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and skeleton_final[ny, nx] > 0:
                yield ny, nx

    for sy, sx in zip(*np.where(endpoints)):
        if visited[sy, sx]: continue
        path = [(sy, sx)]
        visited[sy, sx] = True
        curr, prev = (sy, sx), None
        while True:
            y, x = curr
            nbrs = [n for n in get_neighbors(y, x) if n != prev and not visited[n]]
            if not nbrs: break
            nxt = nbrs[0]
            path.append(nxt)
            visited[nxt] = True
            prev, curr = curr, nxt
        if len(path) > 2: fibers.append(path)

    return skeleton_final, fibers, img_original
    
# --- APP LAYOUT ---
if check_password():
    st.title("🔬 Fiber Analysis Dashboard")
    st.sidebar.header("Settings")
    min_length = st.sidebar.slider("Min Fiber Length (px)", 5, 100, 20)
    
    uploaded_file = st.file_uploader("Upload fiber image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        with st.spinner('Processing fibers...'):
            skeleton_final, fibers, img_original = process_fiber_image(uploaded_file.read())
            
            # Prepare Visualizations
            img_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
            bbox_vis = img_bgr.copy()
            highlight_vis = img_bgr.copy()
            
            for path in fibers:
                length_px = sum(math.hypot(p2[1]-p1[1], p2[0]-p1[0]) for p1, p2 in zip(path[:-1], path[1:]))
                
                if length_px < min_length:
                    continue
                
                # Colors
                color = (0, 255, 0) # Green
                
                # 1. Bounding Box Logic
                ys, xs = [p[0] for p in path], [p[1] for p in path]
                x1, y1, x2, y2 = min(xs)-3, min(ys)-3, max(xs)+3, max(ys)+3
                cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), color, 1)
                cv2.putText(bbox_vis, f"{length_px:.1f}", (x1, max(y1-5, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 2. Highlight Logic (drawing the actual fiber)
                for y, x in path:
                    cv2.circle(highlight_vis, (x, y), 1, (255, 0, 255), -1) # Magenta highlights

            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Bounding Box View")
                st.image(bbox_vis, use_container_width=True)
            
            with col2:
                st.subheader("Highlighted Fiber View")
                st.image(highlight_vis, use_container_width=True)
                
            st.success(f"Detected {len(fibers)} fiber segments.")

    else:
        st.info("Please upload an image to start fiber analysis.")