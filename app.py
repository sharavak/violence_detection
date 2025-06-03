import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import torch
import numpy as np


torch.classes.__path__ = []
model=YOLO('best.pt')
st.set_page_config('Violence Detection',page_icon='https://cdn-icons-png.freepik.com/64/18122/18122849.png')

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
blur_strength=201
upload_anim = load_lottieurl("https://lottie.host/a824695b-b788-4ac3-87b6-6c0c970a1779/RV5B7xGvJ9.json")
processing_anim = load_lottieurl("https://lottie.host/451651c5-32f9-443a-93f1-0e5d259bd9e3/8t3wX9d3Yf.json")
done_anim = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")

st.title("Violence Detection")
st.markdown("Scan videos or image for **blood**, **weapons**, and **violence**. Blur and save results.\n")


if 'video' not in st.session_state:
    st.session_state['video']=False

if 'opt' not in st.session_state:
    st.session_state['opt']=False

if 'img' not in st.session_state:
    st.session_state['img']=False


def process(blurred_frame,detection_summary):
    found_detect = False
    for box in boxes:
        cls = int(box.cls[0])
        label = class_names[cls].lower()
        if label in ["Violence", "blood", "weapon"]:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            roi = blurred_frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            blurred_frame[y1:y2, x1:x2] = blurred_roi
            found_detect = True
            detection_summary.append(label)
    return found_detect,detection_summary,blurred_frame
    

with st.sidebar:
    ans=st.selectbox("Select the type",options=['Image','Video'],index=0)
    if ans:
        st.session_state['opt']=ans

if st.session_state['opt']=='Video':
    with st.sidebar:
        st.header("\U0001F527 Settings")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
        blur_strength = st.slider("Blur Strength", 15, 501, 201, step=2)

    def track_video():
        st.session_state['video']=True
    video_file = st.file_uploader("\U0001F4F9 Upload a video", type=["mp4"],key='file', on_change=track_video)

    if video_file and  st.session_state['video']:
        st_lottie(processing_anim, height=200, key="processing")
        with st.status(label='\U0001F50D Analyzing video. Please wait...') as upd:
            detection_summary, violence_detected = [], []
            found_detect=False
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            stframe = st.empty()
            progress = st.progress(0)
            frame_count = 0
            while cap.isOpened():
                ret,frame=cap.read()
                if not ret:
                    break
                frame_count += 1
                results = model.predict(source=frame, conf=conf_thresh, stream=False)
                class_names = results[0].names
                boxes = results[0].boxes

                blurred_frame = frame.copy()
              
                for box in boxes:
                    cls = int(box.cls[0])
                    label = class_names[cls].lower()
                    if label in ["Violence", "blood", "weapon"]:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        roi = blurred_frame[y1:y2, x1:x2]
                        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
                        blurred_frame[y1:y2, x1:x2] = blurred_roi
                        found_detect = True
                        detection_summary.append(label)

                violence_detected.append(found_detect)                
                out.write(blurred_frame)
                progress.progress(min(frame_count / total_frames, 1.0),text=f'{int((frame_count / total_frames)*100)} %')
            st.session_state['video']=True
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            st_lottie(done_anim, height=200, key="done")
            upd.update(label='Video processing completed!',state='complete',expanded=True)


        if detection_summary:
            counts = Counter(detection_summary)
            with st.expander("\U0001F4CA Detection Summary"):
                for label, count in counts.items():
                    st.markdown(f"- **{label.capitalize()}**: `{count}` frame(s)")

            with st.expander("\U0001F4CA Class Distribution"):
                fig = go.Figure([go.Bar(x=list(counts.keys()), y=list(counts.values()))])
                fig.update_layout(title="Detected Classes", xaxis_title="Class", yaxis_title="Count")
                st.plotly_chart(fig)

        else:
            st.balloons()
            st.success("🎉 No violence detected!")

        if st.session_state['video'] :
            with open(output_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Blurred Video",
                    data=file,
                    file_name=f"{video_file.name.split('.')[0]}_blurred.{video_file.type.split('/')[1]}",
                    mime="video/mp4"
                    )
            st.session_state['video']=False
else:
    def track_image():
        st.session_state['img']=True
    img_file=st.file_uploader("🖼️ Upload a image", type=["jpg",'png','jpeg'],on_change=track_image)
    
    if st.session_state['img'] and img_file:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) 
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{img_file.type.split('/')[1]}").name
        detection_summary, violence_detected = [], []
        with st.status(label='Analyzing Image. Please wait...') as upd:
            results = model.predict(image)
            class_names = results[0].names
            boxes = results[0].boxes
            found_detect=False
            found_detect,sum,blurred_frame=process(image,detection_summary)
            st_lottie(done_anim, height=200, key="done")
            upd.update(label='Image processing completed!',state='complete',expanded=True)
        st.session_state['img']=True

        if found_detect:
            if st.session_state['img']:
                counts = Counter(detection_summary)
                with st.expander("\U0001F4CA Detection Summary"):
                    for label, count in counts.items():
                        st.markdown(f"- **{label.capitalize()}**: `{count}` objects")

                success, encoded_image = cv2.imencode(f".{img_file.type.split('/')[1]}", image)
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Blurred Image",
                        data=encoded_image.tobytes(),
                        file_name=f"{img_file.name.split('.')[0]}_blurred.{img_file.type.split('/')[1]}",
                        mime=img_file.type
                        )
                st.session_state['img']=False
        else:
             st.success("No violence detected!")