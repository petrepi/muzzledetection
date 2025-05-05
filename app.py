import streamlit as st
import cv2
import sqlite3
from datetime import timedelta
from inference_sdk import InferenceHTTPClient
import tempfile
import os
import pandas as pd
from PIL import Image
import numpy as np

# Настройки
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="f4M61moZfIVKSIEj14fm"
)
MODEL_ID = "dog-muzzle-ilhnx/3"
DB_NAME = "detections.db"

# Инициализация БД
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            type TEXT,
            second INTEGER,
            dangerous_dogs INTEGER,
            muzzles INTEGER,
            no_muzzles INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Функции обработки
def analyze_predictions(predictions):
    stats = {"Dangerous_Dogs": 0, "Muzzle": 0, "No_Muzzle": 0}
    for pred in predictions["predictions"]:
        class_name = pred["class"]
        if class_name in stats:
            stats[class_name] += 1
    return stats

def draw_detections(frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for pred in predictions["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        x1, y1 = x - w//2, y - h//2
        x2, y2 = x + w//2, y + h//2
        
        color = (0, 255, 0) if pred["class"] == "Dangerous_Dogs" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{pred['class']} {pred['confidence']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def process_and_save(file):
    is_video = file.name.lower().endswith(('.mp4', '.avi', '.mov'))
    temp_path = os.path.join(tempfile.gettempdir(), file.name)
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if is_video:
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(fps) == 0:
                current_second = int(frame_count // fps)
                result = CLIENT.infer(frame, model_id=MODEL_ID)
                stats = analyze_predictions(result)
                
                cursor.execute('''
                    INSERT INTO detections 
                    (filename, type, second, dangerous_dogs, muzzles, no_muzzles)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (file.name, "video", current_second, 
                     stats["Dangerous_Dogs"], stats["Muzzle"], stats["No_Muzzle"]))
                conn.commit()

                detected_frame = draw_detections(frame.copy(), result)
                st_frame.image(detected_frame, caption=f"Second {current_second}")

            frame_count += 1

        cap.release()
    else:
        image = cv2.imread(temp_path)
        result = CLIENT.infer(image, model_id=MODEL_ID)
        stats = analyze_predictions(result)
        
        cursor.execute('''
            INSERT INTO detections 
            (filename, type, second, dangerous_dogs, muzzles, no_muzzles)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (file.name, "photo", 0, 
             stats["Dangerous_Dogs"], stats["Muzzle"], stats["No_Muzzle"]))
        conn.commit()

        detected_image = draw_detections(image.copy(), result)
        st.image(detected_image, caption="Detection Result")

    conn.close()
    os.remove(temp_path)

# Интерфейс Streamlit
st.set_page_config(layout="wide")
page = st.sidebar.selectbox("Выбеете страницу", ["Детекция", "Логи"])

if page == "Детекция":
    st.title("🐶Детекция собак без намордников в общественных местах🐶")
    uploaded_file = st.file_uploader("Загрузите фото или видео", 
                                   type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file:
        st.write(f"Pr {uploaded_file.name}...")
        process_and_save(uploaded_file)
        st.success("Готово! Чекай логи.")

else:
    st.title("Логи детекции")
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql('''
        SELECT 
            filename, type, second, 
            dangerous_dogs, muzzles, no_muzzles,
            timestamp
        FROM detections 
        ORDER BY timestamp DESC 
        LIMIT 10
    ''', conn)
    conn.close()

    st.dataframe(df.style.highlight_max(axis=0, subset=[
        'dangerous_dogs', 'muzzles', 'no_muzzles'
    ]), use_container_width=True)

    st.download_button(
        label="Export to CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='detections_log.csv',
        mime='text/csv'
    )