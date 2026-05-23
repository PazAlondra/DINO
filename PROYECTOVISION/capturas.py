import cv2
import time
from collections import deque

def capturar_obs_ram_60fps(indice=1, max_frames=60):
    cap = cv2.VideoCapture(indice)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("No se pudo abrir OBS Virtual Camera")
        return

    buffer_frames = deque(maxlen=max_frames)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer frame")
            break

        buffer_frames.append(frame)

        yield frame, buffer_frames

    cap.release()

