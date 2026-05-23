import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Fuente encontrada en índice {i}")
            cv2.imshow(f"Indice {i}", frame)
            cv2.waitKey(2000)

    cap.release()

cv2.destroyAllWindows()