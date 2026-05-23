import cv2
import os
import numpy as np

def procesar_mascaras(carpeta_entrada="img", carpeta_salida="mascaras_procesadas", umbral=127):
    os.makedirs(carpeta_salida, exist_ok=True)

    extensiones = (".jpg", ".jpeg", ".png", ".bmp")

    for archivo in os.listdir(carpeta_entrada):
        if not archivo.lower().endswith(extensiones):
            continue

        ruta = os.path.join(carpeta_entrada, archivo)
        img = cv2.imread(ruta)

        if img is None:
            print(f"No se pudo leer: {archivo}")
            continue

        # 1. Escala de grises
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Suavizado para reducir ruido
        suavizada = cv2.GaussianBlur(gris, (3, 3), 0)

        # 3. Binarización
        _, binaria = cv2.threshold(
            suavizada,
            umbral,
            255,
            cv2.THRESH_BINARY
        )

        # 4. Limpieza morfológica
        kernel = np.ones((3, 3), np.uint8)

        binaria_limpia = cv2.morphologyEx(
            binaria,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        binaria_limpia = cv2.morphologyEx(
            binaria_limpia,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        # 5. Inversa
        inversa = cv2.bitwise_not(binaria_limpia)

        nombre, _ = os.path.splitext(archivo)

        cv2.imwrite(os.path.join(carpeta_salida, f"{nombre}_gris.png"), gris)
        cv2.imwrite(os.path.join(carpeta_salida, f"{nombre}_binaria.png"), binaria_limpia)
        cv2.imwrite(os.path.join(carpeta_salida, f"{nombre}_inversa.png"), inversa)

        print(f"Procesada: {archivo}")

    print("Proceso terminado.")



procesar_mascaras(
    carpeta_entrada="img",
    carpeta_salida="img/mascaras_procesadas",
    umbral=100
)