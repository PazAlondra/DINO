import cv2
import os

def binarizar_imagenes(carpeta="img", umbral=127):
    """
    Convierte imágenes a blanco y negro binarizado
    y las guarda en la misma carpeta.

    Parámetros:
        carpeta: ruta donde están las imágenes
        umbral: valor de corte para binarización
    """

    extensiones = (".jpg", ".jpeg", ".png", ".bmp")

    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(extensiones):
            ruta = os.path.join(carpeta, archivo)

            img = cv2.imread(ruta)

            if img is None:
                continue

            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, binario = cv2.threshold(
                gris,
                umbral,
                255,
                cv2.THRESH_BINARY
            )

            nombre, extension = os.path.splitext(archivo)

            nueva_ruta = os.path.join(
                carpeta,
                f"{nombre}_binario{extension}"
            )

            cv2.imwrite(nueva_ruta, binario)

    print("Proceso terminado.")



binarizar_imagenes("img")