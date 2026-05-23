import cv2
import numpy as np


def fondo_es_claro(img_gris, umbral_fondo=170):
    alto, ancho = img_gris.shape[:2]

    zonas = [
        img_gris[int(alto * 0.25):int(alto * 0.90), int(ancho * 0.05):int(ancho * 0.95)],
        img_gris[int(alto * 0.30):int(alto * 0.85), int(ancho * 0.50):int(ancho * 0.98)],
        img_gris[int(alto * 0.30):int(alto * 0.85), int(ancho * 0.02):int(ancho * 0.50)],
        img_gris[int(alto * 0.35):int(alto * 0.75), int(ancho * 0.15):int(ancho * 0.85)],
    ]

    for zona in zonas:
        if zona.size == 0:
            continue

        mediana = np.median(zona)
        porcentaje_claro = np.count_nonzero(zona > 180) / zona.size

        if mediana > umbral_fondo or porcentaje_claro > 0.45:
            return True

    return False


def binarizar_adaptativo(img_gris, umbral=127, umbral_fondo_claro=220):
    if fondo_es_claro(img_gris):
        tipo = cv2.THRESH_BINARY
        umbral_actual = umbral_fondo_claro
    else:
        tipo = cv2.THRESH_BINARY
        umbral_actual = umbral

    _, binario = cv2.threshold(
        img_gris,
        umbral_actual,
        255,
        tipo
    )

    return binario


def binarizar_grises(img_gris, umbral_negro=60, umbral_blanco=201):
    binario = np.full_like(img_gris, 255)
    mascara_gris = (img_gris > umbral_negro) & (img_gris < umbral_blanco)
    binario[mascara_gris] = 0

    return binario


def binarizar_normal(img_gris, umbral=127):
    _, binario = cv2.threshold(
        img_gris,
        umbral,
        255,
        cv2.THRESH_BINARY
    )

    return binario


def preprocesar_frame(
    frame,
    umbral=127,
    umbral_negro=60,
    umbral_blanco=201
):
    """
    Procesa un frame:
    Original -> escala de grises -> binarizacion segun el fondo
    """

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if fondo_es_claro(gris):
        binario = binarizar_grises(gris, umbral_negro, umbral_blanco)
    else:
        binario = binarizar_normal(gris, umbral)

    return gris, binario
