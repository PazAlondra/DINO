import cv2
import time
import pyautogui

from capturas import capturar_obs_ram_60fps
from preprocesamiento import preprocesar_frame
from detector_fig import (
    cargar_mascaras_sift,
    cargar_templates_dino,
    detectar_cactus_en_roi,
    detectar_obstaculo_harris_en_roi
)

# ----------------------------------------
# Configuración visual
# ----------------------------------------
ANCHO = 480
ALTO = 270
MODO_DETECCION = "sift_mascaras"  # opciones: "sift_mascaras" o "harris"
#MODO_DETECCION = "harris"
UMBRAL_BINARIZADO_NORMAL = 127
UMBRAL_NEGRO_A_BLANCO = 175
UMBRAL_GRIS_A_NEGRO = 250
ROI_AVANCE_CADA_SALTOS = 10
ROI_AVANCE_PX = 3
HARRIS_MIN_ESQUINAS = 4
HARRIS_AREA_MINIMA = 20
HARRIS_ROI_INICIO_FACTOR = 1.80
HARRIS_ROI_FIN_FACTOR = 3.30

# ----------------------------------------
# Carga de templates
# ----------------------------------------
if MODO_DETECCION == "sift_mascaras":
    sift, templates_cactus = cargar_mascaras_sift("img/mascaras_procesadas")
else:
    sift, templates_cactus = None, []

templates_dino = cargar_templates_dino("img/dino")

# ----------------------------------------
# Ventanas
# ----------------------------------------
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", ANCHO, ALTO)

#cv2.namedWindow("Grises", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Grises", ANCHO, ALTO)

cv2.namedWindow("Binarizado", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Binarizado", ANCHO, ALTO)

# ----------------------------------------
# Control de salto
# ----------------------------------------
ultimo_salto = 0
cooldown_salto = 0.03

# ----------------------------------------
# Control del dino
# ----------------------------------------
bbox_dino_fijo = None
ultimo_chequeo_dino = 0
intervalo_chequeo_dino = 60000  #   segundos
contador_saltos = 0

# ----------------------------------------
# Loop principal
# ----------------------------------------
for frame, buffer_frames in capturar_obs_ram_60fps(indice=1):
    gris, binario = preprocesar_frame(
        frame,
        umbral=UMBRAL_BINARIZADO_NORMAL,
        umbral_negro=UMBRAL_NEGRO_A_BLANCO,
        umbral_blanco=UMBRAL_GRIS_A_NEGRO
    )
    # Recalibración automática del dino
    if time.time() - ultimo_chequeo_dino >= intervalo_chequeo_dino:
        bbox_dino_fijo = None
        ultimo_chequeo_dino = time.time()
        print("Recalibrando dino automáticamente...")

    if MODO_DETECCION == "sift_mascaras":
        detectado, frame_marcado, bbox_dino_fijo = detectar_cactus_en_roi(
            frame,
            sift,
            templates_cactus,
            templates_dino,
            bbox_dino_fijo,
            min_matches=6,
            puntaje=contador_saltos,
            roi_avance_cada=ROI_AVANCE_CADA_SALTOS,
            roi_avance_px=ROI_AVANCE_PX
        )
    elif MODO_DETECCION == "harris":
        detectado, frame_marcado, bbox_dino_fijo = detectar_obstaculo_harris_en_roi(
            frame,
            templates_dino,
            bbox_dino_fijo,
            puntaje=contador_saltos,
            roi_avance_cada=ROI_AVANCE_CADA_SALTOS,
            roi_avance_px=ROI_AVANCE_PX,
            roi_inicio_factor=HARRIS_ROI_INICIO_FACTOR,
            roi_fin_factor=HARRIS_ROI_FIN_FACTOR,
            min_esquinas=HARRIS_MIN_ESQUINAS,
            area_minima=HARRIS_AREA_MINIMA
        )
    else:
        raise ValueError(f"Modo de deteccion no valido: {MODO_DETECCION}")

    cv2.putText(frame_marcado, f"Saltos: {contador_saltos}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(frame_marcado, f"Modo: {MODO_DETECCION}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    if detectado and time.time() - ultimo_salto >= cooldown_salto:
        pyautogui.press("up")
        contador_saltos += 1
        print("↑ SALTO")
        ultimo_salto = time.time()

    cv2.imshow("Original", frame_marcado)
    #cv2.imshow("Grises", gris)
    cv2.imshow("Binarizado", binario)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord("r"):
        bbox_dino_fijo = None
        contador_saltos = 0
        ultimo_chequeo_dino = time.time()
        print("Recalibración manual del dino")

cv2.destroyAllWindows()
