import cv2
import time
import pyautogui

from capturas import capturar_obs_ram_60fps
from preprocesamiento import dibujar_estado_fondo, dibujar_zonas_muestreo, preprocesar_frame
from metodo1 import (
    cargar_mascaras_sift,
    cargar_templates_dino,
    detectar_obstaculo_sift_mascaras
)
from metodo2 import detectar_obstaculo_pixeles
from metricas import (
    cargar_mascaras_numeros,
    crear_metricas,
    guardar_metricas,
    reconocer_puntaje_por_mascaras,
    registrar_evento,
    registrar_salto
)


# ----------------------------------------
# Configuracion general
# ----------------------------------------
ANCHO = 480
ALTO = 270

# Opciones: "sift_mascaras" o "pixeles"
MODO_DETECCION = "sift_mascaras"
#MODO_DETECCION = "pixeles"

# Binarizado mostrado en la ventana "Binarizado"
UMBRAL_BINARIZADO_NORMAL = 127
UMBRAL_NEGRO_A_BLANCO = 130
UMBRAL_GRIS_A_NEGRO = 250

# ROI: se mueve a la derecha conforme aumentan los saltos
ROI_AVANCE_CADA_SALTOS = 10
ROI_AVANCE_PX = 3

# Metodo 1: SIFT + mascaras
SIFT_MIN_MATCHES = 6

# Metodo 2: analisis de pixeles
PIXELES_DIFERENCIA_MINIMA = 25
PIXELES_AREA_MINIMA = 30

PIXELES_ROI_INICIO_FACTOR = 0.50
PIXELES_ROI_FIN_FACTOR = 1.70

# Captura
INDICE_CAMARA = 1


# ----------------------------------------
# Carga de recursos
# ----------------------------------------
if MODO_DETECCION == "sift_mascaras":
    sift, templates_obstaculos = cargar_mascaras_sift("img/mascaras_procesadas")
else:
    sift, templates_obstaculos = None, []

templates_dino = cargar_templates_dino("img/dino")
templates_numeros = cargar_mascaras_numeros("img/mascaras_procesadas/numeros")


# ----------------------------------------
# Ventanas
# ----------------------------------------
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", ANCHO, ALTO)

cv2.namedWindow("Binarizado", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Binarizado", ANCHO, ALTO)


# ----------------------------------------
# Estado de partida
# ----------------------------------------
ultimo_salto = 0
cooldown_salto = 0.03

bbox_dino_fijo = None
ultimo_chequeo_dino = 0
intervalo_chequeo_dino = 60000
contador_saltos = 0
metricas = crear_metricas(MODO_DETECCION)


# ----------------------------------------
# Loop principal
# ----------------------------------------
try:
    for frame, buffer_frames in capturar_obs_ram_60fps(indice=INDICE_CAMARA):
        gris, binario = preprocesar_frame(
            frame,
            umbral=UMBRAL_BINARIZADO_NORMAL,
            umbral_negro=UMBRAL_NEGRO_A_BLANCO,
            umbral_blanco=UMBRAL_GRIS_A_NEGRO,
            bbox_dino=bbox_dino_fijo
        )

        if time.time() - ultimo_chequeo_dino >= intervalo_chequeo_dino:
            bbox_dino_fijo = None
            ultimo_chequeo_dino = time.time()
            print("Recalibrando dino automaticamente...")

        puntaje = reconocer_puntaje_por_mascaras(frame, templates_numeros)

        if puntaje is not None:
            metricas["puntaje_final"] = puntaje

        if MODO_DETECCION == "sift_mascaras":
            accion, tipo_detectado, frame_marcado, bbox_dino_fijo = detectar_obstaculo_sift_mascaras(
                frame,
                sift,
                templates_obstaculos,
                templates_dino,
                bbox_dino_fijo,
                min_matches=SIFT_MIN_MATCHES,
                saltos=contador_saltos,
                roi_avance_cada=ROI_AVANCE_CADA_SALTOS,
                roi_avance_px=ROI_AVANCE_PX
            )
        elif MODO_DETECCION == "pixeles":
            detectado, tipo_pixeles, frame_marcado, bbox_dino_fijo = detectar_obstaculo_pixeles(
                frame,
                templates_dino,
                bbox_dino_fijo,
                saltos=contador_saltos,
                roi_avance_cada=ROI_AVANCE_CADA_SALTOS,
                roi_avance_px=ROI_AVANCE_PX,
                roi_inicio_factor=PIXELES_ROI_INICIO_FACTOR,
                roi_fin_factor=PIXELES_ROI_FIN_FACTOR,
                diferencia_minima=PIXELES_DIFERENCIA_MINIMA,
                area_minima=PIXELES_AREA_MINIMA
            )
            accion = "saltar" if detectado else None
            tipo_detectado = tipo_pixeles if detectado else None
        else:
            raise ValueError(f"Modo de deteccion no valido: {MODO_DETECCION}")

        gris, binario = preprocesar_frame(
            frame,
            umbral=UMBRAL_BINARIZADO_NORMAL,
            umbral_negro=UMBRAL_NEGRO_A_BLANCO,
            umbral_blanco=UMBRAL_GRIS_A_NEGRO,
            bbox_dino=bbox_dino_fijo
        )

        alto_frame, ancho_frame = frame_marcado.shape[:2]
        x_info = max(10, ancho_frame - 260)

        cv2.putText(frame_marcado, f"Saltos: {contador_saltos}", (x_info, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame_marcado, f"Modo: {MODO_DETECCION}", (x_info, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        dibujar_zonas_muestreo(frame_marcado, bbox_dino_fijo)
        dibujar_estado_fondo(frame_marcado, gris, bbox_dino_fijo)

        if accion == "saltar" and time.time() - ultimo_salto >= cooldown_salto:
            pyautogui.press("up")
            contador_saltos += 1
            registrar_salto(metricas)
            registrar_evento(metricas, tipo_detectado, accion)
            print("SALTO")
            ultimo_salto = time.time()

        cv2.imshow("Original", frame_marcado)
        cv2.imshow("Binarizado", binario)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == ord("r"):
            bbox_dino_fijo = None
            contador_saltos = 0
            metricas = crear_metricas(MODO_DETECCION)
            ultimo_chequeo_dino = time.time()
            print("Recalibracion manual del dino y metricas reiniciadas")

finally:
    guardar_metricas(metricas)
    cv2.destroyAllWindows()
