import cv2
import os
import numpy as np

from preprocesamiento import binarizar_adaptativo


def binarizar(img_gris, umbral=127):
    return binarizar_adaptativo(img_gris, umbral)


def cargar_mascaras_sift(carpeta_mascaras="img/mascaras_procesadas"):
    sift = cv2.SIFT_create()
    templates = []

    for archivo in os.listdir(carpeta_mascaras):
        if not archivo.lower().endswith("_binaria.png"):
            continue

        ruta = os.path.join(carpeta_mascaras, archivo)
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img_binaria = binarizar(img)

        kp, des = sift.detectAndCompute(img_binaria, None)

        if des is not None and len(kp) > 0:
            templates.append({
                "nombre": archivo,
                "img": img_binaria,
                "kp": kp,
                "des": des
            })

    print(f"Máscaras cactus cargadas: {len(templates)}")
    return sift, templates


def cargar_templates_dino(carpeta_dino="img/dino"):
    templates = []

    for archivo in os.listdir(carpeta_dino):
        if archivo.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            ruta = os.path.join(carpeta_dino, archivo)
            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img_binaria = binarizar(img)

            templates.append((archivo, img_binaria))

    print(f"Templates dino cargados: {len(templates)}")
    return templates


def detectar_dino(frame, templates_dino, umbral=0.18):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = binarizar(gris)

    alto, ancho = gris.shape[:2]

    zona_y1 = int(alto * 0.30)
    zona_y2 = int(alto * 0.75)
    zona_x1 = 0
    zona_x2 = int(ancho * 0.25)

    zona = gris[zona_y1:zona_y2, zona_x1:zona_x2]

    mejor_valor = 999
    mejor_bbox = None
    mejor_nombre = None

    escalas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for nombre, dino_original in templates_dino:
        for escala in escalas:
            w = int(dino_original.shape[1] * escala)
            h = int(dino_original.shape[0] * escala)

            if w < 10 or h < 10:
                continue

            if w > zona.shape[1] or h > zona.shape[0]:
                continue

            dino = cv2.resize(
                dino_original,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            dino = binarizar(dino)

            resultado = cv2.matchTemplate(
                zona,
                dino,
                cv2.TM_SQDIFF_NORMED
            )

            min_val, _, min_loc, _ = cv2.minMaxLoc(resultado)

            if min_val < mejor_valor:
                mejor_valor = min_val
                mejor_nombre = nombre
                mejor_bbox = (
                    zona_x1 + min_loc[0],
                    zona_y1 + min_loc[1],
                    w,
                    h
                )

    if mejor_bbox is None or mejor_valor > umbral:
        return None, None

    return mejor_bbox, mejor_nombre


def crear_roi_peligro(
    frame,
    bbox_dino,
    puntaje=0,
    avance_cada=200,
    avance_px=20,
    inicio_factor=0.1,
    fin_factor=1.4
):
    alto, ancho = frame.shape[:2]

    x, y, w, h = bbox_dino
    desplazamiento = (puntaje // avance_cada) * avance_px

    x1 = x + w + int(w * inicio_factor) + desplazamiento
    x2 = x + w + int(w * fin_factor) + desplazamiento

    y1 = y - int(h * 0.02)
    y2 = y + int(h * 1.0)

    x1 = max(0, x1)
    x2 = min(ancho, x2)
    y1 = max(0, y1)
    y2 = min(alto, y2)

    return x1, y1, x2, y2


def detectar_cactus_en_roi(
    frame,
    sift,
    templates_cactus,
    templates_dino,
    bbox_dino_fijo=None,
    min_matches=6,
    puntaje=0,
    roi_avance_cada=200,
    roi_avance_px=20
):
    if bbox_dino_fijo is None:
        bbox_dino, nombre_dino = detectar_dino(frame, templates_dino)

        if bbox_dino is None:
            cv2.putText(frame, "DINO NO DETECTADO", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return False, frame, None

        bbox_dino_fijo = bbox_dino
        print("Dino fijado:", bbox_dino_fijo)
    else:
        bbox_dino = bbox_dino_fijo

    x, y, w, h = bbox_dino

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    x1, y1, x2, y2 = crear_roi_peligro(
        frame,
        bbox_dino,
        puntaje=puntaje,
        avance_cada=roi_avance_cada,
        avance_px=roi_avance_px
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False, frame, bbox_dino_fijo

    gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binario_roi = binarizar(gris_roi)

    # Para depurar, si quieres ver exactamente qué compara:
    #cv2.imshow("ROI Binario", binario_roi)

    kp_frame, des_frame = sift.detectAndCompute(binario_roi, None)

    if des_frame is None:
        return False, frame, bbox_dino_fijo

    bf = cv2.BFMatcher()

    for template in templates_cactus:
        matches = bf.knnMatch(template["des"], des_frame, k=2)
        buenos = []

        for match in matches:
            if len(match) < 2:
                continue

            m, n = match

            if m.distance < 0.75 * n.distance:
                buenos.append(m)

        if len(buenos) >= min_matches:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "CACTUS DETECTADO", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return True, frame, bbox_dino_fijo

    return False, frame, bbox_dino_fijo


def detectar_obstaculo_harris_en_roi(
    frame,
    templates_dino,
    bbox_dino_fijo=None,
    puntaje=0,
    roi_avance_cada=200,
    roi_avance_px=20,
    roi_inicio_factor=0.35,
    roi_fin_factor=1.80,
    min_esquinas=4,
    area_minima=20
):
    if bbox_dino_fijo is None:
        bbox_dino, nombre_dino = detectar_dino(frame, templates_dino)

        if bbox_dino is None:
            cv2.putText(frame, "DINO NO DETECTADO", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return False, frame, None

        bbox_dino_fijo = bbox_dino
        print("Dino fijado:", bbox_dino_fijo)
    else:
        bbox_dino = bbox_dino_fijo

    x, y, w, h = bbox_dino

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    x1, y1, x2, y2 = crear_roi_peligro(
        frame,
        bbox_dino,
        puntaje=puntaje,
        avance_cada=roi_avance_cada,
        avance_px=roi_avance_px,
        inicio_factor=roi_inicio_factor,
        fin_factor=roi_fin_factor
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False, frame, bbox_dino_fijo

    gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gris_roi = cv2.GaussianBlur(gris_roi, (3, 3), 0)
    binario_roi = binarizar(gris_roi)

    borde = 3
    muestras_fondo = [
        binario_roi[:borde, :],
        binario_roi[-borde:, :],
        binario_roi[:, :borde],
        binario_roi[:, -borde:],
    ]
    fondo_claro = sum(m.mean() for m in muestras_fondo) / len(muestras_fondo) > 127

    if fondo_claro:
        objeto_roi = cv2.bitwise_not(binario_roi)
    else:
        objeto_roi = binario_roi.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    objeto_roi = cv2.morphologyEx(objeto_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    objeto_roi = cv2.dilate(objeto_roi, kernel, iterations=1)

    harris_input = np.float32(objeto_roi)
    respuesta = cv2.cornerHarris(
        harris_input,
        2,
        3,
        0.04
    )
    respuesta = cv2.dilate(respuesta, None)
    umbral_harris = 0.01 * respuesta.max()

    contornos, _ = cv2.findContours(
        objeto_roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mejor_esquinas = 0

    for contorno in contornos:
        cx, cy, cw, ch = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)

        if area < area_minima:
            continue

        if cw < 6 or ch < 8:
            continue

        if cw > roi.shape[1] * 0.95:
            continue

        if cw > ch * 5:
            continue

        zona_harris = respuesta[cy:cy + ch, cx:cx + cw]
        esquinas = int((zona_harris > umbral_harris).sum())
        mejor_esquinas = max(mejor_esquinas, esquinas)

        if esquinas < min_esquinas:
            continue

        ox1 = x1 + cx
        oy1 = y1 + cy
        ox2 = x1 + cx + cw
        oy2 = y1 + cy + ch

        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
        cv2.putText(frame, "OBSTACULO HARRIS", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return True, frame, bbox_dino_fijo

    cv2.putText(frame, f"Harris esquinas: {mejor_esquinas}", (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return False, frame, bbox_dino_fijo
