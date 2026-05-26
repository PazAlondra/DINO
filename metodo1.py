import cv2
import os

from preprocesamiento import binarizar_adaptativo


def binarizar(img_gris, umbral=127):
    """
    Convierte una imagen gris a binaria adaptandose al fondo.

    Entrada:
        img_gris: imagen en escala de grises.
        umbral: valor base de binarizacion.
    Salida:
        Imagen binaria de 0 y 255.
    Para que sirve:
        Normaliza templates y frames antes de compararlos.
    """
    return binarizar_adaptativo(img_gris, umbral)


def cargar_mascaras_sift(carpeta_mascaras="img/mascaras_procesadas"):
    """
    Carga mascaras binarias y calcula sus descriptores SIFT.

    Entrada:
        carpeta_mascaras: carpeta con archivos terminados en "_binaria.png".
    Salida:
        sift: detector SIFT de OpenCV.
        templates: lista de diccionarios con nombre, imagen, keypoints y descriptores.
    Para que sirve:
        Prepara las mascaras de cactus/pajaro para reconocer obstaculos por SIFT.
    """
    sift = cv2.SIFT_create()
    templates = []

    for raiz, _, archivos in os.walk(carpeta_mascaras):
        for archivo in archivos:
            archivo_lower = archivo.lower()

            if not archivo_lower.endswith("_binaria.png"):
                continue

            if archivo_lower.startswith("cactus"):
                tipo = "cactus"
            elif archivo_lower.startswith("pajaro"):
                tipo = "pajaro"
            else:
                continue

            ruta = os.path.join(raiz, archivo)
            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img_binaria = binarizar(img)
            kp, des = sift.detectAndCompute(img_binaria, None)

            if des is not None and len(kp) > 0:
                templates.append({
                    "nombre": archivo,
                    "tipo": tipo,
                    "img": img_binaria,
                    "kp": kp,
                    "des": des
                })

    print(f"Mascaras obstaculos cargadas: {len(templates)}")
    return sift, templates

def cargar_templates_dino(carpeta_dino="img/dino"):
    """
    Carga las imagenes del dinosaurio para localizarlo por template matching.

    Entrada:
        carpeta_dino: carpeta con imagenes del dinosaurio.
    Salida:
        Lista de tuplas (nombre_archivo, imagen_binaria).
    Para que sirve:
        Permite fijar la posicion del dino y construir el ROI de peligro.
    """
    templates = []

    for archivo in os.listdir(carpeta_dino):
        if archivo.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            ruta = os.path.join(carpeta_dino, archivo)
            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            templates.append((archivo, binarizar(img)))

    print(f"Templates dino cargados: {len(templates)}")
    return templates


def detectar_dino(frame, templates_dino, umbral=0.18):
    """
    Busca el dinosaurio en la zona izquierda del frame.

    Entrada:
        frame: imagen BGR capturada.
        templates_dino: templates cargados con cargar_templates_dino.
        umbral: error maximo permitido para aceptar el match.
    Salida:
        bbox: (x, y, w, h) si encuentra el dino; None si falla.
        nombre: nombre del template ganador.
    Para que sirve:
        Fija la referencia desde donde se calcula el ROI de obstaculos.
    """
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

            dino = cv2.resize(dino_original, (w, h), interpolation=cv2.INTER_NEAREST)
            dino = binarizar(dino)
            resultado = cv2.matchTemplate(zona, dino, cv2.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv2.minMaxLoc(resultado)

            if min_val < mejor_valor:
                mejor_valor = min_val
                mejor_nombre = nombre
                mejor_bbox = (zona_x1 + min_loc[0], zona_y1 + min_loc[1], w, h)

    if mejor_bbox is None or mejor_valor > umbral:
        return None, None

    return mejor_bbox, mejor_nombre


def crear_roi_peligro(
    frame,
    bbox_dino,
    saltos=0,
    avance_cada=200,
    avance_px=20,
    inicio_factor=0.35,
    fin_factor=1.65
):
    """
    Calcula el ROI de peligro delante del dinosaurio.

    Entrada:
        frame: imagen BGR capturada.
        bbox_dino: posicion del dino como (x, y, w, h).
        saltos: contador usado para desplazar el ROI conforme avanza la partida.
        avance_cada: cada cuantos saltos se mueve el ROI.
        avance_px: cuantos pixeles avanza por bloque.
        inicio_factor/fin_factor: distancia del ROI medida en anchos del dino.
    Salida:
        Coordenadas (x1, y1, x2, y2).
    Para que sirve:
        Limita la busqueda al area donde un obstaculo ya es peligroso.
    """
    alto, ancho = frame.shape[:2]
    x, y, w, h = bbox_dino
    desplazamiento = (saltos // avance_cada) * avance_px

    x1 = x + w + int(w * inicio_factor) + desplazamiento
    x2 = x + w + int(w * fin_factor) + desplazamiento
    y1 = y + int(h * 0.10)
    y2 = y + int(h * 0.90)

    x1 = max(0, x1)
    x2 = min(ancho, x2)
    y1 = max(0, y1)
    y2 = min(alto, y2)

    return x1, y1, x2, y2


def crear_roi_pajaro(frame, bbox_dino, saltos=0, avance_cada=200, avance_px=20):
    """
    Calcula un ROI superior para detectar pajaros.

    Entrada:
        frame: imagen BGR capturada.
        bbox_dino: posicion del dino como (x, y, w, h).
        saltos/avance_cada/avance_px: control de desplazamiento horizontal.
    Salida:
        Coordenadas (x1, y1, x2, y2).
    Para que sirve:
        Detecta aves altas sin modificar la logica del ROI rojo.
    """
    alto, ancho = frame.shape[:2]
    x, y, w, h = bbox_dino
    desplazamiento = (saltos // avance_cada) * avance_px

    x1 = x + w + int(w * 0.35) + desplazamiento
    x2 = x + w + int(w * 1.85) + desplazamiento
    y1 = y - int(h * 1.10)
    y2 = y - int(h * 0.05)

    x1 = max(0, x1)
    x2 = min(ancho, x2)
    y1 = max(0, y1)
    y2 = min(alto, y2)

    return x1, y1, x2, y2


def asegurar_dino(frame, templates_dino, bbox_dino_fijo):
    """
    Devuelve la posicion fija del dino o intenta detectarla si aun no existe.

    Entrada:
        frame: imagen BGR capturada.
        templates_dino: templates del dino.
        bbox_dino_fijo: bbox guardado de ciclos anteriores.
    Salida:
        bbox_dino, bbox_dino_fijo_actualizado.
    Para que sirve:
        Evita recalcular el dino en cada frame cuando ya esta localizado.
    """
    if bbox_dino_fijo is not None:
        return bbox_dino_fijo, bbox_dino_fijo

    bbox_dino, _ = detectar_dino(frame, templates_dino)

    if bbox_dino is None:
        return None, None

    print("Dino fijado:", bbox_dino)
    return bbox_dino, bbox_dino


def detectar_templates_sift_en_roi(
    sift,
    roi,
    templates,
    min_matches=6,
    min_inliers=5,
    ratio_lowe=0.65,
    validar_geometria=True
):
    """
    Busca un conjunto de templates SIFT dentro de un ROI.

    Entrada:
        sift: detector SIFT.
        roi: recorte BGR donde se busca.
        templates: mascaras con descriptores SIFT.
        min_matches: matches minimos antes de validar geometria.
        min_inliers: matches coherentes minimos despues de RANSAC.
        ratio_lowe: umbral del filtro Lowe; menor valor es mas estricto.
        validar_geometria: si True exige homografia; si False acepta por matches.
    Salida:
        El template detectado o None.
    Para que sirve:
        Reutiliza la comparacion SIFT para cactus y pajaros evitando falsos
        positivos por matches sueltos, como nubes.
    """
    if roi.size == 0 or not templates:
        return None

    gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binario_roi = binarizar(gris_roi)
    kp_frame, des_frame = sift.detectAndCompute(binario_roi, None)

    if des_frame is None:
        return None

    bf = cv2.BFMatcher()

    for template in templates:
        matches = bf.knnMatch(template["des"], des_frame, k=2)
        buenos = []

        for match in matches:
            if len(match) < 2:
                continue

            m, n = match

            if m.distance < ratio_lowe * n.distance:
                buenos.append(m)

        if len(buenos) < min_matches:
            continue

        if not validar_geometria:
            return template

        if len(buenos) < 4:
            continue

        src_pts = cv2.KeyPoint_convert([template["kp"][m.queryIdx] for m in buenos])
        dst_pts = cv2.KeyPoint_convert([kp_frame[m.trainIdx] for m in buenos])
        src_pts = src_pts.reshape(-1, 1, 2)
        dst_pts = dst_pts.reshape(-1, 1, 2)

        homografia, mascara = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if homografia is None or mascara is None:
            continue

        inliers = int(mascara.ravel().sum())

        if inliers < min_inliers:
            continue

        alto_template, ancho_template = template["img"].shape[:2]
        esquinas = cv2.perspectiveTransform(
            cv2.KeyPoint_convert([
                cv2.KeyPoint(0, 0, 1),
                cv2.KeyPoint(ancho_template - 1, 0, 1),
                cv2.KeyPoint(ancho_template - 1, alto_template - 1, 1),
                cv2.KeyPoint(0, alto_template - 1, 1),
            ]).reshape(-1, 1, 2),
            homografia
        )
        x, y, w, h = cv2.boundingRect(esquinas.astype("float32"))

        if w < 6 or h < 6:
            continue

        if w > roi.shape[1] * 0.95 or h > roi.shape[0] * 0.95:
            continue

        if w > h * 6 or h > w * 6:
            continue

        if -10 <= x <= roi.shape[1] and -10 <= y <= roi.shape[0]:
            return template

    return None


def detectar_templates_sift_simple(sift, roi, templates, min_matches=6):
    """
    Busca templates SIFT usando la logica simple original.

    Entrada:
        sift: detector SIFT.
        roi: recorte BGR donde se busca.
        templates: mascaras con descriptores SIFT.
        min_matches: matches buenos minimos para aceptar deteccion.
    Salida:
        El template detectado o None.
    Para que sirve:
        Mantiene el comportamiento original del ROI rojo que ya funcionaba.
    """
    if roi.size == 0 or not templates:
        return None

    gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binario_roi = binarizar(gris_roi)
    kp_frame, des_frame = sift.detectAndCompute(binario_roi, None)

    if des_frame is None:
        return None

    bf = cv2.BFMatcher()

    for template in templates:
        matches = bf.knnMatch(template["des"], des_frame, k=2)
        buenos = []

        for match in matches:
            if len(match) < 2:
                continue

            m, n = match

            if m.distance < 0.75 * n.distance:
                buenos.append(m)

        if len(buenos) >= min_matches:
            return template

    return None


def detectar_obstaculo_sift_mascaras(
    frame,
    sift,
    templates_obstaculos,
    templates_dino,
    bbox_dino_fijo=None,
    min_matches=6,
    saltos=0,
    roi_avance_cada=200,
    roi_avance_px=20
):
    """
    Detecta obstaculos comparando el ROI contra mascaras con SIFT.

    Entrada:
        frame: imagen BGR capturada.
        sift: detector SIFT.
        templates_obstaculos: mascaras con descriptores SIFT.
        templates_dino: templates para ubicar el dinosaurio.
        bbox_dino_fijo: posicion ya calibrada del dino.
        min_matches: matches minimos para disparar salto.
        saltos/roi_avance_cada/roi_avance_px: control de desplazamiento del ROI.
    Salida:
        accion: "saltar", "agacharse" o None.
        tipo_detectado: "cactus", "pajaro_bajo", "pajaro_alto" o None.
        frame: frame marcado para debug visual.
        bbox_dino_fijo: bbox actualizado del dinosaurio.
    Para que sirve:
        Es el metodo 1 del juego: preciso cuando las mascaras representan bien los obstaculos.
    """
    bbox_dino, bbox_dino_fijo = asegurar_dino(frame, templates_dino, bbox_dino_fijo)

    if bbox_dino is None:
        cv2.putText(frame, "DINO NO DETECTADO", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return None, None, frame, None

    x, y, w, h = bbox_dino
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # ROI superior para pajaros desactivado temporalmente.
    # templates_pajaro = [template for template in templates_obstaculos if template["tipo"] == "pajaro"]
    # px1, py1, px2, py2 = crear_roi_pajaro(
    #     frame,
    #     bbox_dino,
    #     saltos=saltos,
    #     avance_cada=roi_avance_cada,
    #     avance_px=roi_avance_px
    # )
    # roi_pajaro = frame[py1:py2, px1:px2].copy()
    # cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

    x1, y1, x2, y2 = crear_roi_peligro(
        frame,
        bbox_dino,
        saltos=saltos,
        avance_cada=roi_avance_cada,
        avance_px=roi_avance_px
    )
    roi = frame[y1:y2, x1:x2].copy()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if roi.size == 0:
        return None, None, frame, bbox_dino_fijo

    obstaculo_bajo = detectar_templates_sift_simple(
        sift,
        roi,
        templates_obstaculos,
        min_matches=min_matches
    )

    if obstaculo_bajo is not None:
        tipo = "pajaro_bajo" if obstaculo_bajo["tipo"] == "pajaro" else "cactus"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        texto_x = min(frame.shape[1] - 180, x2 + 8)
        cv2.putText(frame, f"Detecta: {tipo}", (texto_x, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, "Accion: saltar", (texto_x, y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return "saltar", tipo, frame, bbox_dino_fijo

    # Deteccion de pajaro alto desactivada temporalmente.
    # pajaro_alto = detectar_templates_sift_simple(
    #     sift,
    #     roi_pajaro,
    #     templates_pajaro,
    #     min_matches=max(4, min_matches - 2)
    # )
    #
    # if pajaro_alto is not None:
    #     cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
    #     texto_x = min(frame.shape[1] - 210, px2 + 8)
    #     cv2.putText(frame, "Detecta: pajaro_alto", (texto_x, py1 + 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    #     cv2.putText(frame, "Accion: agacharse", (texto_x, py1 + 45),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    #     return "agacharse", "pajaro_alto", frame, bbox_dino_fijo

    return None, None, frame, bbox_dino_fijo
