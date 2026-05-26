import cv2
import numpy as np

from metodo1 import asegurar_dino, crear_roi_peligro


def crear_roi_pajaro_pixeles(frame, bbox_dino, saltos=0, avance_cada=200, avance_px=20):
    """
    Calcula un ROI superior para pajaros en el metodo de pixeles.

    Entrada:
        frame: imagen BGR capturada.
        bbox_dino: posicion del dino como (x, y, w, h).
        saltos/avance_cada/avance_px: control de desplazamiento horizontal.
    Salida:
        Coordenadas (x1, y1, x2, y2).
    Para que sirve:
        Queda preparado para detectar pajaros altos y agacharse en metodo2.
        Actualmente no se usa; se deja comentado en detectar_obstaculo_pixeles.
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


def estimar_fondo_roi(gris_roi):
    """
    Estima el color del fondo usando los bordes del ROI.

    Entrada:
        gris_roi: ROI en escala de grises.
    Salida:
        Valor mediano del fondo.
    Para que sirve:
        Permite detectar pixeles que se separan del fondo sin usar mascaras.
    """
    borde = max(2, min(gris_roi.shape[:2]) // 12)
    muestras = [
        gris_roi[:borde, :],
        gris_roi[-borde:, :],
        gris_roi[:, :borde],
        gris_roi[:, -borde:],
    ]
    return float(np.median(np.concatenate([m.reshape(-1) for m in muestras])))


def crear_mascara_diferencia(gris_roi, valor_fondo, diferencia_minima=25):
    """
    Crea una mascara con pixeles distintos al fondo.

    Entrada:
        gris_roi: ROI en escala de grises.
        valor_fondo: valor estimado del fondo.
        diferencia_minima: distancia minima para considerar un pixel como objeto.
    Salida:
        Mascara binaria con objeto en blanco y fondo en negro.
    Para que sirve:
        Aisla cactus o pajaro por contraste de pixeles.
    """
    if valor_fondo > 127:
        mascara = gris_roi < valor_fondo - diferencia_minima
    else:
        mascara = gris_roi > valor_fondo + diferencia_minima

    objeto_roi = np.zeros_like(gris_roi, dtype=np.uint8)
    objeto_roi[mascara] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    objeto_roi = cv2.morphologyEx(objeto_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    objeto_roi = cv2.morphologyEx(objeto_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    return objeto_roi


def buscar_obstaculo_por_contornos(objeto_roi, area_minima=30):
    """
    Busca grupos de pixeles con forma razonable de obstaculo.

    Entrada:
        objeto_roi: mascara binaria con posibles obstaculos.
        area_minima: area minima del contorno.
    Salida:
        Caja (x, y, w, h) del obstaculo o None.
    Para que sirve:
        Filtra ruido y decide si hay algo peligroso dentro del ROI.
    """
    contornos, _ = cv2.findContours(
        objeto_roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mejor = None
    mejor_area = 0

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        mejor_area = max(mejor_area, area)

        if area < area_minima:
            continue

        if w < 6 or h < 8:
            continue

        if w > objeto_roi.shape[1] * 0.95:
            continue

        if w > h * 5:
            continue

        mejor = (x, y, w, h)
        break

    return mejor, mejor_area


def clasificar_obstaculo_pixeles(caja, alto_roi):
    """
    Clasifica el obstaculo detectado por posicion vertical.

    Entrada:
        caja: (x, y, w, h) del contorno detectado dentro del ROI.
        alto_roi: altura total del ROI.
    Salida:
        "cactus" si esta abajo; "pajaro" si esta mas arriba.
    Para que sirve:
        Da una etiqueta simple al metodo de pixeles sin usar mascaras.
    """
    _, y, _, h = caja
    toca_zona_baja = y + h > alto_roi * 0.75

    if toca_zona_baja:
        return "cactus"

    return "pajaro"


def detectar_obstaculo_pixeles(
    frame,
    templates_dino,
    bbox_dino_fijo=None,
    saltos=0,
    roi_avance_cada=200,
    roi_avance_px=20,
    roi_inicio_factor=1.80,
    roi_fin_factor=3.30,
    diferencia_minima=25,
    area_minima=30
):
    """
    Detecta obstaculos por analisis directo de pixeles dentro del ROI.

    Entrada:
        frame: imagen BGR capturada.
        templates_dino: templates para ubicar el dinosaurio.
        bbox_dino_fijo: posicion calibrada del dinosaurio.
        saltos/roi_avance_cada/roi_avance_px: control de desplazamiento del ROI.
        roi_inicio_factor/roi_fin_factor: posicion del ROI para este metodo.
        diferencia_minima: contraste minimo contra el fondo.
        area_minima: tamano minimo del grupo detectado.
    Salida:
        detectado: True si hay obstaculo.
        tipo_detectado: "cactus", "pajaro" o None.
        frame: frame marcado para debug visual.
        bbox_dino_fijo: bbox actualizado del dinosaurio.
    Para que sirve:
        Es el metodo 2: detecta cactus/pajaro sin mascaras, usando contraste de pixeles.
    """
    bbox_dino, bbox_dino_fijo = asegurar_dino(frame, templates_dino, bbox_dino_fijo)

    if bbox_dino is None:
        cv2.putText(frame, "DINO NO DETECTADO", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return False, None, frame, None

    x, y, w, h = bbox_dino
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # ROI superior para pajaros en metodo2.
    # Desactivado por ahora: cuando se active, este ROI deberia detectar
    # pajaros altos por pixeles y regresar accion "agacharse".
    #
    # px1, py1, px2, py2 = crear_roi_pajaro_pixeles(
    #     frame,
    #     bbox_dino,
    #     saltos=saltos,
    #     avance_cada=roi_avance_cada,
    #     avance_px=roi_avance_px
    # )
    # roi_pajaro = frame[py1:py2, px1:px2].copy()
    # cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
    #
    # gris_pajaro = cv2.cvtColor(roi_pajaro, cv2.COLOR_BGR2GRAY)
    # gris_pajaro = cv2.GaussianBlur(gris_pajaro, (3, 3), 0)
    # fondo_pajaro = estimar_fondo_roi(gris_pajaro)
    # mascara_pajaro = crear_mascara_diferencia(gris_pajaro, fondo_pajaro, diferencia_minima)
    # caja_pajaro, _ = buscar_obstaculo_por_contornos(mascara_pajaro, area_minima)
    #
    # if caja_pajaro is not None:
    #     cx, cy, cw, ch = caja_pajaro
    #     cv2.rectangle(frame, (px1 + cx, py1 + cy), (px1 + cx + cw, py1 + cy + ch), (0, 255, 0), 2)
    #     return True, "pajaro", frame, bbox_dino_fijo

    x1, y1, x2, y2 = crear_roi_peligro(
        frame,
        bbox_dino,
        saltos=saltos,
        avance_cada=roi_avance_cada,
        avance_px=roi_avance_px,
        inicio_factor=roi_inicio_factor,
        fin_factor=roi_fin_factor
    )
    roi = frame[y1:y2, x1:x2].copy()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if roi.size == 0:
        return False, None, frame, bbox_dino_fijo

    gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gris_roi = cv2.GaussianBlur(gris_roi, (3, 3), 0)
    valor_fondo = estimar_fondo_roi(gris_roi)
    objeto_roi = crear_mascara_diferencia(gris_roi, valor_fondo, diferencia_minima)
    caja, mejor_area = buscar_obstaculo_por_contornos(objeto_roi, area_minima)

    if caja is None:
        cv2.putText(frame, f"Pixeles area: {int(mejor_area)}", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return False, None, frame, bbox_dino_fijo

    cx, cy, cw, ch = caja
    tipo_detectado = clasificar_obstaculo_pixeles(caja, roi.shape[0])
    ox1 = x1 + cx
    oy1 = y1 + cy
    ox2 = x1 + cx + cw
    oy2 = y1 + cy + ch

    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
    cv2.putText(frame, f"PIXELES: {tipo_detectado}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return True, tipo_detectado, frame, bbox_dino_fijo
