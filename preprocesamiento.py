import cv2
import numpy as np


def fondo_es_claro(img_gris, umbral_fondo=170, bbox_dino=None):
    alto, ancho = img_gris.shape[:2]

    zonas = obtener_zonas_muestreo(alto, ancho, bbox_dino)

    for x1, y1, x2, y2 in zonas:
        zona = img_gris[y1:y2, x1:x2]
        if zona.size == 0:
            continue

        mediana = np.median(zona)
        porcentaje_claro = np.count_nonzero(zona > 180) / zona.size

        if mediana > umbral_fondo or porcentaje_claro > 0.45:
            return True

    return False


def medir_fondo(img_gris, umbral_fondo=170, bbox_dino=None):
    """
    Mide la muestra usada para decidir si el fondo es claro u oscuro.

    Entrada:
        img_gris: frame en escala de grises.
        umbral_fondo: mediana minima para considerar fondo claro.
        bbox_dino: posicion del dino, si existe.
    Salida:
        Diccionario con mediana, porcentaje_claro y es_claro.
    Para que sirve:
        Depura por que se activa o no la binarizacion de fondo blanco.
    """
    alto, ancho = img_gris.shape[:2]
    zonas = obtener_zonas_muestreo(alto, ancho, bbox_dino)

    if not zonas:
        return {"mediana": 0, "porcentaje_claro": 0, "es_claro": False}

    x1, y1, x2, y2 = zonas[0]
    zona = img_gris[y1:y2, x1:x2]

    if zona.size == 0:
        return {"mediana": 0, "porcentaje_claro": 0, "es_claro": False}

    mediana = float(np.median(zona))
    porcentaje_claro = float(np.count_nonzero(zona > 180) / zona.size)
    es_claro = mediana > umbral_fondo or porcentaje_claro > 0.45

    return {
        "mediana": mediana,
        "porcentaje_claro": porcentaje_claro,
        "es_claro": es_claro
    }


def obtener_zonas_muestreo(alto, ancho, bbox_dino=None):
    """
    Devuelve las zonas usadas para decidir si el fondo esta claro u oscuro.

    Entrada:
        alto, ancho: dimensiones del frame.
        bbox_dino: posicion del dino como (x, y, w, h), si ya existe.
    Salida:
        Lista de coordenadas (x1, y1, x2, y2).
    Para que sirve:
        Permite reutilizar las mismas zonas para analisis y debug visual.
    """
    if bbox_dino is None:
        x1 = int(ancho * 0.08)
        x2 = int(ancho * 0.28)
        y1 = int(alto * 0.55)
        y2 = int(alto * 0.68)
        return [(x1, y1, x2, y2)]

    x, y, w, h = bbox_dino
    margen_x = int(w * 0.35)
    alto_muestra = max(8, int(h * 0.18))

    x1 = x + margen_x
    x2 = x + w - margen_x
    y1 = y + h + int(h * 0.25)
    y2 = y1 + alto_muestra

    x1 = max(0, x1)
    x2 = min(ancho, x2)
    y1 = max(0, y1)
    y2 = min(alto, y2)

    return [(x1, y1, x2, y2)]


def dibujar_zonas_muestreo(frame, bbox_dino=None, color=(255, 0, 255)):
    """
    Dibuja las zonas de muestreo claro/oscuro sobre el frame.

    Entrada:
        frame: imagen BGR donde se dibujan los cuadros.
        bbox_dino: posicion del dino para ubicar la muestra debajo de el.
        color: color BGR de los cuadros.
    Salida:
        El mismo frame marcado.
    Para que sirve:
        Ayuda a observar que parte de la pantalla se usa para decidir el fondo.
    """
    alto, ancho = frame.shape[:2]

    for indice, (x1, y1, x2, y2) in enumerate(obtener_zonas_muestreo(alto, ancho, bbox_dino), start=1):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texto_x = min(ancho - 120, x2 + 8)
        cv2.putText(frame, f"M{indice} fondo", (texto_x, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def dibujar_estado_fondo(frame, gris, bbox_dino=None):
    """
    Dibuja los valores de la muestra de fondo en el frame.

    Entrada:
        frame: imagen BGR marcada.
        gris: frame en escala de grises.
        bbox_dino: posicion del dino, si existe.
    Salida:
        El mismo frame con texto de debug.
    Para que sirve:
        Ver en pantalla si el sistema decidio fondo CLARO u OSCURO.
    """
    medicion = medir_fondo(gris, bbox_dino=bbox_dino)
    estado = "CLARO" if medicion["es_claro"] else "OSCURO"
    texto = (
        f"Fondo: {estado} "
        f"med={medicion['mediana']:.1f} "
        f"pct={medicion['porcentaje_claro']:.2f}"
    )
    alto, ancho = frame.shape[:2]
    cv2.putText(frame, texto, (max(10, ancho - 430), 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return frame


def binarizar_adaptativo(img_gris, umbral=127, umbral_fondo_claro=220, bbox_dino=None):
    if fondo_es_claro(img_gris, bbox_dino=bbox_dino):
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
    umbral_blanco=201,
    bbox_dino=None
):
    """
    Procesa un frame:
    Original -> escala de grises -> binarizacion segun el fondo
    """

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if fondo_es_claro(gris, bbox_dino=bbox_dino):
        binario = binarizar_grises(gris, umbral_negro, umbral_blanco)
    else:
        binario = binarizar_normal(gris, umbral)

    return gris, binario
