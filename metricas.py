import cv2
import os
import time
from datetime import datetime

from preprocesamiento import binarizar_adaptativo


def crear_metricas(modo):
    """
    Crea el estado inicial de metricas de una partida.

    Entrada:
        modo: metodo de deteccion usado en inicio.py.
    Salida:
        Diccionario mutable con tiempos, saltos y modo.
    Para que sirve:
        Guarda informacion de la partida sin ensuciar el loop principal.
    """
    ahora = time.time()
    return {
        "modo": modo,
        "inicio": ahora,
        "primer_salto": None,
        "ultimo_salto": None,
        "saltos": 0,
        "puntaje_final": None,
        "eventos": [],
    }


def registrar_salto(metricas):
    """
    Registra un salto dentro del estado de metricas.

    Entrada:
        metricas: diccionario creado con crear_metricas.
    Salida:
        El mismo diccionario actualizado.
    Para que sirve:
        Permite calcular cuanto tiempo aguanto desde el primer salto al ultimo.
    """
    ahora = time.time()

    if metricas["primer_salto"] is None:
        metricas["primer_salto"] = ahora

    metricas["ultimo_salto"] = ahora
    metricas["saltos"] += 1

    return metricas


def registrar_evento(metricas, tipo_detectado, accion):
    """
    Registra un evento de deteccion y la accion realizada.

    Entrada:
        metricas: diccionario creado con crear_metricas.
        tipo_detectado: cactus, pajaro_bajo, pajaro_alto u otro texto.
        accion: saltar, agacharse o None.
    Salida:
        El mismo diccionario actualizado.
    Para que sirve:
        Permite auditar en el txt que obstaculo se detecto y que hizo el bot.
    """
    if tipo_detectado is None and accion is None:
        return metricas

    metricas["eventos"].append({
        "tiempo": time.time() - metricas["inicio"],
        "tipo": tipo_detectado,
        "accion": accion,
    })

    return metricas


def cargar_mascaras_numeros(carpeta_numeros="img/numeros"):
    """
    Carga mascaras de numeros para reconocer puntaje.

    Entrada:
        carpeta_numeros: carpeta con archivos 0.png, 1.png, ..., 9.png.
    Salida:
        Diccionario {"0": imagen_binaria, ..., "9": imagen_binaria}.
    Para que sirve:
        Da soporte al reconocimiento de puntaje por mascaras de numeros.
    """
    templates = {}

    if not os.path.isdir(carpeta_numeros):
        return templates

    for archivo in os.listdir(carpeta_numeros):
        nombre, extension = os.path.splitext(archivo)

        if nombre not in "0123456789" or extension.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue

        ruta = os.path.join(carpeta_numeros, archivo)
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        templates[nombre] = binarizar_adaptativo(img)

    return templates


def reconocer_puntaje_por_mascaras(frame, templates_numeros):
    """
    Intenta reconocer el puntaje del juego con mascaras de numeros.

    Entrada:
        frame: imagen BGR capturada.
        templates_numeros: salida de cargar_mascaras_numeros.
    Salida:
        Entero con el puntaje o None si no se pudo reconocer.
    Para que sirve:
        Permite guardar el puntaje final en metricas.txt cuando existan mascaras numericas.
    """
    if not templates_numeros:
        return None

    alto, ancho = frame.shape[:2]
    roi = frame[int(alto * 0.15):int(alto * 0.40), int(ancho * 0.55):int(ancho * 0.98)]

    if roi.size == 0:
        return None

    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binario = binarizar_adaptativo(gris)

    contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)

        if w < 3 or h < 6:
            continue

        if h > binario.shape[0] * 0.6:
            continue

        cajas.append((x, y, w, h))

    if not cajas:
        return None

    cajas.sort(key=lambda caja: caja[0])
    cajas = cajas[-5:]
    digitos = []

    for x, y, w, h in cajas:
        recorte = binario[y:y + h, x:x + w]
        mejor_digito = None
        mejor_valor = -1

        for digito, template in templates_numeros.items():
            template_redim = cv2.resize(template, (w, h), interpolation=cv2.INTER_NEAREST)
            resultado = cv2.matchTemplate(recorte, template_redim, cv2.TM_CCOEFF_NORMED)
            valor = cv2.minMaxLoc(resultado)[1]

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_digito = digito

        if mejor_digito is None or mejor_valor < 0.25:
            return None

        digitos.append(mejor_digito)

    return int("".join(digitos))


def guardar_metricas(metricas, carpeta_salida="metricas"):
    """
    Guarda las metricas de la partida en un archivo txt.

    Entrada:
        metricas: diccionario creado con crear_metricas.
        carpeta_salida: carpeta donde se guardara el txt.
    Salida:
        Ruta del archivo generado.
    Para que sirve:
        Deja registro de modo, saltos, tiempo de partida y puntaje si existe.
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta_salida, f"metricas_{fecha}.txt")

    primer_salto = metricas["primer_salto"]
    ultimo_salto = metricas["ultimo_salto"]

    if primer_salto is not None and ultimo_salto is not None:
        tiempo_saltando = ultimo_salto - primer_salto
    else:
        tiempo_saltando = 0

    tiempo_total = time.time() - metricas["inicio"]

    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("Metricas de partida\n")
        archivo.write("===================\n")
        archivo.write(f"Fecha: {datetime.now().isoformat(timespec='seconds')}\n")
        archivo.write(f"Modo: {metricas['modo']}\n")
        archivo.write(f"Saltos: {metricas['saltos']}\n")
        archivo.write(f"Tiempo total: {tiempo_total:.2f} segundos\n")
        archivo.write(f"Tiempo primer-ultimo salto: {tiempo_saltando:.2f} segundos\n")
        archivo.write(f"Puntaje final: {metricas['puntaje_final']}\n")
        archivo.write("\nEventos detectados\n")
        archivo.write("------------------\n")

        if not metricas["eventos"]:
            archivo.write("Sin eventos registrados.\n")
        else:
            for evento in metricas["eventos"]:
                archivo.write(
                    f"{evento['tiempo']:.2f}s | "
                    f"detectado={evento['tipo']} | "
                    f"accion={evento['accion']}\n"
                )

    print(f"Metricas guardadas en: {ruta}")
    return ruta
