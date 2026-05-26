import cv2
import os
import numpy as np


CATEGORIAS = {
    "dino": ["dino"],
    "cactus": ["cactus"],
    "pajaro": ["pajaro"],
    "numeros": [str(i) for i in range(10)],
}


def categoria_archivo(nombre_archivo):
    """
    Clasifica una imagen por nombre de archivo.

    Entrada:
        nombre_archivo: nombre como cactus1A.png o dino.png.
    Salida:
        Nombre de categoria: dino, cactus, pajaro, numeros u otros.
    Para que sirve:
        Decide en que carpeta guardar cada mascara procesada.
    """
    nombre = os.path.splitext(nombre_archivo.lower())[0]

    for categoria, prefijos in CATEGORIAS.items():
        if any(nombre.startswith(prefijo) for prefijo in prefijos):
            return categoria

    return "otros"


def procesar_imagen_a_mascara(ruta, umbral=100):
    """
    Convierte una imagen a mascara binaria manteniendo su tamano original.

    Entrada:
        ruta: archivo de imagen.
        umbral: valor de binarizacion.
    Salida:
        gris: imagen en escala de grises con el tamano original.
        binaria: mascara binaria.
        inversa: mascara invertida.
    Para que sirve:
        Prepara recursos para SIFT, template matching y numeros.
    """
    img = cv2.imread(ruta)

    if img is None:
        return None, None, None

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)
    inversa = cv2.bitwise_not(binaria)

    return gris, binaria, inversa


def procesar_mascaras(
    carpeta_entrada="img",
    carpeta_salida="img/mascaras_procesadas",
    umbral=127
):
    """
    Procesa imagenes fuente y guarda mascaras por categoria.

    Entrada:
        carpeta_entrada: carpeta con imagenes originales.
        carpeta_salida: carpeta raiz de salida.
        umbral: valor de binarizacion.
    Salida:
        None. Escribe archivos en disco.
    Para que sirve:
        Crea carpetas para dino, pajaro, cactus, numeros y otros con mascaras listas.
        Las mascaras conservan el tamano original de cada imagen.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    extensiones = (".jpg", ".jpeg", ".png", ".bmp")

    for archivo in os.listdir(carpeta_entrada):
        if not archivo.lower().endswith(extensiones):
            continue

        ruta = os.path.join(carpeta_entrada, archivo)
        gris, binaria, inversa = procesar_imagen_a_mascara(ruta, umbral)

        if gris is None:
            print(f"No se pudo leer: {archivo}")
            continue

        categoria = categoria_archivo(archivo)
        carpeta_categoria = os.path.join(carpeta_salida, categoria)
        os.makedirs(carpeta_categoria, exist_ok=True)
        nombre, _ = os.path.splitext(archivo)

        cv2.imwrite(os.path.join(carpeta_categoria, f"{nombre}_gris.png"), gris)
        cv2.imwrite(os.path.join(carpeta_categoria, f"{nombre}_binaria.png"), binaria)
        cv2.imwrite(os.path.join(carpeta_categoria, f"{nombre}_inversa.png"), inversa)

        print(f"Procesada: {archivo} -> {categoria}")

    print("Proceso terminado.")


if __name__ == "__main__":
    procesar_mascaras(
        carpeta_entrada="img",
        carpeta_salida="img/mascaras_procesadas",
        umbral=127
    )
