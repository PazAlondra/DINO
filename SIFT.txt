import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Cargar imágenes
# -----------------------------
objeto = cv2.imread("img\cactus1A.png")      # imagen pequeña
escena = cv2.imread("img\image.png")      # imagen grande

if objeto is None or escena is None:
    print("Error cargando imágenes")
    exit()

# Convertir a grises
objeto_gray = cv2.cvtColor(objeto, cv2.COLOR_BGR2GRAY)
escena_gray = cv2.cvtColor(escena, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Crear detector SIFT
# -----------------------------
sift = cv2.SIFT_create()

# Detectar puntos clave y descriptores
kp1, des1 = sift.detectAndCompute(objeto_gray, None)
kp2, des2 = sift.detectAndCompute(escena_gray, None)

# -----------------------------
# Matcher
# -----------------------------
bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

# -----------------------------
# Lowe ratio test
# -----------------------------
good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print("Matches buenos:", len(good))

# -----------------------------
# Umbral mínimo
# -----------------------------
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    # Homografía
    M, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    h, w = objeto_gray.shape

    pts = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
    ]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)

    escena_detectada = escena.copy()

    cv2.polylines(
        escena_detectada,
        [np.int32(dst)],
        True,
        (0, 255, 0),
        3
    )

    print("Objeto encontrado")

else:
    escena_detectada = escena.copy()
    print("Objeto NO encontrado")

# -----------------------------
# Mostrar matches
# -----------------------------
img_matches = cv2.drawMatches(
    objeto,
    kp1,
    escena_detectada,
    kp2,
    good,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("SIFT Matching")
plt.axis("off")
plt.show()