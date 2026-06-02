"""
RGB -> HSI -> Clustering (K-Means) -> RGB
==========================================
Baseado em:
  - https://machinelearningmastery.com/the-beginners-guide-to-clustering-with-python/
  - https://scikit-learn.org/stable/modules/clustering.html  (secao 2.3.2, K-Means)

Fluxo:
  imagem -> garante RGB -> converte para HSI -> agrupa cores com K-Means
         -> substitui cada pixel pela cor do seu cluster -> volta para RGB -> salva
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# ───────────────────────── RGB → HSI ─────────────────────────
def rgb_para_hsi(img_rgb):
    img = img_rgb.astype(np.float64) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Intensidade
    I = (R + G + B) / 3.0

    # Saturacao
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 0, 1 - (min_rgb / (I + 1e-10)), 0)

    # Matiz (Hue)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)  # normaliza para 0–1

    return np.stack([H, S, I], axis=2)


# ───────────────────────── HSI → RGB ─────────────────────────
def hsi_para_rgb(hsi):
    H = hsi[:, :, 0] * 2 * np.pi
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]

    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)

    # Setor RG: 0 <= H < 120
    idx = (H >= 0) & (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi / 3 - H[idx]))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    # Setor GB: 120 <= H < 240
    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H2 = H[idx] - 2 * np.pi / 3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H2) / np.cos(np.pi / 3 - H2))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    # Setor BR: 240 <= H < 360
    idx = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H3 = H[idx] - 4 * np.pi / 3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H3) / np.cos(np.pi / 3 - H3))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    rgb = np.clip(np.stack([R, G, B], axis=2) * 255, 0, 255).astype(np.uint8)
    return rgb


# ─────────────────── Clustering no espaco HSI ───────────────────
def clusterizar_hsi(hsi, n_clusters):
    """Agrupa as cores com K-Means e devolve a imagem HSI quantizada."""
    altura, largura, _ = hsi.shape

    # Achata para (n_pixels, 3) -> formato que o sklearn espera
    pixels = hsi.reshape(-1, 3)

    # K-Means (scikit-learn)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rotulos = kmeans.fit_predict(pixels)          # cluster de cada pixel
    centros = kmeans.cluster_centers_             # cor media de cada cluster

    # Cada pixel vira a cor do seu cluster
    pixels_quantizados = centros[rotulos]
    return pixels_quantizados.reshape(altura, largura, 3)


# ─────────────────────────── Programa ───────────────────────────
if __name__ == "__main__":
    CAMINHO = "praia-do-patacho.png"

    # 1) Carrega e GARANTE RGB
    img_rgb = np.array(Image.open(CAMINHO).convert("RGB"))
    print(f"Imagem RGB:           shape={img_rgb.shape}, dtype={img_rgb.dtype}")

    # 2) RGB -> HSI
    hsi = rgb_para_hsi(img_rgb)
    print(f"HSI:                  shape={hsi.shape}, dtype={hsi.dtype}")

    # 3) Verificacao do ciclo SEM clustering (deve reconstruir a imagem)
    reconstruida = hsi_para_rgb(hsi)
    Image.fromarray(reconstruida).save("praia_hsi_reconstruida.jpg")
    erro = np.abs(img_rgb.astype(int) - reconstruida.astype(int)).mean()
    print(f"Reconstrucao RGB->HSI->RGB salva (erro medio: {erro:.2f}/255)")

    # 4) RGB -> HSI -> K-Means -> RGB para varios numeros de clusters
    for k in (1, 2, 4):
        hsi_quant = clusterizar_hsi(hsi, n_clusters=k)
        rgb_final = hsi_para_rgb(hsi_quant)
        nome = f"praia_kmeans_k{k}.png"
        Image.fromarray(rgb_final).save(nome)
        print(f"K-Means k={k:>2}  ->  {nome}")

    print("\nConcluido!")
