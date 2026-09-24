#!/usr/bin/env python3
"""HW1 Part 2 - Unsupervised Learning (K-Means with Balanced Feature Scaling & HSV Color Features)
Student ID: 112511120
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent


def load_features(path=None):
    """Returns (n_samples, 60) float32 array; row i == pal with id i."""
    path = path or DATA_DIR / 'features.npy'
    if not Path(path).exists() and (DATA_DIR.parent / 'features.npy').exists():
        path = DATA_DIR.parent / 'features.npy'
    return np.load(path)


def extract_color_features(n_samples):
    """Extract foreground HSV color histogram and statistics from pal_images."""
    images_dir = DATA_DIR / 'pal_images'
    if not images_dir.exists() and (DATA_DIR.parent / 'pal_images').exists():
        images_dir = DATA_DIR.parent / 'pal_images'

    color_features = []
    for pal_id in range(n_samples):
        img_path = images_dir / f'{pal_id}.png'
        if not img_path.exists():
            color_features.append(np.zeros(18, dtype=np.float32))
            continue

        img = Image.open(img_path).convert('RGBA')
        arr = np.array(img)

        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        mask = (alpha > 50) & ~((rgb[:, :, 0] > 240) & (rgb[:, :, 1] > 240) & (rgb[:, :, 2] > 240))

        if np.sum(mask) == 0:
            fg_rgb = rgb.reshape(-1, 3)
        else:
            fg_rgb = rgb[mask]

        fg_float = fg_rgb / 255.0
        r_ch, g_ch, b_ch = fg_float[:, 0], fg_float[:, 1], fg_float[:, 2]
        maxc = np.maximum(np.maximum(r_ch, g_ch), b_ch)
        minc = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        v = maxc
        deltac = maxc - minc
        s = np.where(maxc == 0, 0, deltac / (maxc + 1e-7))

        h = np.zeros_like(r_ch)
        idx = deltac > 0
        rc = (maxc - r_ch) / (deltac + 1e-7)
        gc = (maxc - g_ch) / (deltac + 1e-7)
        bc = (maxc - b_ch) / (deltac + 1e-7)

        h_r = (bc - gc) % 6
        h_g = gc - rc + 2
        h_b = rc - bc + 4

        h[idx & (maxc == r_ch)] = h_r[idx & (maxc == r_ch)]
        h[idx & (maxc == g_ch)] = h_g[idx & (maxc == g_ch)]
        h[idx & (maxc == b_ch)] = h_b[idx & (maxc == b_ch)]
        h = (h / 6.0) % 1.0

        h_hist, _ = np.histogram(h, bins=12, range=(0, 1), weights=s)
        h_hist = h_hist / (np.sum(h_hist) + 1e-7)

        mean_h, mean_s, mean_v = np.mean(h), np.mean(s), np.mean(v)
        std_h, std_s, std_v = np.std(h), np.std(s), np.std(v)

        feat = np.concatenate([h_hist, [mean_h, mean_s, mean_v, std_h, std_s, std_v]])
        color_features.append(feat)

    return np.array(color_features, dtype=np.float32)


def zscore(mat):
    """Zero-mean, unit-variance normalization."""
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0) + 1e-7
    return (mat - mean) / std


class KMeansFromScratch:
    """K-means clustering algorithm implemented purely in NumPy."""
    def __init__(self, k=9, max_iter=300, n_init=30, seed=0):
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init
        self.seed = seed
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _init_centroids_kmeans_pp(self, X, rng):
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        centroids[0] = X[rng.choice(n_samples)]

        for i in range(1, self.k):
            dists = np.min([np.sum((X - c)**2, axis=1) for c in centroids[:i]], axis=0)
            probs = dists / (np.sum(dists) + 1e-10)
            cumprobs = np.cumsum(probs)
            r = rng.random()
            next_idx = np.searchsorted(cumprobs, r)
            next_idx = min(next_idx, n_samples - 1)
            centroids[i] = X[next_idx]
        return centroids

    def fit_predict(self, X):
        best_inertia = float('inf')
        best_labels = None
        best_centroids = None

        for init_idx in range(self.n_init):
            rng = np.random.RandomState(self.seed + init_idx)
            centroids = self._init_centroids_kmeans_pp(X, rng)

            for iteration in range(self.max_iter):
                dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
                labels = np.argmin(dists, axis=1)

                new_centroids = np.array([
                    X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j]
                    for j in range(self.k)
                ])

                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            dists = np.min([np.sum((X - centroids[j])**2, axis=1) for j in range(self.k)], axis=0)
            inertia = np.sum(dists)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        self.inertia_ = best_inertia
        self.labels = best_labels
        self.centroids = best_centroids
        return best_labels


def clustering(path=None, label_col='label', k=9, alpha=0.5, seed=0):
    """Perform K-means clustering on balanced feature fusion."""
    path = path or DATA_DIR / '112511120_2.csv'
    features = load_features()
    n_samples = features.shape[0]
    ids = list(range(n_samples))

    features_norm = zscore(features)
    color_feats = extract_color_features(n_samples)
    color_norm = zscore(color_feats)

    combined_features = np.hstack([features_norm, alpha * color_norm])

    kmeans = KMeansFromScratch(k=k, n_init=30, seed=seed)
    labels = kmeans.fit_predict(combined_features)

    df = pd.DataFrame({'id': ids, label_col: labels})
    df.to_csv(path, index=False)
    return path


if __name__ == '__main__':
    features = load_features()
    print(f'Part 2 (Balanced K-Means): Loaded {features.shape[0]} pals, feature dim = {features.shape[1]}')

    out_path = clustering(k=9, alpha=0.5, seed=0)
    print(f'Wrote predictions to {out_path}')
