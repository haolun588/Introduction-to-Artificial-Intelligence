#!/usr/bin/env python3
"""Basic data loading utilities for the HW1 pal dataset.

Every pal is addressed by a numeric id (0..286), assigned in alphabetical
order by pal_id. Row i of features.npy corresponds to id i.

Files in this folder:
  features.npy            - (n_samples, 60) float32 feature matrix, row i = id i
                             (color feature + CLIP image feature, standardized)
                             This is the only file the default pipeline uses.
  sample_submission.csv    - id,label template (written by clustering,
                             called from __main__ below) showing the exact format
                             a Kaggle submission should have

  Extra files (optional, for bonus exploration -- not used by default):
  pal_images/<id>.png     - artwork, one file per pal, named by numeric id
  metadata.json            - list of per-pal numeric attributes
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent


def load_features(path=None):
    """Returns (n_samples, 60) float32 array; row i == pal with id i."""
    path = path or DATA_DIR / 'features.npy'
    return np.load(path)


def load_metadata(path=None):
    """Bonus: returns a list of dicts, one per pal, ordered by id (sorted ascending)."""
    path = path or DATA_DIR / 'metadata.json'
    with open(path, encoding='utf-8') as f:
        rows = json.load(f)
    return sorted(rows, key=lambda r: r['id'])


def load_image(pal_id, images_dir=None):
    """Bonus: pal_id here is the numeric id (int)."""
    images_dir = images_dir or DATA_DIR / 'pal_images'
    return Image.open(Path(images_dir) / f'{pal_id}.png')


def load_all():
    """Load features, indexed by id (0..n_samples-1)."""
    features = load_features()
    ids = list(range(features.shape[0]))
    return ids, features


class KMeansFromScratch:
    """K-means clustering algorithm implemented purely in NumPy (no sklearn.cluster)."""
    def __init__(self, k=9, max_iter=300, n_init=20, seed=0):
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
                # Distance matrix of shape (n_samples, k)
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


def clustering(path=None, label_col='label', k=9, seed=0):
    """Perform K-means clustering on features.npy with K=9 and save predictions."""
    path = path or DATA_DIR / 'sample_submission.csv'
    features = load_features()
    ids = list(range(features.shape[0]))

    kmeans = KMeansFromScratch(k=k, n_init=20, seed=seed)
    labels = kmeans.fit_predict(features)

    df = pd.DataFrame({'id': ids, label_col: labels})
    df.to_csv(path, index=False)
    return path


if __name__ == '__main__':
    ids, features = load_all()
    print(f'{len(ids)} pals, feature dim = {features.shape[1]}')

    out_path = clustering(k=9, seed=0)
    print(f'Wrote predictions to {out_path}')

    try:
        from scoring import score_submission
        scores = score_submission(out_path)
        print(f"Silhouette Score (K=9): {scores['silhouette']:.4f} (Baseline: > 0.06)")
    except Exception as e:
        print(f"Scoring error: {e}")

