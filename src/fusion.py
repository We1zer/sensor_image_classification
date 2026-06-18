import numpy as np


def pulse_quality_weight(local_coverage: float, amp_to_noise: float) -> float:
    cov = max(float(local_coverage), 1e-3)
    snr = max(float(amp_to_noise), 1e-3)
    return cov * snr


def fuse_majority(probs: np.ndarray, weights: np.ndarray | None = None) -> int:
    probs = np.atleast_2d(probs)
    n_classes = probs.shape[1]
    preds = np.argmax(probs, axis=1)
    counts = np.bincount(preds, minlength=n_classes)
    top = np.flatnonzero(counts == counts.max())
    if len(top) == 1:
        return int(top[0])
    masked = np.full(n_classes, -np.inf)
    masked[top] = probs.sum(axis=0)[top]
    return int(np.argmax(masked))


def fuse_mean_proba(probs: np.ndarray, weights: np.ndarray | None = None) -> int:
    probs = np.atleast_2d(probs)
    return int(np.argmax(probs.mean(axis=0)))


def fuse_weighted(probs: np.ndarray, weights: np.ndarray | None = None) -> int:
    probs = np.atleast_2d(probs)
    if weights is None:
        return fuse_mean_proba(probs)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != probs.shape[0] or not np.isfinite(w).any() or w.sum() <= 0:
        return fuse_mean_proba(probs)
    w = w / w.sum()
    return int(np.argmax((probs * w[:, None]).sum(axis=0)))


STRATEGIES = {
    "majority": fuse_majority,
    "mean_proba": fuse_mean_proba,
    "weighted": fuse_weighted,
}


def fuse(probs: np.ndarray, strategy: str = "weighted",
         weights: np.ndarray | None = None) -> int:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Невідома стратегія fusion: {strategy}. "
            f"Доступні: {list(STRATEGIES)}")
    return STRATEGIES[strategy](probs, weights)


def fuse_proba(probs: np.ndarray, strategy: str = "weighted",
               weights: np.ndarray | None = None) -> np.ndarray:
    probs = np.atleast_2d(probs)
    n_classes = probs.shape[1]
    if strategy == "mean_proba":
        return probs.mean(axis=0)
    if strategy == "weighted":
        if weights is None:
            return probs.mean(axis=0)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != probs.shape[0] or w.sum() <= 0 or not np.isfinite(w).any():
            return probs.mean(axis=0)
        w = w / w.sum()
        return (probs * w[:, None]).sum(axis=0)
    if strategy == "majority":
        preds = np.argmax(probs, axis=1)
        counts = np.bincount(preds, minlength=n_classes).astype(float)
        return counts / counts.sum() if counts.sum() > 0 else probs.mean(axis=0)
    raise ValueError(f"Невідома стратегія fusion: {strategy}.")
