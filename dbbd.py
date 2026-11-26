"""
DBBD (Deterministic Binary Block-Diagonal) compression -- identical to the
operator used in the deployed Raspberry Pi pipeline (Approach 1):

    Y_{m,n}^{(c)} = A X_{m,n}^{(c)} A^T,   A in {0,1}^{Mp x Np},  Np = 32, Mp = Np / b

applied to every 32x32 patch of every colour channel of the 224x224 frame,
followed by GLOBAL min-max normalisation (over the whole RGB tensor) and
8-bit scaling.  b = 2 gives the 112x112 image used in the existing system;
b = 4 gives 56x56; b = 1 means "no compression" (only the min-max stretch).

Because every row of A is a contiguous run of b ones, A X A^T is exactly a
b x b block SUM (box down-sampling).  `dbbd_fast` uses that identity and the
self-test at the bottom checks it against the matrix form.
"""
import numpy as np

NP_PATCH = 32  # patch size N_p used by the deployed pipeline


def dbbd_matrix(n_p: int = NP_PATCH, m_p: int = 16) -> np.ndarray:
    """Deterministic block matrix A (m_p x n_p): row i has ones in columns [i*b, (i+1)*b)."""
    if n_p % m_p:
        raise ValueError("M_p must divide N_p")
    b = n_p // m_p
    A = np.zeros((m_p, n_p), dtype=np.float64)
    for i in range(m_p):
        A[i, i * b:(i + 1) * b] = 1.0
    return A


def _normalise_8bit(y: np.ndarray) -> np.ndarray:
    """Global min-max over the whole tensor, then 0..255.  Truncation to uint8
    mirrors `.astype(np.uint8)` in the deployed code; change to np.rint if your
    code rounds."""
    lo, hi = float(y.min()), float(y.max())
    if hi > lo:
        y = (y - lo) / (hi - lo)
    else:
        y = np.zeros_like(y)
    return (255.0 * y).astype(np.uint8)


def dbbd_matrix_form(x: np.ndarray, b: int = 2, n_p: int = NP_PATCH, normalise: bool = True):
    """Reference implementation, patch by patch: Y = A X A^T."""
    x = np.asarray(x, dtype=np.float64)
    H, W, C = x.shape
    if b == 1:
        y = x.copy()
    else:
        if H % n_p or W % n_p or n_p % b:
            raise ValueError("image size must be a multiple of N_p and b must divide N_p")
        m_p = n_p // b
        A = dbbd_matrix(n_p, m_p)
        y = np.zeros((H // b, W // b, C))
        for c in range(C):
            for r in range(H // n_p):
                for q in range(W // n_p):
                    P = x[r * n_p:(r + 1) * n_p, q * n_p:(q + 1) * n_p, c]
                    y[r * m_p:(r + 1) * m_p, q * m_p:(q + 1) * m_p, c] = A @ P @ A.T
    return _normalise_8bit(y) if normalise else y


def dbbd_fast(x: np.ndarray, b: int = 2, normalise: bool = True):
    """Same result as dbbd_matrix_form, via b x b block sums (used for speed)."""
    x = np.asarray(x, dtype=np.float64)
    H, W, C = x.shape
    if b == 1:
        y = x
    else:
        y = x.reshape(H // b, b, W // b, b, C).sum(axis=(1, 3))
    return _normalise_8bit(y) if normalise else y


def dbbd_tf(x, b: int):
    """TensorFlow version for the training pipeline.  x: float32 [H, W, 3] in 0..255.
    Returns float32 [H/b, W/b, 3] in 0..255 (integer-valued, like the uint8 image)."""
    import tensorflow as tf
    if b > 1:
        x = tf.nn.avg_pool2d(x[None], ksize=b, strides=b, padding="VALID")[0] * float(b * b)
    lo = tf.reduce_min(x)
    hi = tf.reduce_max(x)
    x = tf.where(hi > lo, (x - lo) / (hi - lo + 1e-12), tf.zeros_like(x))
    return tf.floor(x * 255.0)


def measurements_per_frame(b: int, side: int = 224, channels: int = 3) -> int:
    return (side // b) * (side // b) * channels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(224, 224, 3)).astype(np.uint8)
    for b in (1, 2, 4, 8):
        a = dbbd_matrix_form(img, b)
        f = dbbd_fast(img, b)
        assert a.shape == f.shape == (224 // b, 224 // b, 3), (a.shape, f.shape)
        assert np.array_equal(a, f), f"mismatch at b={b}"
        print(f"b={b}: output {a.shape}, {measurements_per_frame(b)} values "
              f"({100.0 / b**2:.2f}% of the 224x224x3 frame) -- matrix form == block-sum form")
    print("DBBD self-test passed.")
