import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

WINDOW       = 24
ALPHA        = 1.0
DYADIC_ORDER = 0


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------

def _make_paths(df, window):
    vals = df.values.astype(np.float64)
    t    = np.linspace(0.0, 1.0, window)
    paths, dates = [], []
    for end in range(window - 1, len(vals)):
        w = vals[end - window + 1: end + 1]
        paths.append(np.column_stack([t, w]))
        dates.append(df.index[end])
    return np.array(paths), pd.DatetimeIndex(dates)


def _make_test_path(train_df, test_df, window):
    context = pd.concat([train_df.iloc[-(window - 1):], test_df])
    vals    = context.values.astype(np.float64)
    t       = np.linspace(0.0, 1.0, window)
    return np.column_stack([t, vals])[np.newaxis, ...]  # (1, window, D+1)


# ---------------------------------------------------------------------------
# RBF static kernel  —  mirrors static_kernels.py used in sigkernel.py
# ---------------------------------------------------------------------------

def _rbf_gram(X, Y, sigma):
    """
    X : (A, M, D)  Y : (B, N, D)
    Returns (A, B, M, N) pairwise RBF kernel evaluations.
    """
    X_e = X.unsqueeze(1).unsqueeze(3)   # (A, 1, M, 1, D)
    Y_e = Y.unsqueeze(0).unsqueeze(2)   # (1, B, 1, N, D)
    sq  = ((X_e - Y_e) ** 2).sum(-1)   # (A, B, M, N)
    return torch.exp(-sq / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Signature kernel PDE  —  naive solver from sigkernel.py (SigKernel_naive /
# SigKernelGramMat_naive) with dyadic_order=0, ported to pure PyTorch so no
# Cython / CUDA backends are required.
# ---------------------------------------------------------------------------

def _sig_kernel_gram(G_static):
    """
    G_static : (A, B, M, N)  —  static kernel Gram matrix over path steps
    Returns  : (A, B)         —  signature kernel Gram matrix
    """
    A, B, M, N = G_static.shape

    # finite-difference increments  (matches sigkernel.py line 715-716)
    G = (G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1]
       - G_static[:, :, 1:, :-1] - G_static[:, :, :-1, 1:])

    # initialise kernel table with boundary conditions  (K[:,0,:]=1, K[:,:,0]=1)
    K = torch.ones((A, B, M, N), dtype=G_static.dtype)

    # PDE recurrence  (matches the non-naive branch in SigKernelGramMat_naive)
    for i in range(M - 1):
        for j in range(N - 1):
            inc = G[:, :, i, j]
            K[:, :, i+1, j+1] = (
                (K[:, :, i+1, j] + K[:, :, i, j+1])
                * (1.0 + 0.5 * inc + (1.0 / 12.0) * inc ** 2)
                - K[:, :, i, j]
                * (1.0 - (1.0 / 12.0) * inc ** 2)
            )

    return K[:, :, -1, -1]


# ---------------------------------------------------------------------------
# Model function
# ---------------------------------------------------------------------------

def sig_kernel_model_function(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    y_train:  pd.Series,
):
    window = WINDOW

    # --- impute NaNs ---
    medians     = train_df.median()
    train_clean = train_df.fillna(medians)
    test_clean  = test_df.fillna(medians)

    # --- scale features (fit on train only, matching SVM convention) ---
    scaler      = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_clean),
        index   = train_clean.index,
        columns = train_clean.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_clean),
        index   = test_clean.index,
        columns = test_clean.columns,
    )

    # --- build path tensors ---
    X_paths_np, path_dates = _make_paths(train_scaled, window)
    X_test_np              = _make_test_path(train_scaled, test_scaled, window)

    # --- align targets to path endpoints ---
    y_aligned = y_train.reindex(path_dates)
    valid     = y_aligned.notna().values
    X_paths_np = X_paths_np[valid]
    y_fit      = y_aligned.dropna().values

    # --- convert to torch float64 ---
    X_train = torch.tensor(X_paths_np, dtype=torch.float64)  # (n, window, D+1)
    X_test  = torch.tensor(X_test_np,  dtype=torch.float64)  # (1, window, D+1)

    # --- sigma: median heuristic over flattened training paths ---
    X_flat   = X_train.reshape(len(X_train), -1)
    sq_dists = torch.cdist(X_flat, X_flat).pow(2)
    sigma    = sq_dists[sq_dists > 0].median().sqrt().item()
    sigma    = max(sigma, 1e-6)

    # --- compute Gram matrices via RBF static kernel + signature PDE ---
    K_train = _sig_kernel_gram(_rbf_gram(X_train, X_train, sigma)).numpy()  # (n, n)
    K_test  = _sig_kernel_gram(_rbf_gram(X_test,  X_train, sigma)).numpy()  # (1, n)

    # --- KernelRidge with precomputed kernel ---
    model = KernelRidge(alpha=ALPHA, kernel="precomputed")
    model.fit(K_train, y_fit)
    y_pred = model.predict(K_test)

    return y_pred, test_df.index
