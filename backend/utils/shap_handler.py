import shap
import numpy as np


def get_top_features(explainer, x_np, feature_names, top=5):
    """
    Return top `top` features by absolute SHAP value for the first sample in x_np.

    Handles different shapes returned by various SHAP explainers (array or list
    of arrays) and normalizes to a 1D array for the first sample.
    """
    # Normalize feature_names to Python strings
    try:
        fnames = [str(f) for f in feature_names]
    except Exception:
        fnames = list(feature_names)

    # Get raw shap values (may be an array or a list of arrays for multiclass)
    raw = explainer.shap_values(x_np)

    # If a list is returned (e.g., one array per class), take the first element
    if isinstance(raw, list):
        raw0 = raw[0]
    else:
        raw0 = raw

    arr = np.asarray(raw0)

    # arr can be shape (n_samples, n_features) or (n_features,) depending on input
    if arr.ndim == 1:
        sample_shap = arr
    else:
        # take the first sample
        sample_shap = arr[0]

    abs_vals = np.abs(sample_shap)
    top_idx = abs_vals.argsort()[-top:][::-1]

    results = []
    for i in top_idx:
        # Ensure we cast to native python float
        val = sample_shap[i]
        try:
            val = float(val)
        except Exception:
            # If value is array-like, take the first element
            val = float(np.asarray(val).ravel()[0])
        results.append({
            "feature": fnames[i],
            "value": float(val),
        })

    return results
