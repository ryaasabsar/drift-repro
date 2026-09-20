"""Basic numerical error statistics shared by the standalone benchmark."""
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metrics(a, b, atol=0., rtol=0.):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and a.size > 0, 'Metrics require equal, nonempty shapes')
    x, y = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    err = np.abs(x[finite] - y[finite])
    close = np.zeros(x.shape, dtype=bool)
    close[finite] = err <= atol + rtol * np.abs(y[finite])
    return {'nonfinite_pairs': int((~finite).sum()),
            'max_absolute_error': float(err.max()) if err.size else None,
            'mean_absolute_error': float(err.mean()) if err.size else None,
            'relative_l2_error': float(np.linalg.norm(err) / max(np.linalg.norm(y[finite]), 1e-300)) if err.size else None,
            'within_tolerance_percent': float(close.mean() * 100),
            'within_tolerance': bool(close.all()), 'atol': atol, 'rtol': rtol}
