"""Standalone NFW numerical helpers, copied from ITAMAE 23d01e8.

MIT License

Copyright (c) 2026 Shunichi Horigome

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from scipy.optimize import brentq


def _real_numeric(value, name):
    """Reject non-real coordinates before any float cast can discard data."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values.")
    return np.asarray(array, dtype=float)


def nfw_mass_function(x):
    """Return ``ln(1+x)-x/(1+x)`` for finite nonnegative ``x``.

    A small-radius series avoids subtracting two nearly equal terms. Values
    smaller than the floating-point subnormal range can still underflow.
    """
    x = _real_numeric(x, "NFW radius ratio")
    if not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("NFW radius ratio must be finite and nonnegative.")
    result = np.empty_like(x)
    small = x < 0.01
    # f(x) = sum_{n=2}^infinity (-1)^n (n-1)/n x^n.
    coefficient = [(-1.0) ** n * (n - 1.0) / n for n in range(2, 11)]
    result[small] = x[small] ** 2 * np.polynomial.polynomial.polyval(x[small], coefficient)
    result[~small] = np.log1p(x[~small]) - x[~small] / (1.0 + x[~small])
    return result


def invert_nfw_mass_function(y):
    """Invert the NFW enclosed-mass function over representable positive radii.

    Solve in log-radius so an absolute tolerance in radius cannot erase a
    small, positive solution. The output retains the input shape.
    """
    y = _real_numeric(y, "Enclosed-mass function")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("Enclosed-mass function values must be finite and nonnegative.")
    max_radius = np.finfo(float).max
    max_log_radius = np.log(max_radius)
    max_enclosed = float(nfw_mass_function(max_radius))
    if np.any(y > max_enclosed):
        raise ValueError(
            "The inverse NFW radius is outside the representable floating-point range."
        )

    def one(value: float) -> float:
        if value == 0.0:
            return 0.0
        if value == max_enclosed:
            return max_radius
        # f(x) <= x^2/2 and f(exp(y+1)) >= y bound the positive root.
        lower = 0.5 * (np.log(2.0) + np.log(value)) - np.log(2.0)
        upper = min(value + 1.0, max_log_radius)

        def residual(log_radius):
            radius = max_radius if log_radius == max_log_radius else np.exp(log_radius)
            enclosed = float(nfw_mass_function(radius))
            # The upper bracket can be far above a subnormal target. Preserve
            # its sign without overflowing a ratio that is used only to bracket.
            if value < 1.0 and enclosed > max_radius * value:
                return max_radius
            return enclosed / value - 1.0

        root = brentq(residual, lower, upper, xtol=5.0e-14, rtol=4.0 * np.finfo(float).eps)
        return float(np.exp(root))

    # A safeguarded vector Newton solve avoids a Python/Brent call for every
    # catalog node. The scalar log-radius solver remains the fallback for
    # extreme floating-point values or any unconverged element.
    out = np.zeros_like(y)
    active = (y >= 1e-200) & (y <= 600.)
    values = y[active]
    if values.size:
        lower = .5*(np.log(2.)+np.log(values))-np.log(2.)
        upper = values+1.
        radius_guess = np.sqrt(2*values)+4*values/3
        log_radius = np.where(values<.5,np.log(radius_guess),upper)
        converged = np.zeros(values.shape,dtype=bool)
        for _ in range(30):
            radius = np.exp(log_radius)
            residual = nfw_mass_function(radius)-values
            derivative = (radius/(1+radius))**2
            correction = residual/derivative
            converged |= np.abs(correction) <= 2e-13
            if np.all(converged):
                break
            lower = np.where(residual<0,log_radius,lower)
            upper = np.where(residual>0,log_radius,upper)
            proposed = log_radius-correction
            inside = (proposed>lower) & (proposed<upper)
            proposed = np.where(inside,proposed,.5*(lower+upper))
            log_radius = np.where(converged,log_radius,proposed)
        solved = np.exp(log_radius)
        if np.any(~converged):
            solved[~converged] = [one(float(value)) for value in values[~converged]]
        out[active] = solved
    extreme = (y>0) & ~active
    if np.any(extreme):
        out[extreme] = [one(float(value)) for value in y[extreme]]
    return float(out) if out.ndim == 0 else out

