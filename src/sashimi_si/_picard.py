"""SI physical-state adapter to ITAMAE's shared log-mass solver."""

import hashlib
import numpy as np
from itamae.evolution import PicardDomain, PicardMassLossSolver


def physics_key(solver):
    """Fingerprint mutable SI host inputs and its selected cosmology backend.

    Numerical cache fields are excluded. Custom providers with hidden/nested
    state must expose an immutable ``picard_physics_key()`` value covering it.
    This key is process-local, not a serialization or scientific identity.

    Parameters
    ----------
    solver : TidalStrippingSolver
        SI-owned tidal provider preserving its background, host history and SI internal mass
        units.

    Returns
    -------
    tuple
        Process-local identity of all inspected scalar/array/callback inputs plus an
        optional provider key.

    Notes
    -----
    SI supplies its own tidal coefficients and host state; ITAMAE performs the shared
    log-mass integration. The declared validation rectangle is z in [0,7] and
    log10(ma/M_host) in [-24,3]. Full histories preserve the requested epoch axis;
    out-of-domain or unconverged requests trigger a counted direct-ODE fallback.
    """

    def inputs(owner):
        """Fingerprint numerical provider inputs while excluding derived tidal caches.

        Parameters
        ----------
        owner : object
            Solver or cosmology provider whose mutable inputs must invalidate old tidal tables.

        Returns
        -------
        tuple
            Ordered scalar values, array dtype/shape/content hashes and callable identities.

        Notes
        -----
        SI supplies its own tidal coefficients and host state; ITAMAE performs the shared
        log-mass integration. The declared validation rectangle is z in [0,7] and
        log10(ma/M_host) in [-24,3]. Full histories preserve the requested epoch axis;
        out-of-domain or unconverged requests trigger a counted direct-ODE fallback.
        """
        result = []
        for name, value in sorted(getattr(owner, "__dict__", {}).items()):
            if name.startswith(("_picard", "_eps", "eps_")):
                continue
            if isinstance(value, (float, int, str, bool, np.number)):
                result.append((name, repr(value)))
            elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
                result.append(
                    (
                        name,
                        value.dtype.str,
                        value.shape,
                        hashlib.sha256(value.tobytes()).hexdigest(),
                    )
                )
            elif callable(value):
                result.append((name, id(value)))
        return tuple(result)

    result = [inputs(solver)]
    backend = getattr(solver, "itamae_cosmology", None)
    if backend is not None:
        # NativeFlatLCDM is immutable; adapters can also have mutable scalars.
        result.append(("cosmology", id(backend), inputs(backend)))
    if hasattr(solver, "picard_physics_key"):
        result.append(("provider", solver.picard_physics_key()))
    for name in ("Phi", "zetaMz", "Mzvir"):
        value = getattr(solver, name)
        result.append((name, id(getattr(value, "__func__", value))))
    return tuple(result)


def get_picard_solver(solver):
    """Keep SI's calibrated coefficients and units outside the numerical core.

    Parameters
    ----------
    solver : TidalStrippingSolver
        SI-owned tidal provider preserving its background, host history and SI internal mass
        units.

    Returns
    -------
    PicardMassLossSolver
        Shared history integrator bound to the SI host/tidal callbacks and physical-state
        key.

    Notes
    -----
    SI supplies its own tidal coefficients and host state; ITAMAE performs the shared
    log-mass integration. The declared validation rectangle is z in [0,7] and
    log10(ma/M_host) in [-24,3]. Full histories preserve the requested epoch axis;
    out-of-domain or unconverged requests trigger a counted direct-ODE fallback.
    """
    backend = getattr(solver, "_picard_backend", None)
    if backend is None or (backend.z_min, backend.z_max) != (solver.z_min, solver.z_max):
        backend = PicardMassLossSolver(
            lambda z: solver.Phi(z),
            lambda z: solver.zetaMz(z),
            lambda z: solver.Mzvir(z),
            z_min=solver.z_min,
            z_max=solver.z_max,
            state_key=lambda: physics_key(solver),
            domain=PicardDomain((0.0, 7.0), (-24.0, 3.0)),
        )
        solver._picard_backend = backend
    return backend
