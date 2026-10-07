# Native SIDM API

`SIDM` is an immutable, reusable specification. Physical settings remain owned
by SI; the existing ITAMAE population pipeline executes each fresh run.

```python
from sashimi_si import SIDM

model = SIDM().configure(
    interaction={"sigma0_cm2_g": 147.1, "velocity_scale_km_s": 24.33,
                 "collapse_time_ratio_cap": 1.1},
    accretion={"mass_nodes": 40, "redshift_step": 0.1, "host_history_nodes": 20},
    concentration={"scatter_dex": 0.128, "quadrature_nodes": 5},
    stripping={"solver": "pert2_shanks"},
    disruption={"ct_threshold": 0.0},
)
catalogs = model.population(
    host_mass_msun=1e12, host_mass_definition="200c", host_mass_redshift=0.0,
    redshift=0.5,
    reference_mass_range_msun=(1e5, 1e8), reference_mass_definition="200c",
    reference_mass_redshift=0.5, accretion_redshift_range=(0.5, 3.0),
)
cdm, sidm = catalogs["cdm_reference"], catalogs["sidm"]
```

## Mass coordinates, histories and paired states

SI's legacy logarithmic mass limits describe a **reference M200c grid at the
output epoch**. They do not describe a fixed accretion-mass grid. The reference
epoch must equal `redshift` (`None` resolves to that epoch). No misleading
`accretion_mass_range_msun` alias is accepted. Actual accretion masses follow
the existing backward history and formation gates, and vary with redshift.
The optional upper reference bound remains `0.1 * host_mass_at_z0`, even though
the reference grid is defined at the output epoch; this preserved convention
is recorded explicitly. The default lower reference mass is 1e6 Msun.

Host mass is independently specified by `host_mass_redshift`. The historical
1000-node/3-dex inversion is performed once; z=0 bypasses it exactly. The native
API rejects nonfinite, nonmonotonic or out-of-bracket inversions instead of
extrapolating. Only `200c` input mass definitions are currently supported.

The default result is a read-only mapping containing the existing
`WeightedSubhaloCatalog` objects for both `cdm_reference` and `sidm`. Their node
identity and base/concentration weights are aligned; their survival masks and
final weights remain independent. Use `state="sidm"` or
`state="cdm_reference"` explicitly for a single view. Catalog velocities are
km/s; SI history kernels and the old 27-field tuple retain Mpc/s. The original
formation/prehistory calculations, invalid-accretion policy, and profile
validity selection are unchanged. Catalogs are not a new observable facade.

## Settings and defaults

- `interaction`: Yang2023 prescription; sigma0=147.1 cm2/g, velocity scale=24.33
  km/s, dimensionless collapse-time ratio cap=1.1
- `accretion`: Yang2011 model3, mass_nodes500, redshift_step0.01,
  host_history_nodes200
- `concentration`: Correa2015, scatter_dex0.128, quadrature_nodes20
- `stripping`: CDM tidal prescription, solver `pert2_shanks`,
  interpolation_nodes64, solver_options={}
- `disruption`: existing paired formation/truncation/profile rule, ct_threshold0

`collapse_time_ratio_cap` caps `(t - t_formation) / t_collapse`, controls the
existing derivative freeze and final profile map, and is **not a physical
time**. The native adapter supports [0,1.1], anchored to the existing default
cutoff and tested at both endpoints and an intermediate value. This is a
supported numerical range, not an independent scientific calibration claim.
The old beta=4 parameter is not consumed by this population path; it remains
fixed and is recorded as inactive rather than exposed as an ignored knob.

Supported stripping solvers are `pert0`, `pert1`, `pert2`, `pert2_shanks`,
`pert3`, and `odeint`. Only odeint accepts scalar `rtol`, `atol`, `h0`, `hmax`,
`hmin`, and integer `mxstep`, `mxhnil`, `mxordn`, `mxords`. Positive h0 is invalid
for decreasing-redshift integration; orders are positive and limited to LSODA
maxima. Callbacks/reserved options and incompatible solver settings fail early.
Custom concentration relations/backends are not a supported native capability.

`configure` returns a new specification with partial overrides; caller mappings
and nested options are copied/frozen. An empty solver_options mapping clears
options; a nonempty one partially updates them. `resolved_settings` is recursively
read-only. Every run constructs fresh physical and solver objects, without a
shared host-mass field, previous catalog or execution cache.

## Integration support and provenance

Explicit accretion-redshift support is `(lower, upper]`, lower>=target. Nodes
start at lower+step and never exceed the upper bound; there must be at least two.
Omitted support preserves the exact old arange grid to zmax=5, including its
off-grid endpoint behavior. Physical support and quadrature resolution are
separate, and requested bounds, candidate nodes, formation-selected executed
nodes/support and support policy are recorded separately.
At least two mass nodes and positive quadrature orders are required.

At least one candidate redshift node must retain a mass node satisfying
`z_formation > z_acc`. If formation selection removes every row, the API raises
`ValueError` before solver setup or population execution, for paired and
single-state requests alike. Choose a lower accretion-redshift range or a
reference-mass range with formed halos. This condition is rejected explicitly;
the API does not return empty catalogs for it.

Metadata stores canonical effective settings and their stable configuration
hash, reference and host epochs, mass definitions/ranges, fixed history controls,
solver options, state identity and unit contracts. It remains JSON serializable.
The original old entry points and migration defaults remain unchanged.

## Verification scope

Independent small pre-edit outputs were built from SI
`ce5a3c11518a609ac056bb136653268b05b1ecd7` and core
`23d01e8758a88b061b87de9e488c38ec89fd8e4f`. They cover paired states, nonzero
redshift, host-reference conversion, changed interaction/cap and explicit ODE.
Stored references are compared with `rtol=5e-12` and `atol=1e-300`; old/new paths
in one runtime must agree exactly. Original equation/formation/velocity and
frozen reference tests remain in the suite. The reference-grid semantics are
SI-owned, not a generic alias for another variant's accretion coordinate.
