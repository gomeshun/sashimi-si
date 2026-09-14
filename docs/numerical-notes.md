# Numerical notes for SASHIMI-SI

The original module imports, public classes, tuple order, units and positional
arguments remain supported. See the example notebook for catalog usage and
[tests](../tests/README.md) for installation checks.

## Changes affecting results

- The total cross section is the angular integral of the differential cross
  section: `sigma0_m / (1 + (v / w)**2)`. SIDM evolution uses the effective cross
  section, so this total-cross-section correction does not itself alter a catalog.
- The effective cross section uses equivalent positive quadrature where the
  analytic expression loses precision; unused branches are not evaluated.
- EPS applies physical support before division and retains the finite normalized
  zero-barrier-gap limit. Single-node host quadrature preserves the mass/redshift
  array shape. Invalid active domains raise errors.
- Nodes not formed at accretion retain zero SIDM structural placeholders and a
  false SIDM survival flag. Their existing formation and CDM weight gates apply.
- A safeguarded NFW inverse resolves the existing disruption threshold accurately.

The SIDM evolution equations, coefficients and collapse threshold retain their
definitions. Picard supplies the full post-accretion mass history needed by the
SIDM time integral, including the zero-scattering limit of infinite collapse time.

## Tidal stripping solver

The default is `method="picard"`. Use `method="pert2_shanks"` to select the
previous default explicitly, or `method="dop853"` for direct log-mass integration.
Other existing named solvers remain available. Unknown options raise errors.

SI evaluates the requested mass trajectories together and retains the
catalog's 100 post-accretion times. It uses three nonlinear updates and a fourth
convergence check, with mass interpolation spacing no wider than a quarter dex.

The automatic table envelope is `0 <= z_obs <= z_acc <= 7` and
`-24 <= log10(ma/Mvir(z_acc)) <= 3`. Outside it, or after a failed convergence
check, `PicardFallbackWarning` and `solver._picard_events` record direct-ODE
fallback. Queries do not silently extrapolate; invalid physical input raises.
Caches include the host/background state, final redshift, numerical options and
particle settings where applicable. Wrapper calls rebuild after state changes.

Each variant supplies its own host history, background and stripping coefficients.
The standalone helper retains the MIT notice from SASHIMI-C PR #5, commit
`88ae730762fb153be7a7433bb563b0b8ab3ec2c2`.

## Validation scope

The checked domains met a `1e-3` relative mass-error gate against refined
independent ODE integrations. This is a finite-domain numerical check, not a
bound on changes from older approximations or a physical calibration.
Simulation agreement and downstream inference require separate validation.

The [historical validation report](https://github.com/gomeshun/sashimi-si/blob/fecbd2c33c63d79124bde22bcd0172a434f4e0e6/docs/standalone-maintenance.md)
records the evaluated grids, reference methods, timings and limitations and
links to the full development evidence at that fixed commit. Those generated
reports and intermediate arrays are not needed in a working checkout.
