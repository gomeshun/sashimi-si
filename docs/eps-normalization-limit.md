# Finite normalized Yang kernel at zero barrier gap

The host-quadrature convergence sweep failed at 64 points: the 1000-point
redshift search assigns the same collapse barrier to 57 active low-host-mass
nodes. Their raw Hermite weights are 5.54e-49 to 3.42e-39. The numerator and
normalization both vanish, although their ratio has a finite limit. The prior
guard correctly raised instead of silently replacing the result. The original
arrays, parameters and failed regression are in `validation/eps-zero-gap`.

Let delta be the barrier difference, dS the subhalo variance gap, and dSmin the
gap at the support limit. The normalized Yang (2011) Eq. (14) kernel is

`delta * exp(-delta²/(2*dS)) / (sqrt(2*pi)*dS**1.5*erf(x))`,
where `x=delta/sqrt(2*dSmin)`.

Since `erf(x)=(2*x/sqrt(pi))*integral_0^1 exp(-x²*t²) dt`, its zero-gap limit is
`sqrt(dSmin)/(2*dS**1.5)`. For x²<1e-8 the implementation uses the denominator
series `1-x²/3+x⁴/10-x⁶/42+x⁸/216`; the first omitted term is below 8e-44
at the switch. The ordinary branch preserves the previous arithmetic.
Negative barrier gaps or nonpositive variance gaps still fail. No node weights,
redshift search, physical formula or defaults are clipped or changed.

Six comparisons to independent definite integrals and the actual failed
64-point calculation fail before and pass after the repair. All 45 migration
tests pass, including the unchanged independent full-catalog tolerance.
This extends stable evaluation to a removable singularity, not a new EPS
prescription. The broader grid/solver convergence results are recorded separately.
