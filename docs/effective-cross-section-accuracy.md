# Stable evaluation of the existing effective cross section

The definition of the effective cross section is unchanged. With
`a = w^2/(4 (0.64 Vmax)^2)`, its ratio to sigma0/m is

`-a^2 [exp(a) (1+a) Ei(-a) + 1]`.

For a near 700 the bracket is tiny and subtracts nearly equal floating-point
numbers. Against a 65-digit independent evaluation, the frozen `58701d5`
implementation has up to 5.76e-11 relative error on its interpolation nodes.
The Python 3.12 CI density discrepancy triggered this investigation; reproducing
its dependency versions locally did not reproduce the exact platform error.
The independently measured cancellation error is sufficient to require a
numerically stable evaluation regardless of the platform.

Integration by parts and `u=a*t` give the identical positive integral

`integral_0^infinity u exp(-u)/(1+u/a)^2 du`.

The product evaluates it with 64 generalized Gauss-Laguerre nodes for
`20 <= a <= a_threshold`. Smaller a retains the direct expression. Above the
existing threshold (default 703), the existing degree-six asymptotic expansion
is retained. The interpolation nodes and public threshold/degree arguments are
unchanged; no interaction prescription or physical default changes.

A 51-point independent 65-digit comparison over a=20..703 gives maximum
relative error 4.45e-16. Orders 16, 32, 64 and 128 all agree at floating-point
precision. A new interpolation-node regression uses 5e-13 tolerance and fails
before this repair. A validation-only B patch evaluates the original expression
with mpmath at 65 digits, independently of the product quadrature. Existing B
files are preserved; new B-accurate files are separately identified and hashed.

The isolated effective-cross-section correction changes the small-catalog
collapse-time ratio by at most 4.37e-11, SIDM core radius by at most 1.87e-11,
and density by at most 3.43e-12. CDM columns, variance/growth probes, weights and
survival masks are bitwise unchanged. B/C comparisons keep their existing
5e-12 tolerance. The calculation specification advances to v2 and records the
numerical prescription.

The separate question about `sigma_total` concerns a different formula and is
not changed by this numerical evaluation repair. It was subsequently adopted
and corrected in the separate [total-cross-section unit](total-cross-section.md).
