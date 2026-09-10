# Canonical velocities in shared states

The initializer and evolver now return km/s for every `v_max_*` state array,
alongside Msun, Mpc and Msun/Mpc^3. Previously, the final named catalog was
canonical but the intermediate state carried Mpc/s. This violated the shared
execution boundary and made those states unsafe to compose with other stages.

The SI-owned history context and gravothermal kernels keep their calibrated
floating units. The component explicitly converts the canonical initial
velocity to the kernel's Mpc/s, then converts evolved velocities back to km/s.
The catalog copies these arrays directly. Historical tuple conversion still
returns Mpc/s. Metadata records `stage_unit_contract=Msun-Mpc-km/s:v1`.

An independent NFW velocity-density-radius identity fails before this change
and passes after it at both real stage boundaries. All 38 migration tests pass,
including every array of the independent B reference at its unchanged 5e-12
tolerance, both states' weights, and canonical/historical format conversion.
This is a unit-boundary repair; the SI physical constants, histories, model
parameters, survival and public catalog units are unchanged.
