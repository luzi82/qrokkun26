# Sv24 — Global-Scale Physical Spawner Policy

Sv24 resolves the three structural regressions identified in the Sv23 (`u=750`) analysis while retaining Sv23's unbiased physical action space `(birth_bearing, physical_speed, fire_bearing)` and exact conditional likelihoods:

1. **State-Independent Bounded Global Scales (`scale_logits[3]`)**:
   - Replaces Sv23's state-dependent scale head (`actor[-1][5:8]`) with a state-independent global parameter vector mapped via sigmoid into:
     - `birth_std` $\in [0.05, \pi]$ rad (init `2.8 rad`)
     - `speed_std` $\in [2.0, 80.0]$ px/s (init `42.0 px/s`)
     - `fire_std` $\in [0.01, 0.25]$ rad (init `0.04 rad` $\approx 2.3^\circ$, max $\approx 14.3^\circ$)
   - Prevents state-conditional variance explosion under sparse Monte Carlo returns (which drove Sv23's `fire_std` to $0.71\text{ rad}$ mean / $3.14\text{ rad}$ max and degraded validation survival to `99.41s`).

2. **Properly Scaled Pixel Lead Offset (`AIM_MAX_PX = 48.0`)**:
   - Maps actor outputs `means[..., 3:5]` via `48.0 * tanh(raw)` pixels before computing the conditional `spawn -> target` firing bearing, matching O(1) network pre-activations to physical lead distances (`~0.23 rad` bearing shift per unit output instead of Sv23's `0.006 rad`).

3. **Soft-Damped Bearing Derivative Near Origin (`1 / hypot(r, 0.25)`)**:
   - Keeps forward `atan2(y, x)` exact for all nonzero vectors while bounding the backward Jacobian to $\le 4.0$ when `||birth_vec|| -> 0`, eliminating `1/r` gradient spikes that previously triggered global gradient clipping (`0.5`) and starved speed/aim updates.
