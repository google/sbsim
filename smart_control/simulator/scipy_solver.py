r"""Sparse direct solver for the finite volume heat balance (SciPy).

`ScipySolverSimulator` solves the same discrete heat balance as
`simulator.Simulator` (one CV at a time) and `tf_simulator.TFSimulator` (the
whole grid at once with shifted tensors), but instead of relaxing it, it
assembles the balance of every CV into one sparse linear system and solves it
directly. It is meant as a reference: with the outer iteration converged, its
answer is the exact solution of the discrete equations, so the two relaxation
solvers can be checked against it and their convergence thresholds can be
judged by how far they stop from it.

Heat balance of one CV
----------------------
For a CV $P$ with interior faces $I_P$ and faces open to the ambient $B_P$,
backward Euler gives

$$\sum_{f \in I_P} A_f \Gamma_f (T_{N_f} - T_P)
  + \sum_{f \in B_P} A_f \Gamma_{\infty,f} (T_\infty - T_P) + Q_P
  = C_P \left(T_P - T_P^{(-)}\right), \qquad
  C_P = \frac{\rho c\, u v z}{\Delta t},$$

with the series half-cell conductances of `Simulator._get_face_conductance`
and `Simulator._get_ambient_conductance`,

$$\Gamma_f = \left(\frac{\ell_{P,f}}{2k_P} +
    \frac{\ell_{N_f,f}}{2k_{N_f}}\right)^{-1}, \qquad
  \Gamma_{\infty,f} = \left(\frac{1}{h} + \frac{\ell_{P,f}}{2k_P}\right)^{-1}.$$

Point relaxation vs. direct solve
---------------------------------
The relaxation solvers (FDM sweep) isolate $T_P$ and update one CV from its
neighbors' current estimates, repeating until nothing moves:

$$T_P \leftarrow \frac{\sum_{f \in I_P} A_f \Gamma_f T_{N_f}
  + \sum_{f \in B_P} A_f \Gamma_{\infty,f} T_\infty + Q_P + C_P T_P^{(-)}}
  {C_P + \sum_{f \in I_P} A_f \Gamma_f + \sum_{f \in B_P} A_f
  \Gamma_{\infty,f}}.$$

The same equation, with the unknowns $T_P$ and $T_{N_f}$ moved to the left,
is one row of a linear system $\mathbf{A}\mathbf{T} = \mathbf{b}$:

$$\Big(C_P + \sum_{f \in I_P} G_{P,N_f} + \sum_{f \in B_P}
  G_{P,\infty,f}\Big) T_P - \sum_{f \in I_P} G_{P,N_f} T_{N_f}
  = C_P T_P^{(-)} + Q_P + \sum_{f \in B_P} G_{P,\infty,f} T_\infty,$$

where $G_{P,N_f} = A_f \Gamma_f$ and $G_{P,\infty,f} = A_f \Gamma_{\infty,f}$
[W/K]. So

$$A_{P,P} = C_P + \sum_{f \in I_P} G_{P,N_f} + \sum_{f \in B_P}
  G_{P,\infty,f}, \qquad A_{P,N_f} = -G_{P,N_f}, \qquad
  b_P = C_P T_P^{(-)} + Q_P + \sum_{f \in B_P} G_{P,\infty,f} T_\infty.$$

For example, three CVs in a row with the ambient on the left of CV 1 and on
the right of CV 3:

```text
  T_amb | T1 | T2 | T3 | T_amb

  [ D1        -A12G12      0      ] [T1]   [b1]
  [ -A12G12    D2       -A23G23   ] [T2] = [b2]
  [ 0         -A23G23      D3     ] [T3]   [b3]

  D1 = C1 + A12G12 + A1G1inf        b1 = C1 T1^(-) + Q1 + A1G1inf T_amb
  D2 = C2 + A12G12 + A23G23         b2 = C2 T2^(-) + Q2
  D3 = C3 + A23G23 + A3G3inf        b3 = C3 T3^(-) + Q3 + A3G3inf T_amb
```

A face conductance is the same seen from either CV (both the series
resistance and the shared area $\min(w_P, w_{N_f})\,z$ are symmetric in $P$
and $N_f$), so the off-diagonal part of $\mathbf{A}$ is symmetric, and the
diagonal dominates it by $C_P$ plus any ambient conductance. Exterior CVs are
pinned at $T_\infty$ with a unit row; their coupling to a boundary CV moves to
$b_P$.

Without radiation or interior mass this system is linear and one direct solve
is the exact answer of the time step. The relaxation solvers converge to the
same answer, only through many sweeps.

Radiation and interior mass: outer Picard iteration
---------------------------------------------------
Interior longwave exchange depends on $T^4$, so the system is no longer
linear, and the interior mass temperature is a second unknown. Both are
handled by the outer fixed-point (Picard) loop inherited from
`Simulator.finite_differences_timestep`; this class only replaces the inner
air solve. In outer iteration $k$:

1. Evaluate the interior longwave flux $q_{\text{lwx}}$ [W/m^2] from the
   current estimate $T^{(k-1)}$ and add $q_{\text{lwx}}\,\delta_x z$ [W] to
   $b_P$ of the CVs that carry a radiating surface.
2. Add the mass coupling $K_m = k_{\text{mass}} u v / z$ to $A_{P,P}$ and
   $K_m T_{\text{mass}}^{(k-1)}$ to $b_P$.
3. Solve $\mathbf{A}\mathbf{T}^{(k)} = \mathbf{b}$ directly.
4. Update the mass from the new air temperature, always starting from the
   previous time step's mass temperature (`update_interior_mass_temperatures`):
   $T_{\text{mass}}^{(k)} = (T^{(k)} + t_0 T_{\text{mass}}^{(-)}) / (1 + t_0)$.
5. Repeat until neither field moves by more than the convergence threshold.

At the fixed point the radiative term and the mass coupling are evaluated at
the converged field, so the result solves the full nonlinear, coupled balance
of the time step. The difference from the relaxation solvers is only that each
outer iteration returns the exact solution of the current linearized system
rather than one relaxation sweep toward it, so far fewer outer iterations are
needed.

The radiative exchange could also be linearized, as
$\sigma(T_j^4 - T_i^4) = h_{ij}(T_j - T_i)$ with
$h_{ij} = \sigma(T_i^2 + T_j^2)(T_i + T_j)$ evaluated at the previous iterate,
and the mass node could be folded into the same matrix. Neither is done here,
so that all three solvers share the same treatment of those terms.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from smart_control.simulator import simulator_flexible_floor_plan


class ScipySolverSimulator(
    simulator_flexible_floor_plan.SimulatorFlexibleGeometries
):
  r"""Solves each linearized heat balance with a SciPy sparse direct solve.

  Overrides only `update_temperature_estimates`: instead of one relaxation
  sweep, each call assembles $\mathbf{A}\mathbf{T} = \mathbf{b}$ for all CVs
  (see the module docstring) and solves it with `scipy.sparse.linalg.spsolve`.
  The outer Picard loop for radiation and interior mass is the one inherited
  from `Simulator.finite_differences_timestep`. Without radiation and interior
  mass, the first call is already the exact solution of the step and the
  second one only confirms convergence.
  """

  def update_temperature_estimates(
      self,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> tuple[np.ndarray, float]:
    """Assembles and directly solves the current linearized heat balance.

    Args:
      temperature_estimates: Current temperature estimate for each CV.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).

    Returns:
      A pair: the solved temperatures, and the largest change any CV saw
      relative to `temperature_estimates`.
    """
    building = self.building
    rows_count, cols_count = building.temp.shape
    cv_count = rows_count * cols_count
    flat_index = np.arange(cv_count).reshape(rows_count, cols_count)
    exterior = np.fromiter(
        (
            len(building.neighbors[row][column]) <= 1
            for row in range(rows_count)
            for column in range(cols_count)
        ),
        dtype=bool,
        count=cv_count,
    ).reshape(rows_count, cols_count)

    q_lwx_array = None
    if getattr(building, 'include_radiative_heat_transfer', False):
      q_lwx_array = building.apply_longwave_interior_radiative_heat_transfer(
          temperature_estimates
      )

    matrix_rows = []
    matrix_columns = []
    matrix_values = []
    rhs = np.zeros(cv_count, dtype=np.float64)
    floor_height = building.floor_height_cm / 100.0
    delta_x = building.cv_size_cm / 100.0

    for row in range(rows_count):
      for column in range(cols_count):
        index = flat_index[row, column]
        if exterior[row, column]:
          # Exterior CVs are pinned at the ambient temperature.
          matrix_rows.append(index)
          matrix_columns.append(index)
          matrix_values.append(1.0)
          rhs[index] = ambient_temperature
          continue

        # Storage term C_P on the diagonal and C_P T_P^(-) + Q_P in b_P.
        u, v = self._get_cv_dimensions((row, column))
        storage = (
            building.density[row, column]
            * building.heat_capacity[row, column]
            * u
            * v
            * floor_height
            / self._time_step_sec
        )
        diagonal = storage
        rhs_value = (
            storage * building.temp[row, column] + building.input_q[row, column]
        )

        # Interior faces: G_{P,N} on the diagonal and -G_{P,N} off it. A face
        # onto a pinned exterior CV moves to b_P instead.
        for neighbor_row, neighbor_column in building.neighbors[row][column]:
          conductance = self._get_face_conductance(
              (row, column), (neighbor_row, neighbor_column)
          )
          diagonal += conductance
          if exterior[neighbor_row, neighbor_column]:
            rhs_value += conductance * ambient_temperature
          else:
            matrix_rows.append(index)
            matrix_columns.append(flat_index[neighbor_row, neighbor_column])
            matrix_values.append(-conductance)

        # Faces open to the ambient: film in series with the outer half cell.
        ambient_conductance = self._get_ambient_conductance(
            (row, column), convection_coefficient
        )
        diagonal += ambient_conductance
        rhs_value += ambient_conductance * ambient_temperature

        # Interior mass coupling, with the mass at the current outer iterate.
        if (
            getattr(building, 'include_interior_mass', False)
            and building.interior_mass_mask[row, column]
        ):
          mass_conductance = (
              building.interior_mass_conductivity[row, column]
              * u
              * v
              / floor_height
          )
          diagonal += mass_conductance
          rhs_value += (
              mass_conductance * building.interior_mass_temp[row, column]
          )

        if q_lwx_array is not None:
          radiation_index = building.lwx_index[row, column]
          if radiation_index >= 0:
            # The exchange is a flux in W/m^2 across a wall face of area
            # delta_x * z, as in the other two solvers.
            rhs_value += q_lwx_array[radiation_index] * delta_x * floor_height

        matrix_rows.append(index)
        matrix_columns.append(index)
        matrix_values.append(diagonal)
        rhs[index] = rhs_value

    system = sparse.csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(cv_count, cv_count),
    )
    solution = sparse_linalg.spsolve(system, rhs).reshape(
        rows_count, cols_count
    )
    max_delta = float(np.max(np.abs(solution - temperature_estimates)))
    return solution, max_delta
