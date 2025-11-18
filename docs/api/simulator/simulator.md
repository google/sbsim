# Simulator

::: smart_control.simulator.simulator.Simulator
    options:
      # Ensure single-underscore private members are included.
      filters:
        # excludes dunder methods:
        - "!^__"
        # includes everything else:
        - ".*"
    members:
      # Use "*" to include all standard public methods/attributes
      - "*"
      # Explicitly list the specific private method you want to show
      - _get_interior_cv_temp_estimate


::: smart_control.simulator.simulator_flexible_floor_plan

::: smart_control.simulator.base_convection_simulator

::: smart_control.simulator.stochastic_convection_simulator

::: smart_control.simulator.tf_simulator
