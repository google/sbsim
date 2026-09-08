"""Promptmaker class with floor-specific zone comfort info."""

import functools

import pandas as pd
from smart_buildings.smart_control.llm.prompts import promptmaker as pm


class FloorBasedPromptmaker(pm.Promptmaker):
  """Updated promptmaker class, with floor-specific zone comfort info."""

  @functools.cached_property
  def zone_conditions_histogram_by_floor(self) -> pd.DataFrame:
    """A histogram of zone conditions by floor."""
    return self.reward_info_parser.get_zone_conditions_histogram_by_floor(
        zones=self.env.building.zones
    ).T

  @property
  def zone_conditions_subsection(self) -> str:
    """A section describing the current conditions in the building."""

    return self.dedent(f"""
      ### Current Zone Temperatures

      The table below conveys the comfort conditions across all zones in the building, by floor:

      {self.zone_conditions_histogram_by_floor.to_markdown(index=True)}

      The row 'occupancy_count' shows the total number of occupants building-wide at a specific temperature.
      The row 'setpoint_mask' indicates with a '0' if the temperature is within comfort range, a '-1' if the temperature is too cold, and a '1' if the temperature is too hot.
      The row 'setpoint_range' indicates with '+' if the temperature is inside the acceptable range, and '-' if it is outside.
      The row 'exposed_count' indicates the count of occupants being exposed to unacceptable comfort conditions.
      The rows starting with 'occ@floor' show the normalized distribution of zone counts for each floor at that temperature.
    """)
