"""Promptmaker for Building SB-1.

This is a building-specific promptmaker used to generate prompts for controlling
Building 'SB-1'.
"""

from smart_buildings.smart_control.llm.prompts import floor_based_promptmaker as fbpm
from smart_buildings.smart_control.llm.prompts import promptmaker as pm


class SB1Promptmaker(pm.Promptmaker):
  """Promptmaker for Building 'SB-1'."""


class SB1FloorBasedPromptmaker(fbpm.FloorBasedPromptmaker):
  """Floor-based Promptmaker for Building 'SB-1'."""
