"""Configuration and helpers for building radiation tests."""

import copy

import numpy as np

from smart_control.simulator import building

FLOOR_PLAN = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])


def create_building_with_radiative_properties(
    view_factor_method: str = "ScriptF",
    floor_plan: np.array = FLOOR_PLAN,
    initial_temp: float = 292.0,
    cv_size_cm: float = 20.0,
    floor_height_cm: float = 300.0,
    inside_air_radiative_properties: building.RadiationProperties = None,
    inside_wall_radiative_properties: building.RadiationProperties = None,
    building_exterior_radiative_properties: building.RadiationProperties = None,
):

  inside_air_properties = building.MaterialProperties(
      conductivity=50.0, heat_capacity=700.0, density=1.0
  )
  inside_wall_properties = building.MaterialProperties(
      conductivity=2.0, heat_capacity=500.0, density=1800.0
  )
  building_exterior_properties = building.MaterialProperties(
      conductivity=0.05, heat_capacity=500.0, density=3000.0
  )

  inside_air_radiative_properties = inside_air_radiative_properties or (
      building.DefaultInsideAirRadiationProperties()
  )
  inside_wall_radiative_properties = inside_wall_radiative_properties or (
      building.DefaultInsideWallRadiationProperties()
  )
  building_exterior_radiative_properties = (
      building_exterior_radiative_properties
      or building.DefaultExteriorWallRadiationProperties()
  )

  zone_map = copy.deepcopy(floor_plan)

  return building.FloorPlanBasedBuilding(
      cv_size_cm=cv_size_cm,
      floor_height_cm=floor_height_cm,
      initial_temp=initial_temp,
      inside_air_properties=inside_air_properties,
      inside_wall_properties=inside_wall_properties,
      building_exterior_properties=building_exterior_properties,
      floor_plan=floor_plan,
      zone_map=zone_map,
      buffer_from_walls=0,
      inside_air_radiative_properties=inside_air_radiative_properties,
      inside_wall_radiative_properties=inside_wall_radiative_properties,
      building_exterior_radiative_properties=building_exterior_radiative_properties,  # pylint: disable=line-too-long
      include_radiative_heat_transfer=True,
      view_factor_method=view_factor_method,
  )
