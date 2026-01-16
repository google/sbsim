"""Occupancy models for building simulation.

This module contains various occupancy models that simulate the presence and
behavior of people in building zones. These models are used to calculate
average occupancy for different time intervals, which is an important input
for building simulation and control.

Available Models:
    - EnhancedOccupancy: Enhanced stochastic model with minute-level control
      and different worker types (weekday-only, weekend-regular, etc.)
    - LIGHTSWITCHOccupancy: Stochastic model based on the LIGHTSWITCH algorithm
      with arrival, departure, and lunch break patterns
    - RandomizedArrivalDepartureOccupancy: Model with randomized arrival and
      departure times within specified windows
    - StepFunctionOccupancy: Simple model with constant occupancy levels for
      work and non-work periods
"""

from smart_control.simulator.occupancy.enhanced_occupancy import EnhancedOccupancy
from smart_control.simulator.occupancy.enhanced_occupancy import MinuteLevelZoneOccupant
from smart_control.simulator.occupancy.enhanced_occupancy import WorkerType
from smart_control.simulator.occupancy.randomized_arrival_departure_occupancy import RandomizedArrivalDepartureOccupancy
from smart_control.simulator.occupancy.step_function_occupancy import StepFunctionOccupancy
from smart_control.simulator.occupancy.stochastic_occupancy import LIGHTSWITCHOccupancy
from smart_control.simulator.occupancy.stochastic_occupancy import OccupancyStateEnum
from smart_control.simulator.occupancy.stochastic_occupancy import ZoneOccupant

__all__ = [
    "EnhancedOccupancy",
    "MinuteLevelZoneOccupant",
    "WorkerType",
    "LIGHTSWITCHOccupancy",
    "ZoneOccupant",
    "RandomizedArrivalDepartureOccupancy",
    "StepFunctionOccupancy",
    "OccupancyStateEnum",
]
