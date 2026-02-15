---
layout: "default"
title: "Occupancy Models"
nav_order: 4
parent: "Simulation Components"
grand_parent: "Smart Control Project Documentation"
---

# Occupancy Models

**Purpose**: Simulates occupancy patterns within the building, affecting internal heat gains and productivity.

## Key Classes and Components

- **`RandomizedArrivalDepartureOccupancy`**:

  - Models occupancy with randomized arrival and departure times.

  - **Attributes**:

    - `zone_assignment`: Occupancy level assigned to zones.
    - `earliest_expected_arrival_hour`: Earliest possible arrival time.
    - `latest_expected_arrival_hour`: Latest possible arrival time.
    - `earliest_expected_departure_hour`: Earliest possible departure time.
    - `latest_expected_departure_hour`: Latest possible departure time.
    - `time_step_sec`: Time step in seconds.
    - `time_zone`: Time zone for the simulation.

---

[Back to Simulation Components](simulation-components.md)

[Back to Home](../index.md)
