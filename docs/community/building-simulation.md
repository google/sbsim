---
layout: "default"
title: "Building Simulation"
nav_order: 1
parent: "Simulation Components"
grand_parent: "Smart Control Project Documentation"
---

# Building Simulation

**Purpose**: Represents the thermal and physical properties of the building, simulating how it responds to HVAC inputs and environmental conditions.

## Key Classes and Components

- **`FloorPlanBasedBuilding`**:

  - Simulates the building's structure based on a floor plan and zone mappings.

  - **Attributes**:

    - `cv_size_cm`: Control volume size in centimeters.
    - `floor_height_cm`: Height of each floor.
    - `initial_temp`: Initial uniform interior temperature.
    - `inside_air_properties`: Thermal properties of the air inside the building.
    - `inside_wall_properties`: Thermal properties of the interior walls.
    - `building_exterior_properties`: Thermal properties of the exterior building.
    - `floor_plan_filepath`: Path to the floor plan file.
    - `zone_map_filepath`: Path to the zone mapping file.
    - `convection_simulator`: Simulates heat convection within the building.
    - `reset_temp_values`: Function to reset temperature values.

---

[Back to Simulation Components](simulation-components.md)

[Back to Home](../index.md)
