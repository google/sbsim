"""Defines constants for use in simulation code suite.

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Here we use a specific placeholder value that helps us pick out interior walls
# and will not be used by connectedComponents() function (which only counts
# upwards positively) or the FileInputFloorPlan, which has 0, 1, and 2.
INTERIOR_WALL_VALUE_IN_FUNCTION = -3

# Here we use a specific placeholder value that helps us pick out interior walls
# AFTER connectedComponents labels them.
INTERIOR_WALL_VALUE_IN_COMPONENT = 0

# Here we use a specific placeholder value that helps us pick out exterior walls
# and will not be used by connectedComponents() function (which only counts
# upwards positively) or the FileInputFloorPlan, which has 0, 1, and 2.
EXTERIOR_WALL_VALUE_IN_FUNCTION = -2

# Here we use a specific placeholder value, matching with the file input schema,
# that designates exterior space in the file input.
EXTERIOR_SPACE_VALUE_IN_FILE_INPUT = 2

# Here we designate a specific placeholder to help use demarcate which CVs
# are for exterior space once processed in the function. It is intentionally
# set to -1 so that the connectedComponent function can have access to all
# nonzero integers to count upwards in an unbounded way.
EXTERIOR_SPACE_VALUE_IN_FUNCTION = -1

# Here we designate a specific placeholder to help use demarcate which CVs
# are for exterior space are noted in the component. It is intentionally
# set to -1 so that the connectedComponent function can have access to all
# nonzero integers to count upwards in an unbounded way.
EXTERIOR_SPACE_VALUE_IN_COMPONENT = 0

# Here we use a specific placeholder value, matching with the file input schema,
# that designates interior space in the file input.
INTERIOR_SPACE_VALUE_IN_FILE_INPUT = 0

# Here we pick out a specific value that we know will code for interior space
# after connectedComponents() processes it. We know this because we have ensured
# that the CV at index (0,0) will always be an "space" CV when ready for
# input to connectedComponents, but we previously index the exterior space CVs
# in their own data array. Thus, after overwriting the exterior space CVs to
# the value _EXTERIOR_SPACE_VALUE_IN_FUNCTION, all connectedComponents of
# value 0 code for interior space, and so we set
# _INTERIOR_SPACE_VALUE_IN_CONNECTION_INPUT equal to 0.
INTERIOR_SPACE_VALUE_IN_FUNCTION = 0

# connectedComponents function operates by accepting a matrix in which
# components, defined as 1's, are surrounded by 0's. This schema is
# opposite that which we were fed via the floorplan file input.
# Thus, we pick out 1's here to help process a ConnectionReadyFloorPlan.
INTERIOR_SPACE_VALUE_IN_CONNECTION_INPUT = 1

# Here we are deciding by how many control volumes in to consider the reach
# of exterior walls. I.e. if _EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT is 2, then
# we will consider two layers of walls to be exterior walls.
EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT = 2

# Here we wish to specifically set exterior space as indistinguishable
# from exterior walls, as we wish to perform connectedComponents only on
# connected groups of interior space. Thus, we set exterior space to a generic
# space value, i.e. 0.
GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT = 0

# Here we create a boolean count of the intersection of control volumes marked
# both as walls and as exterior walls enlarged, to make sure that we are
# returning expanded exterior walls when calling enlarge_component()
WALLS_AND_EXPANDED_BOOLS = 2

# Here we wish to specifically set exterior space as indistinguishable
# from exterior walls, as we wish to perform connectedComponents only on
# connected groups of interior space. Thus, we set exterior space to a generic
# space value, i.e. 0.
GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT = 0

# Here we use a specific placeholder value, matching with the file input schema,
# that designates interior space in the file input.
INTERIOR_WALL_VALUE_IN_FILE_INPUT = 1

# Here we designate a specific placeholder to help use demarcate which CVs
# are for interior walls once processed in the function. It is intentionally
# set to -3 so that the connectedComponent function can have access to all
# nonzero integers to count upwards in an unbounded way.
INTERIOR_WALL_VALUE_IN_FUNCTION = -3

# Here we set a specific string for exterior space to be labelled as in
# constructing a room dictionary.
EXTERIOR_SPACE_NAME_IN_ROOM_DICT = "exterior_space"

# Here we set a specific string for interior_walls to be labelled as in
# constructing a room dictionary.
INTERIOR_WALL_NAME_IN_ROOM_DICT = "interior_wall"

# The following constants are defined to identify CVs by type for use in the
# sweep function. Here is the label for all wall types.
LABEL_FOR_WALLS = "wall"

# The following constants are defined to identify CVs by type for use in the
# sweep function. Here is the label for all interior space types.
LABEL_FOR_INTERIOR_SPACE = "interior_space"

# The following constants are defined to identify CVs by type for use in the
# sweep function. Here is the label for all exterior space types.
LABEL_FOR_EXTERIOR_SPACE = "exterior_space"

# Here we list the string that all rooms should start with in the room_dict,
# by design of the construct_room_dict() function.
ROOM_STRING_DESIGNATOR = "room"

# Path to save videos generated by the simulation's visual logger.
VIDEO_PATH_ROOT = "/cns/oz-d/home/smart-buildings-control-team/smart-buildings/geometric_sim_videos/"  # pylint: disable=line-too-long

# The limit above which we do not want thermal diffusers to be dispensing energy
WATT_LIMIT = 500
