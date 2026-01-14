#     Copyright 2023 Google LLC
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
#     BUILD file for simulator package.
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

py_strict_test(
    name = "air_handler_test",
    srcs = ["air_handler_test.py"],
    deps = [
        ":air_handler",
        ":weather_controller",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

py_strict_test(
    name = "boiler_test",
    srcs = ["boiler_test.py"],
    deps = [
        ":boiler",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

py_strict_test(
    name = "building_test",
    srcs = ["building_test.py"],
    deps = [
        ":building",
        ":building_utils",
        ":constants",
        ":stochastic_convection_simulator",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "building_utils_test",
    srcs = ["building_utils_test.py"],
    deps = [
        ":building_utils",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "hvac_floorplan_based_test",
    srcs = ["hvac_floorplan_based_test.py"],
    deps = [
        ":air_handler",
        ":hot_water_system",
        ":hvac_floorplan_based",
        ":setpoint_schedule",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "hvac_test",
    srcs = ["hvac_test.py"],
    deps = [
        ":air_handler",
        ":hot_water_system",
        ":hvac",
        ":setpoint_schedule",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "randomized_arrival_departure_occupancy_test",
    srcs = ["randomized_arrival_departure_occupancy_test.py"],
    deps = [
        ":randomized_arrival_departure_occupancy",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "rejection_simulator_building_test",
    srcs = ["rejection_simulator_building_test.py"],
    deps = [
        ":rejection_simulator_building",
        ":simulator_building",
        ":simulator_building_test_lib",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

py_strict_test(
    name = "setpoint_schedule_test",
    srcs = ["setpoint_schedule_test.py"],
    deps = [
        ":setpoint_schedule",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/pytz",
    ],
)

py_strict_test(
    name = "simulator_building_test",
    srcs = ["simulator_building_test.py"],
    deps = [
        ":simulator_building",
        ":simulator_building_test_lib",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
    ],
)

py_strict_test(
    name = "simulator_flexible_floor_plan_test",
    srcs = ["simulator_flexible_floor_plan_test.py"],
    deps = [
        ":air_handler",
        ":building",
        ":hot_water_system",
        ":hvac_floorplan_based",
        ":setpoint_schedule",
        ":simulator_flexible_floor_plan",
        ":step_function_occupancy",
        ":weather_controller",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "simulator_test",
    srcs = ["simulator_test.py"],
    deps = [
        ":air_handler",
        ":building",
        ":hot_water_system",
        ":hvac",
        ":setpoint_schedule",
        ":simulator",
        ":step_function_occupancy",
        ":weather_controller",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "smart_device_test",
    srcs = ["smart_device_test.py"],
    deps = [
        ":smart_device",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

py_strict_test(
    name = "stochastic_occupancy_test",
    srcs = ["stochastic_occupancy_test.py"],
    deps = [
        ":stochastic_occupancy",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "step_function_occupancy_test",
    srcs = ["step_function_occupancy_test.py"],
    deps = [
        ":step_function_occupancy",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "thermal_diffuser_utils_test",
    srcs = ["thermal_diffuser_utils_test.py"],
    deps = [
        ":thermal_diffuser_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "thermostat_test",
    srcs = ["thermostat_test.py"],
    deps = [
        ":setpoint_schedule",
        ":thermostat",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "air_handler",
    srcs = ["air_handler.py"],
    deps = [
        ":smart_device",
        ":weather_controller",
        "//third_party/py/gin",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

pytype_strict_library(
    name = "boiler",
    srcs = ["boiler.py"],
    deps = [
        ":smart_device",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

pytype_strict_library(
    name = "building",
    srcs = ["building.py"],
    deps = [
        ":base_convection_simulator",
        ":building_utils",
        ":constants",
        ":thermal_diffuser_utils",
        "//third_party/py/gin",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "building_utils",
    srcs = ["building_utils.py"],
    deps = [
        ":constants",
        "//third_party/py/cv2",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_library(
    name = "hvac",
    srcs = ["hvac.py"],
    deps = [
        ":air_handler",
        ":hot_water_system",
        ":setpoint_schedule",
        ":thermostat",
        ":vav",
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "hvac_floorplan_based",
    srcs = ["hvac_floorplan_based.py"],
    deps = [
        ":air_handler",
        ":constants",
        ":hot_water_system",
        ":setpoint_schedule",
        ":thermostat",
        ":vav",
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "randomized_arrival_departure_occupancy",
    srcs = ["randomized_arrival_departure_occupancy.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "rejection_simulator_building",
    srcs = ["rejection_simulator_building.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_building",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

pytype_strict_library(
    name = "setpoint_schedule",
    srcs = ["setpoint_schedule.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/pytz",
    ],
)

pytype_strict_library(
    name = "simulator",
    srcs = ["simulator.py"],
    deps = [
        ":building",
        ":hvac",
        ":weather_controller",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "tf_simulator",
    srcs = ["tf_simulator.py"],
    deps = [
        ":building",
        ":hvac_floorplan_based",
        ":simulator_flexible_floor_plan",
        ":weather_controller",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "tf_simulator_test",
    srcs = ["tf_simulator_test.py"],
    deps = [
        "tf_simulator",
        ":air_handler",
        ":building",
        ":hot_water_system",
        ":hvac_floorplan_based",
        ":setpoint_schedule",
        ":weather_controller",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "simulator_building",
    srcs = ["simulator_building.py"],
    deps = [
        ":simulator",
        ":simulator_flexible_floor_plan",
        ":smart_device",
        ":tf_simulator",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_building",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "simulator_building_test_lib",
    srcs = ["simulator_building_test_lib.py"],
    deps = [
        ":air_handler",
        ":building",
        ":hot_water_system",
        ":hvac",
        ":setpoint_schedule",
        ":simulator",
        ":simulator_building",
        ":step_function_occupancy",
        ":weather_controller",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

pytype_strict_library(
    name = "simulator_flexible_floor_plan",
    srcs = ["simulator_flexible_floor_plan.py"],
    deps = [
        ":building",
        ":constants",
        ":hvac_floorplan_based",
        ":simulator",
        ":weather_controller",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:building_renderer",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
        "//third_party/py/smart_buildings/smart_control/utils:visual_logger",
    ],
)

pytype_strict_library(
    name = "smart_device",
    srcs = ["smart_device.py"],
    deps = [
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

pytype_strict_library(
    name = "stochastic_occupancy",
    srcs = ["stochastic_occupancy.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "step_function_occupancy",
    srcs = ["step_function_occupancy.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "thermal_diffuser_utils",
    srcs = ["thermal_diffuser_utils.py"],
    deps = [
        ":building_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/cv2",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "thermostat",
    srcs = ["thermostat.py"],
    deps = [
        ":setpoint_schedule",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "vav",
    srcs = ["vav.py"],
    deps = [
        ":air_handler",
        ":hot_water_system",
        ":smart_device",
        ":thermostat",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

pytype_strict_library(
    name = "weather_controller",
    srcs = ["weather_controller.py"],
    data = [
        # load "local_weather_moffett_field_20230701_20231122.csv" for replay weather controller:
        "//third_party/py/smart_buildings/smart_control/configs/resources/sb1:config_data",
    ],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "stochastic_convection_simulator",
    srcs = ["stochastic_convection_simulator.py"],
    deps = [
        ":base_convection_simulator",
        "//third_party/py/gin",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "base_convection_simulator",
    srcs = ["base_convection_simulator.py"],
    deps = ["//third_party/py/numpy"],
)

pytype_strict_library(
    name = "pump",
    srcs = ["pump.py"],
    deps = [
        ":smart_device",
        "//third_party/py/gin",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

pytype_strict_library(
    name = "hot_water_system",
    srcs = ["hot_water_system.py"],
    deps = [
        ":boiler",
        ":pump",
        ":smart_device",
        "//third_party/py/gin",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

py_strict_test(
    name = "hot_water_system_test",
    srcs = ["hot_water_system_test.py"],
    deps = [
        ":boiler",
        ":hot_water_system",
        ":pump",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

py_strict_test(
    name = "pump_test",
    srcs = ["pump_test.py"],
    deps = [
        ":pump",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

py_strict_test(
    name = "vav_test",
    srcs = ["vav_test.py"],
    deps = [
        ":air_handler",
        ":hot_water_system",
        ":setpoint_schedule",
        ":thermostat",
        ":vav",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

py_strict_test(
    name = "weather_controller_test",
    srcs = ["weather_controller_test.py"],
    data = ["local_weather_test_data.csv"],
    deps = [
        ":weather_controller",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/pandas",
    ],
)
