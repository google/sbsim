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
#      Top-level BUILD for Smart Buildings Smart Control

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_library(
    name = "conversion_utils",
    srcs = ["conversion_utils.py"],
    deps = [
        "//google/protobuf:timestamp_py_pb2",
        "//third_party/py/holidays",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

py_strict_test(
    name = "agent_utils_test",
    srcs = ["agent_utils_test.py"],
    deps = [
        ":agent_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "bounded_action_normalizer_test",
    srcs = ["bounded_action_normalizer_test.py"],
    deps = [
        ":bounded_action_normalizer",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/tf_agents/specs",
    ],
)

py_strict_test(
    name = "energy_utils_test",
    srcs = ["energy_utils_test.py"],
    deps = [
        ":energy_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "histogram_reducer_test",
    srcs = ["histogram_reducer_test.py"],
    deps = [
        ":histogram_reducer",
        ":reader_lib",
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

py_strict_test(
    name = "observation_normalizer_test",
    srcs = ["observation_normalizer_test.py"],
    deps = [
        ":observation_normalizer",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
    ],
)

py_strict_test(
    name = "reducer_test",
    srcs = ["reducer_test.py"],
    deps = [
        ":reducer",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "regression_building_utils_test",
    srcs = ["regression_building_utils_test.py"],
    deps = [
        ":conversion_utils",
        ":regression_building_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/simulator:setpoint_schedule",
    ],
)

py_strict_test(
    name = "controller_read_write_test",
    srcs = ["controller_read_write_test.py"],
    deps = [
        ":controller_reader",
        ":controller_writer",
        ":reader_lib",
        ":writer_lib",
        "//net/proto2/contrib/pyutil:compare",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "run_command_predictor_test",
    srcs = ["run_command_predictor_test.py"],
    deps = [
        ":conversion_utils",
        ":reader_lib",
        ":run_command_predictor",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

py_test(
    name = "conversion_utils_test",
    srcs = ["conversion_utils_test.py"],
    deps = [
        ":conversion_utils",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

pytype_strict_library(
    name = "agent_utils",
    srcs = ["agent_utils.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "bounded_action_normalizer",
    srcs = ["bounded_action_normalizer.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/smart_buildings/smart_control/models:base_normalizer",
        "//third_party/py/tf_agents/specs",
    ],
)

pytype_strict_library(
    name = "building_image_generator",
    srcs = ["building_image_generator.py"],
    deps = [
        ":building_renderer",
        ":real_building_temperature_array_generator",
        "//third_party/py/PIL:pil",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/importlib_resources",
        "//third_party/py/numpy",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

pytype_strict_library(
    name = "building_renderer",
    srcs = ["building_renderer.py"],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/matplotlib",
        "//third_party/py/mediapy",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/seaborn",
        "//third_party/py/smart_buildings/smart_control/simulator:building_utils",
        "//third_party/py/smart_buildings/smart_control/simulator:constants",
    ],
)

pytype_strict_library(
    name = "energy_utils",
    srcs = ["energy_utils.py"],
    deps = [
        ":constants",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "controller_reader",
    srcs = ["controller_reader.py"],
    deps = [
        ":constants",
        ":reader_lib",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

pytype_strict_library(
    name = "controller_writer",
    srcs = ["controller_writer.py"],
    deps = [
        ":constants",
        ":writer_lib",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/google/protobuf",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

py_strict_test(
    name = "controller_writer_test",
    srcs = ["controller_writer_test.py"],
    deps = [
        ":controller_writer",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/pandas",
        "//third_party/py/plotly",
    ],
)

pytype_strict_library(
    name = "controller_writer_test_lib",
    testonly = True,
    srcs = ["controller_writer_test.py"],
    deps = [
        ":controller_writer",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/pandas",
        "//third_party/py/plotly",
    ],
)

pytype_strict_library(
    name = "histogram_reducer",
    srcs = ["histogram_reducer.py"],
    deps = [
        ":reader_lib",
        ":reducer",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

pytype_strict_library(
    name = "observation_normalizer",
    srcs = ["observation_normalizer.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/smart_buildings/smart_control/models:base_normalizer",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
    ],
)

pytype_strict_library(
    name = "plot_utils",
    srcs = ["plot_utils.py"],
    deps = [
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "reader_lib",
    srcs = ["reader_lib.py"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/importlib_resources",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

pytype_strict_library(
    name = "real_building_temperature_array_generator",
    srcs = ["real_building_temperature_array_generator.py"],
    deps = [
        ":conversion_utils",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

pytype_strict_library(
    name = "reducer",
    srcs = ["reducer.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "regression_building_utils",
    srcs = ["regression_building_utils.py"],
    deps = [
        ":conversion_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/simulator:setpoint_schedule",
    ],
)

pytype_strict_library(
    name = "run_command_predictor",
    srcs = ["run_command_predictor.py"],
    deps = [
        ":conversion_utils",
        ":reader_lib",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sklearn",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
    ],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        ":bounded_action_normalizer",
        ":conversion_utils",
        ":reader_lib",
        "//testing/pybase",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/environment",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/simulator:setpoint_schedule",
    ],
)

pytype_strict_library(
    name = "visual_logger",
    srcs = ["visual_logger.py"],
    deps = [
        ":building_renderer",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "writer_lib",
    srcs = ["writer_lib.py"],
    deps = [
        "//third_party/py/importlib_resources",
        "//third_party/py/pandas",
        "//third_party/py/plotly",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
    ],
)

pytype_strict_library(
    name = "environment_utils",
    srcs = ["environment_utils.py"],
    deps = [
        ":bounded_action_normalizer",
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
    ],
)
