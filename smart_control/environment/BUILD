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

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "environment",
    srcs = ["environment.py"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/bidict",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_building",
        "//third_party/py/smart_buildings/smart_control/models:base_normalizer",
        "//third_party/py/smart_buildings/smart_control/models:base_reward_function",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:building_image_generator",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
        "//third_party/py/smart_buildings/smart_control/utils:histogram_reducer",
        "//third_party/py/smart_buildings/smart_control/utils:plot_utils",
        "//third_party/py/smart_buildings/smart_control/utils:regression_building_utils",
        "//third_party/py/smart_buildings/smart_control/utils:run_command_predictor",
        "//third_party/py/smart_buildings/smart_control/utils:writer_lib",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents/environments:py_environment",
        "//third_party/py/tf_agents/specs:array_spec",
        "//third_party/py/tf_agents/trajectories:time_step",
        "//third_party/py/tf_agents/typing:types",
    ],
)

pytype_strict_library(
    name = "environment_test_utils",
    srcs = ["environment_test_utils.py"],
    deps = [
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_building",
        "//third_party/py/smart_buildings/smart_control/models:base_reward_function",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "environment_test",
    srcs = ["environment_test.py"],
    deps = [
        ":environment",
        ":environment_test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/bidict",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_building",
        "//third_party/py/smart_buildings/smart_control/models:base_reward_function",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_normalization_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:bounded_action_normalizer",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
        "//third_party/py/smart_buildings/smart_control/utils:histogram_reducer",
        "//third_party/py/smart_buildings/smart_control/utils:observation_normalizer",
        "//third_party/py/smart_buildings/smart_control/utils:test_utils",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tf_agents/environments:utils",
        "//third_party/py/tf_agents/specs:array_spec",
        "//third_party/py/tf_agents/trajectories:time_step",
    ],
)
