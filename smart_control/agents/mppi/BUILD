#     Copyright 2025 Google LLC
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
    default_visibility = [
        "//corp/ml/smart_buildings/smart_control:packages",
    ],
)

pytype_strict_library(
    name = "mppi_utils",
    srcs = ["mppi_utils.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/smart_buildings/smart_control/utils:bounded_action_normalizer",
        "//third_party/py/tf_agents/trajectories:policy_step",
    ],
)

py_strict_test(
    name = "mppi_utils_test",
    srcs = ["mppi_utils_test.py"],
    deps = [
        ":mppi_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/mock",
        "//third_party/py/numpy",
        "//third_party/py/smart_buildings/smart_control/utils:bounded_action_normalizer",
        "//third_party/py/tf_agents/trajectories:policy_step",
    ],
)

pytype_strict_library(
    name = "world_model",
    srcs = ["world_model.py"],
    deps = [
        ":mppi_utils",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/environment",
        "//third_party/py/smart_buildings/smart_control/models:base_world_model",
        "//third_party/py/smart_buildings/smart_control/utils:bounded_action_normalizer",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents/environments:tf_py_environment",
        "//third_party/py/tf_agents/trajectories:policy_step",
    ],
)

pytype_strict_library(
    name = "base_forecaster",
    srcs = ["base_forecaster.py"],
    deps = ["//third_party/py/pandas"],
)

py_strict_test(
    name = "world_model_test",
    srcs = ["world_model_test.py"],
    deps = [
        ":world_model",
        "//testing/pybase",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/utils:bounded_action_normalizer",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents/specs:array_spec",
        "//third_party/py/tf_agents/trajectories:time_step",
    ],
)

pytype_strict_library(
    name = "mppi_agent",
    srcs = ["mppi_agent.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/smart_buildings/smart_control/models:base_world_model",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/tf_agents/agents:tf_agent",
        "//third_party/py/tf_agents/environments:tf_py_environment",
        "//third_party/py/tf_agents/policies:tf_policy",
        "//third_party/py/tf_agents/trajectories:policy_step",
        "//third_party/py/tf_agents/trajectories:time_step",
        "//third_party/py/tf_agents/typing:types",
    ],
)

py_strict_test(
    name = "mppi_agent_test",
    srcs = ["mppi_agent_test.py"],
    deps = [
        ":mppi_agent",
        "//testing/pybase",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/smart_buildings/smart_control/utils:bounded_action_normalizer",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tf_agents/specs:array_spec",
        "//third_party/py/tf_agents/trajectories:time_step",
    ],
)
