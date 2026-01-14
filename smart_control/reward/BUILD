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
#     Defines implementations of rewards for Smart Buildings.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "setpoint_energy_carbon_regret",
    srcs = ["setpoint_energy_carbon_regret.py"],
    deps = [
        ":base_setpoint_energy_carbon_reward",
        "//third_party/py/gin",
        "//third_party/py/smart_buildings/smart_control/models:base_energy_cost",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "setpoint_energy_carbon_regret_test",
    srcs = ["setpoint_energy_carbon_regret_test.py"],
    deps = [
        ":setpoint_energy_carbon_regret",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_energy_cost",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "base_setpoint_energy_carbon_reward",
    srcs = ["base_setpoint_energy_carbon_reward.py"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_energy_cost",
        "//third_party/py/smart_buildings/smart_control/models:base_reward_function",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "base_setpoint_energy_carbon_reward_test",
    srcs = ["base_setpoint_energy_carbon_reward_test.py"],
    deps = [
        ":base_setpoint_energy_carbon_reward",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_energy_cost",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "setpoint_energy_carbon_reward",
    srcs = ["setpoint_energy_carbon_reward.py"],
    deps = [
        ":base_setpoint_energy_carbon_reward",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_energy_cost",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "setpoint_energy_carbon_reward_test",
    srcs = ["setpoint_energy_carbon_reward_test.py"],
    deps = [
        ":setpoint_energy_carbon_reward",
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_energy_cost",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

pytype_strict_library(
    name = "natural_gas_energy_cost",
    srcs = ["natural_gas_energy_cost.py"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_energy_cost",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

py_strict_test(
    name = "natural_gas_energy_cost_test",
    srcs = ["natural_gas_energy_cost_test.py"],
    deps = [
        ":natural_gas_energy_cost",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
    ],
)

pytype_strict_library(
    name = "electricity_energy_cost",
    srcs = ["electricity_energy_cost.py"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pint",
        "//third_party/py/smart_buildings/smart_control/models:base_energy_cost",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "electricity_energy_cost_test",
    srcs = ["electricity_energy_cost_test.py"],
    deps = [
        ":electricity_energy_cost",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/pandas",
    ],
)
