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
#     BUILD file for simulator/occupancy package.
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

# Occupancy Model Libraries

pytype_strict_library(
    name = "enhanced_occupancy",
    srcs = ["enhanced_occupancy.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
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
    name = "step_function_occupancy",
    srcs = ["step_function_occupancy.py"],
    deps = [
        "//third_party/py/gin",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/models:base_occupancy",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

# Occupancy Model Tests

py_strict_test(
    name = "enhanced_occupancy_test",
    srcs = ["enhanced_occupancy_test.py"],
    deps = [
        ":enhanced_occupancy",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "stochastic_occupancy_test",
    srcs = ["stochastic_occupancy_test.py"],
    deps = [
        ":stochastic_occupancy",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "randomized_arrival_departure_occupancy_test",
    srcs = ["randomized_arrival_departure_occupancy_test.py"],
    deps = [
        ":randomized_arrival_departure_occupancy",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/pandas",
    ],
)
