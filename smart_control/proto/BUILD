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

load("//third_party/bazel_rules/rules_python/python:proto.bzl", "py_proto_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

proto_library(
    name = "smart_control_reward_proto",
    srcs = ["smart_control_reward.proto"],
    deps = ["//google/protobuf:timestamp"],
)

py_proto_library(
    name = "smart_control_reward_py_pb2",
    api_version = 2,
    visibility = [
        "//corp/ml/smart_buildings/smart_control:packages",
        "//third_party/py/smart_buildings/smart_control/environment:__pkg__",
        "//third_party/py/smart_buildings/smart_control/models:__pkg__",
        "//third_party/py/smart_buildings/smart_control/reward:__pkg__",
        "//third_party/py/smart_buildings/smart_control/simulator:__pkg__",
        "//third_party/py/smart_buildings/smart_control/utils:__pkg__",
    ],
    deps = [":smart_control_reward_proto"],
)

proto_library(
    name = "smart_control_building_proto",
    srcs = ["smart_control_building.proto"],
    deps = ["//google/protobuf:timestamp"],
)

py_proto_library(
    name = "smart_control_building_py_pb2",
    api_version = 2,
    visibility = [
        "//corp/ml/smart_buildings/smart_control:packages",
        "//third_party/py/smart_buildings/smart_control/environment:__pkg__",
        "//third_party/py/smart_buildings/smart_control/models:__pkg__",
        "//third_party/py/smart_buildings/smart_control/simulator:__pkg__",
        "//third_party/py/smart_buildings/smart_control/utils:__pkg__",
    ],
    deps = [":smart_control_building_proto"],
)

proto_library(
    name = "smart_control_normalization_proto",
    srcs = ["smart_control_normalization.proto"],
    deps = ["//google/protobuf:timestamp"],
)

py_proto_library(
    name = "smart_control_normalization_py_pb2",
    api_version = 2,
    deps = [":smart_control_normalization_proto"],
)
