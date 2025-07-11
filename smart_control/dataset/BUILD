# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/utils:constants",
        "//third_party/py/requests",
    ],
)

pytype_strict_library(
    name = "partition",
    srcs = ["partition.py"],
    deps = [
        ":dataset",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "conftest",
    testonly = True,
    srcs = ["conftest.py"],
    deps = [
        ":dataset",
        ":partition",
        "//third_party/py/dotenv"
        "//third_party/py/pytest",
    ],
)

py_strict_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    deps = [
        ":dataset",
        ":conftest",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
    ],
)

py_strict_test(
    name = "partition_test",
    srcs = ["partition_test.py"],
    deps = [
        ":partition",
        ":conftest",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
    ],
)
