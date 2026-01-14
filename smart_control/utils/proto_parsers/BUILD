load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "observation_response_parser",
    srcs = ["observation_response_parser.py"],
    deps = [
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_building_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "observation_response_parser_test",
    srcs = ["observation_response_parser_test.py"],
    deps = [
        ":observation_response_parser",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/utils:test_utils",
    ],
)

pytype_strict_library(
    name = "reward_info_parser",
    srcs = ["reward_info_parser.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
    ],
)

py_strict_test(
    name = "reward_info_parser_test",
    srcs = ["reward_info_parser_test.py"],
    deps = [
        ":reward_info_parser",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/proto:smart_control_reward_py_pb2",
        "//third_party/py/smart_buildings/smart_control/utils:conversion_utils",
        "//third_party/py/smart_buildings/smart_control/utils:test_utils",
    ],
)
