load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

py_strict_test(
    name = "llm_environment_test",
    srcs = ["llm_environment_test.py"],
    deps = [
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/mock",
        "//third_party/py/pandas",
        "//third_party/py/smart_buildings/smart_control/environment",
        "//third_party/py/smart_buildings/smart_control/environment:conftest",
        "//third_party/py/smart_buildings/smart_control/environment:hybrid_action_environment",
        "//third_party/py/smart_buildings/smart_control/models:base_building",
        "//third_party/py/smart_buildings/smart_control/models:base_reward_function",
        "//third_party/py/smart_buildings/smart_control/utils:building_image_generator",
        "//third_party/py/smart_buildings/smart_control/utils:controller_writer",
        "//third_party/py/smart_buildings/smart_control/utils:observation_normalizer",
    ],
)
