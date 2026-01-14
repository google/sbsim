load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "llm_service",
    srcs = ["llm_service.py"],
)

pytype_strict_library(
    name = "gemini_service",
    srcs = ["gemini_service.py"],
    deps = [
        ":llm_service",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/dotenv",
        "//third_party/py/google/genai",
    ],
)

pytype_strict_binary(
    name = "gemini_service_script",
    srcs = ["gemini_service.py"],
    main = "gemini_service.py",
    deps = [
        ":gemini_service",
    ],
)

pytype_strict_library(
    name = "vertex_service",
    srcs = ["vertex_service.py"],
    deps = [
        ":gemini_service",
        "//third_party/py/dotenv",
        "//third_party/py/google/auth:__init__",
        "//third_party/py/google/genai",
    ],
)

py_strict_test(
    name = "vertex_service_test",
    srcs = ["vertex_service_test.py"],
    deps = [
        ":conftest",
        ":vertex_service",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/auth:credentials",
        "//third_party/py/google/genai",
    ],
)

pytype_strict_library(
    name = "conftest",
    srcs = ["conftest.py"],
    deps = [
        ":gemini_service",
        ":llm_service",
        ":vertex_service",
        "//third_party/py/dotenv",
        "//third_party/py/google/auth:credentials",
        "//third_party/py/google/genai",
    ],
)

py_strict_test(
    name = "llm_service_test",
    srcs = ["llm_service_test.py"],
    deps = [
        ":conftest",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
    ],
)

py_strict_test(
    name = "gemini_service_test",
    srcs = ["gemini_service_test.py"],
    deps = [
        ":conftest",
        ":gemini_service",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/google/genai",
    ],
)
