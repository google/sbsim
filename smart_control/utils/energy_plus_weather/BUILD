load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "downloader",
    srcs = [
        "__init__.py",
        "__main__.py",
        "downloader.py",
    ],
    deps = [
        "//third_party/py/pandas",
        "//third_party/py/pvlib",
    ],
)

py_strict_test(
    name = "downloader_test",
    srcs = ["downloader_test.py"],
    data = ["test_data/USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3.zip"],
    deps = [":downloader"],
)
