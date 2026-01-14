# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "config_data",
    srcs = [
        "demo_temps.npy",
        "double_resolution_zone_1_2.npy",
        "local_weather_moffett_field_20230701_20231122.csv",  # to do: move into weather data dir
        "observation_response_2024.08.29.00",
        "reset_temps.npy",
        "sim_202512/full_config.gin",
        "sim_202512/observations_config.gin",
        "sim_config.gin",
        "sim_config_legacy.gin",
        "weather_data/2023.csv",
        "weather_data/2024.csv",
        "weather_data/2025.csv",
        "weather_data/station.json",
    ],
)
