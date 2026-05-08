"""Promptmaker for optimal control of HVAC systems in smart buildings.

This promptmaker extends the base promptmaker class to create a prompt for
controlling HVAC systems in smart buildings.

It uses the SetpointsAction pydantic model to provide formatting instructions
for the LLM response, to include a validity interval, overall strategy, a list
of setpoints and corresponding setpoint-specific justifications.

This promptmaker constructs a basic non-opinionated prompt that could be used
as a basis for more specialized child classes. Prompts are expected to be an
active area of experimentation, so this class is designed to support
extensibility.

The promptmaker uses a number of 'sections' that comprise the prompt. Each
section is a piece of the prompt that serves a specific purpose. By inheriting
from the promptmaker class, you can override specific sections to customize
the prompt without having to rewrite the entire prompt.

In terms of content formatting, we are using Markdown. Research suggests this
may help the LLM better understand the organizational structure of the content.
See 'Does Prompt Formatting Have Any Impact on LLM Performance?' by He, et al.

We are also using new-line characters to separate each sentence, keeping each
sentence fully contained on the same line. There is research to suggest that
new-line characters are effective delimiters for helping the LLM understand the
content (specifically examples) and generate a better response. See: 'A single
character can make or break your LLM evals' by Jingtong Su, et al.
"""

import dataclasses
from typing import Any, Callable, Final

import pandas as pd

from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment as hybrid_env
from smart_buildings.smart_control.llm.prompts import base_promptmaker
from smart_buildings.smart_control.llm.schema import output_schema
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2
from smart_buildings.smart_control.utils import temperature_conversion as tc
from smart_buildings.smart_control.utils.proto_parsers import observation_response_parser as or_parser
from smart_buildings.smart_control.utils.proto_parsers import reward_info_parser as ri_parser

SerializableData = dict[str, Any]

# TODO(mjrossetti): Consider importing these constants from other more central
# locations related to the devices, once they are available there.
AHU_STATIC_PRESSURE_SETPOINT: Final[str] = "static_pressure_setpoint"
AHU_SUPPLY_AIR_TEMPERATURE_SETPOINT: Final[str] = "supply_air_temperature_setpoint"  # pylint: disable=line-too-long
HWS_DIFFERENTIAL_PRESSURE_SETPOINT: Final[str] = "differential_pressure_setpoint"  # pylint: disable=line-too-long
HWS_SUPPLY_WATER_TEMPERATURE_SETPOINT: Final[str] = "supply_water_setpoint"


@dataclasses.dataclass
class BuildingInfo:
  """Information about the building under control.

  This information is provided to the LLM to give it context about the building.

  Attributes:
    stories: The number of stories in the building.
    sqft: The square footage of the building.
    location: The location of the building.
    name: The name of the building, if applicable.
  """
  name: str = "SB-1"
  stories: str = "two"
  sqft: int = 96_000
  location: str = "Mountain View, California"


class Promptmaker(base_promptmaker.BasePromptmaker):
  """Promptmaker for building control.

  This specific promptmaker assumes you are using a HybridActionEnvironment.
  """

  def __init__(
      self,
      env: environment.Environment,
      *,
      observation_response: building_pb2.ObservationResponse | None = None,
      reward_info: reward_pb2.RewardInfo | None = None,
      building_info: BuildingInfo | None = None,
      output_schema_class: (
          type[output_schema.SetpointsAction] | None
      ) = output_schema.SetpointsAction,
      dedent: Callable[[str], str] = base_promptmaker.full_dedent,
      include_weights: bool = False,
      occupancy_mode_min: int = 10,
      temp_display_unit: tc.TempUnit | str = tc.TempUnit.FAHRENHEIT,
      lazy_init_protos: bool = False,
  ):
    """Initializes the instance.

    Args:
      env: The environment containing information about the building,
        observation space, action space, reward function, etc.
      observation_response: The observation response from the environment. If
        None, the observation response will be retrieved from the environment.
      reward_info: The reward info from the environment. If None, the reward
        info will be retrieved from the environment.
      building_info: Information about the building being controlled, such as
        the number of stories, square footage, and location.
      output_schema_class: The pydantic model class used to provide JSON
        response formatting instructions in the prompt. Uses the pre-configured
        `SetpointsAction` model by default. To use custom validity interval
        options, construct a custom output schema class using the
        `output_schema.create_action_model` function, and pass that class here.
      dedent: The function used to remove leading whitespace from the prompt.
        Uses the `full_dedent` function by default, because otherwise the
        inserted tables seem to be aligned to the left of the rest of the
        content.
      include_weights: Whether to include the reward function weights in the
        prompt.
      occupancy_mode_min: The minimum number of occupants in the building to
        be considered in occupancy mode.
      temp_display_unit: The temperature unit to be used by the LLM in its
        justifications and reasoning. All input temperatures are in Kelvin.
      lazy_init_protos: Whether to lazily setup the observation
        response and reward info. If False, (by default), the protos
        should be passed in during initialization, or will automatically be set,
        for convenience. If True, the protos are expected to be passed in after
        initialization, using the `set_protos` method.
    """
    super().__init__(output_schema_class=output_schema_class, dedent=dedent)
    self.env = env
    self.include_weights = include_weights
    self.occupancy_mode_min = occupancy_mode_min
    self.temp_display_unit = tc.assign_temp_unit(temp_display_unit)
    self.building_info = building_info or BuildingInfo()
    self.lazy_init_protos = lazy_init_protos
    self._observation_response_parser: (
        or_parser.ObservationResponseParser | None
    ) = None
    self._reward_info_parser: ri_parser.RewardInfoParser | None = None

    if not self.lazy_init_protos:
      self.set_protos(
          observation_response=observation_response,
          reward_info=reward_info,
      )

  def set_protos(
      self,
      observation_response: building_pb2.ObservationResponse | None,
      reward_info: reward_pb2.RewardInfo | None,
  ) -> None:
    """Sets up the observation response and reward info parsers.

    If you lazy initialized the protos, you must call this method to set them.

    Args:
      observation_response: The observation response from the environment. If
        None, the observation response will be retrieved from the environment.
      reward_info: The reward info from the environment. If None, the reward
        info will be retrieved from the environment.
    """
    self._observation_response_parser = self._setup_observation_response(
        observation_response=observation_response,
    )
    self._reward_info_parser = self._setup_reward_info(reward_info=reward_info)

  def _setup_observation_response(
      self,
      observation_response: building_pb2.ObservationResponse | None = None,
  ) -> or_parser.ObservationResponseParser:
    """Returns an observation response parser.

    Args:
      observation_response: The observation response from the environment. If
        None, the observation response will be retrieved from the environment.

    Returns:
      An observation response parser.
    """
    if observation_response is None:
      observation_response = self.env.get_observation_response()

    return or_parser.ObservationResponseParser(
        observation_response=observation_response
    )

  def _setup_reward_info(
      self, reward_info: reward_pb2.RewardInfo | None = None
  ) -> ri_parser.RewardInfoParser:
    """Returns a reward info parser.

    Args:
      reward_info: The reward info from the environment. If None, the reward
        info will be retrieved from the environment.

    Returns:
      A reward info parser.
    """
    if reward_info is None:
      reward_info = self.env.get_reward_info()

    return ri_parser.RewardInfoParser(reward_info=reward_info)

  @property
  def observation_response_parser(self) -> or_parser.ObservationResponseParser:
    """The observation response parser. Assumed to have been set up already."""
    if self._observation_response_parser is None:
      raise ValueError("Observation response parser is None.")
    return self._observation_response_parser

  @property
  def reward_info_parser(self) -> ri_parser.RewardInfoParser:
    """The reward info parser. Assumed to have been set up already."""
    if self._reward_info_parser is None:
      raise ValueError("Reward info parser is None.")
    return self._reward_info_parser

  # DATA AND PROPERTIES

  @property
  def json_metadata(self) -> SerializableData:
    """Info to write into a JSON file. Needs to be serializable."""
    return super().json_metadata | {
        "include_weights": self.include_weights,
        "occupancy_mode_min": self.occupancy_mode_min,
        "temp_display_unit": self.temp_display_unit.value,
        "building_info": dataclasses.asdict(self.building_info),
    }

  @property
  def building_info_series(self) -> pd.Series:
    """A pandas.Series describing the building information."""
    return pd.Series(
        dataclasses.asdict(self.building_info),
        name="building_info"
    )

  @property
  def setpoints_df(self) -> pd.DataFrame:
    """A dataframe describing the devices and setpoints under control.

    Includes information about the range of possible native values for each
    setpoint.

    The LLM will use the device_id and setpoint_name values as a composite key
    to uniquely identify setpoints in its responses.

    Returns:
      A dataframe describing the devices and setpoints under control.
    """
    df = self.env.action_fields_df[[
        "device_id",
        "setpoint_name",
        "setpoint_type",
        "units",
        "min_native_value",
        "max_native_value",
    ]].copy()
    return df.sort_values(by=["device_id", "setpoint_name"]).reset_index(
        drop=True
    )

  @property
  def weights(self) -> dict[str, float] | None:
    """Returns the reward function weights, if available."""
    if hasattr(self.env.reward_function, "weights"):
      weights = self.env.reward_function.weights.copy()
      # Rename "productivity_weight" to "comfort_weight":
      if "productivity_weight" in weights:
        weights["comfort_weight"] = weights.pop("productivity_weight")
      return weights
    return None

  @property
  def weights_series(self) -> pd.Series | None:
    """A pandas.Series describing the reward function weights, if available."""
    if self.weights is not None:
      return pd.Series(self.weights, name="weight")

  @property
  def validity_intervals(self) -> list[int]:
    """A list of validity intervals (in minutes) for the LLM to choose from."""
    return self.output_schema["properties"]["validity_interval"]["enum"]

  # PROMPT CONTENT

  @property
  def base_prompt(self) -> str:
    """The base prompt, excluding formatting instructions."""
    return "\n\n".join([
        "# Agent Instructions",
        self.objectives_section,
        self.zone_info_section,
        self.occupancy_modes_section,
        self.hvac_system_guidelines_section,
        self.action_guidelines_section,
        self.current_conditions_section,
        self.current_action_section,
    ])

  @property
  def objectives_section(self) -> str:
    """A section describing the LLM's role and objectives.

    Includes the reward function weights, if available and enabled via the
    `include_weights` argument.

    Returns:
      A section describing the LLM's role and objectives.
    """

    section = self.dedent(f"""
      ## Objectives

      ### Role

      You are a skilled, experienced, and innovative operator of a commercial office building.
      You possess in-depth and complete knowledge about HVAC systems, as well as ASHRAE standards and certifications.
      Your job is to optimally control HVAC devices in a given commercial office building.

      **Building Information**:

      {self.building_info_series.to_markdown(index=True)}

      ### Overall Goal

      As the building operator, your **Optimal Control Objectives** are to:

        + Minimize energy consumption / costs, and
        + Minimize carbon emissions, and
        + Maintain occupant comfort (a.k.a. productivity)

      This is a multi-objective optimization problem, where you must balance competing objectives.
    """)

    weights_series = self.weights_series
    if self.include_weights and weights_series is not None:
      section += "\n\n" + self.dedent(f"""
        ### Reward Function Weights

        We have assigned a weight to designate the importance of each objective.
        Your job is to maximize the weighted sum of the objectives, placing a higher priority on objectives with greater weights.
        The weights are designated in the table below:

        {weights_series.to_markdown(index=True)}
      """)

    return self.dedent(section)

  @property
  def zone_info_section(self) -> str:
    """A section describing zone related terminology."""

    return self.dedent("""
      ## Zone Information

      A **zone** is a room, or space in the office building that is potentially occupied by humans, and must be conditioned for comfort when occupied.

      ### Zone Comfort

      The **zone air temperature** is the average temperature in a zone and the measure of comfort in the zone.

      The **zone air heating setpoint** is the minimum temperature that zone is allowed to be, without actively heating the zone.
      It's like the minimum of the occupant comfort range.
      The **zone air cooling setpoint** is the maximum temperature that zone is allowed to be, without actively cooling the zone.
      It's like the maximum of the occupant comfort range.
      The zone air heating temperature setpoint is always below the zone air cooling temperature setpoint.

      Ideally: `zone air heating setpoint < zone air temperature if occupied < zone air cooling setpoint`
    """)

  @property
  def occupancy_modes_section(self) -> str:
    """A section describing and contrasting the different occupancy modes."""

    # TODO(mjrossetti): Add a table of hourly occupancy trends, for each day of
    # the week.

    return self.dedent(f"""
      ## Occupancy Modes

      You should operate the building in an occupancy mode and an efficiency mode.

      **Occupancy mode** is when the building has at least {self.occupancy_mode_min} occupants.
      When in occupancy mode, you should try to maintain zone air temperatures within comfort range (for all occupied zones), while also minimizing energy consumption and carbon emissions.

      **Efficiency mode** is when the building has fewer than {self.occupancy_mode_min} occupants.
      When in efficiency mode, your only objective should be to SIGNIFICANTLY reduce energy consumption and carbon emissions.

      ### Heating and Cooling Guidelines

      To save energy, you should transition from efficiency mode to occupancy mode in the morning as late as possible, but early enough to ensure the building is in setpoints when the occupants arrive.
      Depending on the outside air temperature, the building will take some time to get into setpoint ranges, especially in the mornings before transitioning from efficiency mode to occupancy mode.
      Therefore, you must apply heating or cooling early enough to ensure that the setpoint temperatures are met before occupancy mode setpoints are applied.

      Time it takes to increase zone air temperature by 1 degree Fahrenheit:

        + Under standard conditions with lower outside air temperature, and active heating, it takes 10 minutes.
        + Under standard conditions with higher outside air temperature, and no active cooling, it takes 20 minutes.

      Time it takes to decrease zone air temperature by 1 degree Fahrenheit:

        + Under standard conditions with higher outside air temperature, and active cooling, it takes 10 minutes.
        + Under standard conditions with lower outside air temperature, and with no active heating, it takes 20 minutes.
    """)

  @property
  def hvac_system_guidelines_section(self) -> str:
    """A section describing building-specific HVAC system setup and guidelines.

    This section describes the HVAC devices under control, and provides
    guidance for controlling them.
    """

    return self.dedent(f"""
      ## HVAC System Control Guidelines

      There are two systems under your control, with three devices total.
      The Air Handler System (AHS) includes two air handler / air conditioner devices (AC-1 and AC-2).
      The Hot Water System (HWS) includes one boiler device (BLR).

      ### Devices and Setpoints

      **AC-1**: Air Conditioner / Air Handler Unit (for all zones on the first floor)

        * '{hybrid_env.DISCRETE_ACTION_COMMAND}': you can turn the device ON (1) and OFF (0)
        * '{AHU_STATIC_PRESSURE_SETPOINT}': you can increase/decrease airflow by increasing/decreasing static pressure
        * '{AHU_SUPPLY_AIR_TEMPERATURE_SETPOINT}': you can cool the zones by lowering the supply air temperature

      **AC-2**: Air Conditioner / Air Handler Unit (for all zones on the second floor)

        * '{hybrid_env.DISCRETE_ACTION_COMMAND}': you can turn the device ON (1) and OFF (0)
        * '{AHU_STATIC_PRESSURE_SETPOINT}': you can increase/decrease airflow by increasing/decreasing static pressure
        * '{AHU_SUPPLY_AIR_TEMPERATURE_SETPOINT}': you can cool the zones by lowering the supply air temperature

      **BLR**: Boiler (for both floors):

        * '{hybrid_env.DISCRETE_ACTION_COMMAND}': you can turn the device ON (1) and OFF (0)
        * '{HWS_DIFFERENTIAL_PRESSURE_SETPOINT}': you can increase/decrease water flow to the zones by increasing/decreasing differential pressure
        * '{HWS_SUPPLY_WATER_TEMPERATURE_SETPOINT}': you can heat the zones by increasing the water supply temperature

      ### Air Conditioner (AC) / Air Handler (AHU) Guidelines

      Turning on an AC will consume electricity by running the air blowers and running the refrigeration compressors.
      Turning them off will not consume any electricity, but will also remove air cooling and ventilation.

      Lowering an AC's supply air temperature below outside air temperature will cause the compressor to run, consuming electricity, and will cool the zones.
      Setting the supply air temperature only enables you to cool, but not heat the zones.

      Increasing an AC's static pressure will increase air circulation through the zones, which results in cooling or heating the zones.

      ### Boiler (BLR) Guidelines

      Lowering the boiler's supply water temperature will reduce carbon emission, but will also reduce the ability to heat zones.

      ### Zone Temperature Control Guidelines

      If a zone is occupied and the zone air temperature is below the zone air heating temperature setpoint, the VAV in the zone will request air flow and hot water circulation to heat the zone.
      You control air flow by managing the AHU static pressure setpoints, and hot water circulation by managing the HWS differential pressure and supply water temperature setpoints.

      If the zone is occupied and the zone air temperature is above the zone air cooling temperature setpoint, the VAV in the zone will request cool air from the AHU.
      You control the amount of cooling by managing the AHU static pressure and supply air temperature setpoints.
    """)

  @property
  def action_guidelines_section(self) -> str:
    """A section describing the action space."""

    return self.dedent(f"""
      ## Action Guidelines

      Throughout the day, you will be prompted to choose your actions.
      Your actions will be used to control the HVAC systems in the building.
      An action requires a value and justification for each of the device setpoints listed below.

      {self.setpoints_df.to_markdown(index=False)}

      Note about temperature units:
      All temperatures will be reported to you in Kelvin.
      The temperatures you choose to set should be in Kelvin.
      However, in your textual responses and justifications only,
      you should communicate temperatures in {self.temp_display_unit.value} instead,
      accurately converting and translating between units as necessary.
    """)

  @property
  def current_conditions_section(self) -> str:
    """A section describing the current conditions in the building."""

    # TODO(mjrossetti): Add upcoming temperature forecast for at least the next
    # six hours, using interpolation and caching strategies.

    return self.dedent(f"""
      ## Current Conditions

      The current local time is: {self.env.current_local_timestamp.strftime('%A, %B %d, %Y %l:%M %p %Z')}.

      The current outside air temperature is: {self.observation_response_parser.outside_air_temp:.1f} Kelvin.

      Total number of zones: {len(self.env.building.zones)}

      Current number of occupants: {self.reward_info_parser.total_occupancy}.

      Current number of occupants exposed to unacceptable comfort conditions: {self.reward_info_parser.num_occupants_uncomfortable}.

      {self.zone_conditions_subsection}

      ### Current Power Consumption

      The table below shows the current energy consumption for each device:

      {self.reward_info_parser.energy_consumption_df_watts.to_markdown(index=False)}
    """)

  @property
  def zone_conditions_subsection(self) -> str:
    """A subsection describing the current zone conditions.

    For floor-by-floor occupant comfort, see the FloorBasedPromptmaker class.
    """

    return self.dedent(f"""
      ### Current Zone Temperatures

      The table below conveys the comfort conditions across all zones in the building:

      {self.reward_info_parser.zone_conditions_histogram.to_markdown(index=True)}

      The first two rows show the number of zones and the number of occupants at a specific temperature.
      The row marked 'temperature setpoint range' makes a '+' for a temperature inside acceptable range, and a '-' for a temperature outside of acceptable range.
      The row labeled 'count of occupants exposed' indicates the count of all occupants being exposed to unacceptable comfort conditions.
    """)

  @property
  def current_action_section(self) -> str:
    """A section containing guidance for choosing the next action."""

    return self.dedent(f"""
      ## Current Action

      First, observe the building conditions (including occupancy levels, outside air temperature, zone air temperatures, energy consumption levels, etc.), and use this information to devise an overall strategy for your next action.

      According to your strategy, decide to turn each device ON (1) or OFF (0), using their discrete '{hybrid_env.DISCRETE_ACTION_COMMAND}' setpoints.

      For each device, also decide on values for that device's continuous setpoints.
      NOTE: even if the devices are off, you still need to supply values for these continuous setpoints, however they will not be used, so it is ok to choose a value in the middle of the setpoint range.

      Provide an overall justification explaining your strategy in a sentence or two.
      Also provide a justification for each setpoint you chose in a sentence or two.

      Finally, select a validity interval from the following options: {self.validity_intervals}.
      The **validity interval** is the number of minutes the setpoints will remain in effect.
      Choose long validity times when under steady conditions, and only apply short validity intervals when the building is undergoing high amount of change.
      After the validity interval expires, you will be allowed to assign new setpoints.

      IMPORTANT NOTE: you MUST structure your response according to the "Formatting Instructions" below.
    """)
