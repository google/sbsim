"""Stochastic simulator of convection flow in bldg.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

A convection simulator that randomly shuffles control volumes as a stochastic
imitation of convection
We specify probability of a control volume getting shuffled, as well as maximum
distance that any given
control volume can be moved.
"""

import collections
import copy
import random
from typing import MutableSequence, Optional

import gin
import numpy as np
from smart_control.simulator import base_convection_simulator


@gin.configurable
class StochasticConvectionSimulator(
    base_convection_simulator.BaseConvectionSimulator
):
  """Stochastic method of simulating air convection.

  Attributes:
    _p: shuffling probability
    _distance: max distance a CV can move
  """

  def __init__(self, p: float, distance: int, seed: Optional[int]):
    """Initializes the stochastic convection simulator.

    Args:
      p: shuffling probability
      distance: max distance a CV can move
      seed: random seed
    """
    self._p = p
    self._distance = distance
    # cache is used to store precalculated shuffling patterns
    # so we do not need to calculate them again
    self._cache = collections.defaultdict(lambda: {})

    if seed is not None:
      random.seed(seed)

  def apply_convection(
      self,
      room_dict: dict[str, MutableSequence[tuple[int, int]]],
      temp: np.ndarray,
  ) -> None:
    """Applies convection to the temperature array given, splitting up rooms via room_dict."""
    p = self._p
    distance = self._distance
    if p == 0 or distance == 0:
      return

    for k, v in room_dict.items():
      if k == "exterior_space":
        continue
      if k == "interior_wall":
        continue
      if distance == -1 and p == 1:  # special case, can be more efficient
        self._shuffle_no_max_dist(v, temp)
      else:
        self._shuffle_max_dist(p, v, distance, temp)

  def _shuffle_no_max_dist(self, v, temp):
    """Special case of shuffling when no max dist is specified.and p=1.

    Args:
      v: list of CVs to shuffle
      temp: temperature array
    """
    v = copy.deepcopy(v)
    vals = {}
    for cv in v:
      vals[cv] = temp[cv[0], cv[1]]
    v_shuffle = copy.deepcopy(v)
    random.shuffle(v_shuffle)
    for i, _ in enumerate(v_shuffle):
      cv = v_shuffle[i]
      val = vals[v[i]]
      temp[cv[0], cv[1]] = val

  def _shuffle_max_dist(self, p, v, max_dist, temp):
    """Special case of shuffling when max dist is specified.

    Args:
      p: shuffle probability
      v: list of CVs to shuffle
      max_dist: max distance to shuffle
      temp: temperature array
    """
    if max_dist == -1:
      max_dist = 1000
    cv_is_in_v = {}
    for val in v:
      cv_is_in_v[val] = True

    swap_list = []
    for val in v:
      if random.uniform(0, 1) > p:
        continue
      if max_dist in self._cache and val in self._cache[max_dist]:
        candidates = self._cache[max_dist][val]
      else:
        candidates = []
        cv = val
        for cv_0 in range(cv[0] - max_dist, cv[0] + max_dist):
          for cv_1 in range(cv[1] - max_dist, cv[1] + max_dist):
            cv_2 = (cv_0, cv_1)
            # check if cv_2 is in the room
            if cv_2 not in cv_is_in_v:
              continue
            dist = (cv[0] - cv_2[0]) ** 2 + (cv[1] - cv_2[1]) ** 2
            if dist <= max_dist:
              candidates.append(cv_2)
        self._cache[max_dist][val] = candidates

      swap_list.append((val, random.choice(candidates)))
    random.shuffle(swap_list)

    for i, _ in enumerate(swap_list):
      cv_1 = swap_list[i][0]
      cv_2 = swap_list[i][1]

      val = temp[cv_1[0], cv_1[1]]
      temp[cv_1[0], cv_1[1]] = temp[cv_2[0], cv_2[1]]
      temp[cv_2[0], cv_2[1]] = val
