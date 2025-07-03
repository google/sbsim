"""Tests for radiation utility functions."""

from absl.testing import absltest
import numpy as np
from numpy.testing import assert_array_almost_equal

from smart_control.simulator import constants
from smart_control.simulator.building_radiation_utils import calculate_A_tilde_inv
from smart_control.simulator.building_radiation_utils import calculate_IFAinv
from smart_control.simulator.building_radiation_utils import mark_air_connected_interior_walls
from smart_control.simulator.building_radiation_utils import mark_directly_seeing_nodes
from smart_control.simulator.building_radiation_utils import net_radiative_heatflux_function_of_T

# we are choosing to keep the mathematical notation names
# pylint: disable=invalid-name


class RadiationUtilsTest(absltest.TestCase):

  def test_calculate_A_tilde_inv_and_IFAinv(self):
    epsilon = np.array([0.8, 0.4, 0.8])
    F = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    expected_A_tilde_inv = np.array([
        [0.83982684, 0.04761905, 0.11255411],
        [0.28571429, 0.42857143, 0.28571429],
        [0.11255411, 0.04761905, 0.83982684],
    ])

    expected_IFAinv = np.array([
        [0.64069264, -0.19047619, -0.45021645],
        [-0.19047619, 0.38095238, -0.19047619],
        [-0.45021645, -0.19047619, 0.64069264],
    ])

    A_tilde_inv = calculate_A_tilde_inv(epsilon, F)
    IFAinv = calculate_IFAinv(F, A_tilde_inv)
    with self.subTest("A_tilde_inv shape"):
      self.assertEqual(A_tilde_inv.shape, F.shape)
    with self.subTest("IFAinv shape"):
      self.assertEqual(IFAinv.shape, F.shape)

    with self.subTest("A_tilde_inv"):
      assert_array_almost_equal(A_tilde_inv, expected_A_tilde_inv, decimal=3)
    with self.subTest("IFAinv"):
      assert_array_almost_equal(IFAinv, expected_IFAinv, decimal=3)

  def test_net_radiative_heatflux_function_of_T(self):
    # fmt: off
    #pylint:disable=line-too-long
    temperatures=np.array([1200,500,1102])#  [K]
    IFAinv = np.array([
        [0.64069264, -0.19047619, -0.45021645],
        [-0.19047619, 0.38095238, -0.19047619],
        [-0.45021645, -0.19047619, 0.64069264],
    ])
    # fmt: on
    # pylint:enable=line-too-long
    expected_q = np.array([3.70061961e04, -3.69724724e04, -3.37237040e01])

    q = net_radiative_heatflux_function_of_T(temperatures, IFAinv)

    with self.subTest("q results as expected"):
      assert_array_almost_equal(
          np.round(q, 4), np.round(expected_q, 4), decimal=4
      )

  def test_mark_air_connected_interior_walls(self):
    # fmt: off
    #pylint:disable=line-too-long

    indexed_floor_plan =\
      np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
                [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3, -3, -3,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3,  0, -3,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3,  0, -3,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3,  0, -3,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
                [-1, -2, -3,  0, -3,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0, -3,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3, -3, -3,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
                [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])


    expected_result = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -2,  -3, -33, -33, -33, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33, -33, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -33, -33,  -3, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])
    # fmt: on
    # pylint:enable=line-too-long
    result, _ = mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(2, 3),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=-33,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )
    # setup (temperatures and IFAinv have same number of rows):
    self.assertEqual(result.shape, indexed_floor_plan.shape)

    # result has same shape as the temperatures array:
    with self.subTest("result as expected"):
      assert_array_almost_equal(result, expected_result)

  def test_mark_directly_seeing_nodes(self):
    # fmt: off
    #pylint:disable=line-too-long

    indexed_floor_plan =\
       np.array(
        [[ -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
        [ -1,  -2, -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
        [ -1,  -2, -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,  -3,  -3,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
        [ -1,  -2, -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
        [ -1,  -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]]
      )



    expected_result_23 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -2,  -3, -67, -34, -34, -34, -34, -34,  -3,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -33, -33, -33,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -33,   0, -34,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -33,   0, -34,   0, -34,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -33,   0, -34,   0, -34,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -33,   0,   0,   0, -34,  -2,  -1],
       [ -1,  -2,  -3, -33, -33,  -3, -34, -34, -34,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])

    expected_result_27 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -2,  -3, -34, -34, -34, -34, -67, -34,  -3,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -33, -33, -33,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0, -34,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0, -34,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0, -34,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2,  -3, -34, -34,  -3, -34, -34, -33,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])

    expected_result_116 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -2,  -3, -34, -34, -34, -34, -34, -34,  -3,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -34,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -34,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -34,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33, -33, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -34, -34, -34, -67, -34, -34,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0, -33,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3,   0, -33,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -34, -33,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -33, -33, -33, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])
    # fmt: on
    # pylint:enable=line-too-long
    result23_, _ = mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(2, 3),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=-33,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )

    result23 = mark_directly_seeing_nodes(
        floor_plan=result23_, base_node=(2, 3)
    )

    # result has same shape as the temperatures array:
    with self.subTest("result as expected"):
      assert_array_almost_equal(result23, expected_result_23)

    result27_, _ = mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(2, 7),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=-33,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )

    result27 = mark_directly_seeing_nodes(
        floor_plan=result27_, base_node=(2, 7)
    )

    # result has same shape as the temperatures array:
    with self.subTest("result as expected"):
      assert_array_almost_equal(result27, expected_result_27)

    result116_, _ = mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(11, 6),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=-33,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )

    result116 = mark_directly_seeing_nodes(
        floor_plan=result116_, base_node=(11, 6)
    )

    # result has same shape as the temperatures array:
    with self.subTest("result as expected"):
      assert_array_almost_equal(result116, expected_result_116)


if __name__ == "__main__":
  absltest.main()
