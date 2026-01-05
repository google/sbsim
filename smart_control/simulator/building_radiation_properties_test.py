"""Tests for building radiation properties."""

from absl.testing import absltest

from smart_control.simulator import building


class RadiationPropertiesTest(absltest.TestCase):

  def test_validations(self):
    with self.subTest("alpha should be between 0 and 1"):
      building.RadiationProperties(alpha=0, epsilon=0, tau=0)
      building.RadiationProperties(alpha=0.5, epsilon=0, tau=0)
      building.RadiationProperties(alpha=1, epsilon=0, tau=0)

      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=-0.5, epsilon=0.0, tau=0.0)
      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=1.5, epsilon=0.0, tau=0.0)

    with self.subTest("epsilon should be between 0 and 1"):
      building.RadiationProperties(alpha=0, epsilon=0, tau=0)
      building.RadiationProperties(alpha=0, epsilon=0.5, tau=0)
      building.RadiationProperties(alpha=0, epsilon=1, tau=0)

      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=0, epsilon=-0.5, tau=0)
      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=0, epsilon=1.5, tau=0)

    with self.subTest("tau should be between 0 and 1"):
      building.RadiationProperties(alpha=0, epsilon=0, tau=0)
      building.RadiationProperties(alpha=0, epsilon=0, tau=0.5)
      building.RadiationProperties(alpha=0, epsilon=0, tau=1)

      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=0, epsilon=0, tau=-0.5)
      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=0, epsilon=0, tau=1.5)

    with self.subTest("rho should be between 0 and 1"):
      building.RadiationProperties(alpha=0, epsilon=0, tau=0, rho=None)
      building.RadiationProperties(alpha=0, epsilon=0, tau=1, rho=0)
      building.RadiationProperties(alpha=0, epsilon=0, tau=0.5, rho=0.5)
      building.RadiationProperties(alpha=0, epsilon=0, tau=0, rho=1)

      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=0, epsilon=0, tau=0, rho=-0.5)
      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=0, epsilon=0, tau=0, rho=1.5)

    with self.subTest("rho gets set automatically if omitted"):
      props = building.RadiationProperties(alpha=0, epsilon=0, tau=0)
      self.assertEqual(props.rho, 1)

      props = building.RadiationProperties(alpha=0, epsilon=0, tau=0.5)
      self.assertEqual(props.rho, 0.5)

      props = building.RadiationProperties(alpha=0.5, epsilon=0, tau=0)
      self.assertEqual(props.rho, 0.5)

    with self.subTest("sum of alpha, rho, and tau should be 1"):
      building.RadiationProperties(alpha=0.5, epsilon=0, tau=0.5, rho=None)
      building.RadiationProperties(alpha=0.5, epsilon=0, tau=0.5, rho=0)
      building.RadiationProperties(alpha=0, epsilon=0, tau=0.5, rho=0.5)

      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=0.5, epsilon=0.5, tau=0.6, rho=None)
      with self.assertRaises(ValueError):
        building.RadiationProperties(alpha=0.5, epsilon=0.5, tau=0.5, rho=0.1)

  def test_defaults(self):
    with self.subTest("inside air defaults:"):
      props = building.DefaultInsideAirRadiationProperties()
      self.assertEqual(props.alpha, 0)
      self.assertEqual(props.epsilon, 0)
      self.assertEqual(props.tau, 1)
      self.assertEqual(props.rho, 0)

    with self.subTest("inside wall defaults:"):
      props = building.DefaultInsideWallRadiationProperties()
      self.assertEqual(props.alpha, 0.2)
      self.assertEqual(props.epsilon, 0.8)
      self.assertEqual(props.tau, 0)
      self.assertEqual(props.rho, 0.8)

    with self.subTest("exterior wall defaults"):
      props = building.DefaultExteriorWallRadiationProperties()
      self.assertEqual(props.alpha, 0.65)
      self.assertEqual(props.epsilon, 0.93)
      self.assertEqual(props.tau, 0)
      self.assertEqual(props.rho, 0.35)


if __name__ == "__main__":
  absltest.main()
