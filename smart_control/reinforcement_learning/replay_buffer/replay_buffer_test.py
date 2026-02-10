"""Replay Buffer Test"""

import sys
import unittest

# we are skipping these tests on Mac for now, until we can resolve the dm-reverb
# package installation on Mac. See: https://github.com/google/sbsim/issues/102
RUNNING_ON_MAC = sys.platform.startswith("darwin")
SKIP_REASON = "Issues installing dm-reverb on Mac."


class ReverbInstallationTest(unittest.TestCase):
  """Testing if we can install the dm-reverb package. Skipping on Mac for now.
  We can remove the skip logic and push to GitHub Actions to test / prove our
  ability to install across all platforms. Then we can remove this test class.
  """

  @unittest.skipIf(RUNNING_ON_MAC, SKIP_REASON)
  def test_reverb_installation(self):
    import reverb  # pylint:disable=import-outside-toplevel

    print("Reverb imported successfully.")
    print(dir(reverb))
    assert True


# TODO: add more replay buffer related tests here (using the skip logic)

if __name__ == "__main__":
  unittest.main()
