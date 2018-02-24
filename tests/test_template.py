"""Testing template"""
import unittest
from .context import bfr


class BasicTests(unittest.TestCase):
    """Basic test cases."""

    def test(self):
        """Tests imports from bfr package"""
        self.assertEqual("Hello world!", bfr.hello_world(), "Return value differs")


if __name__ == '__main__':
    unittest.main()
