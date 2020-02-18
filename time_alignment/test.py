import unittest
from unittest import TestCase
import numpy as np

from Lagrange1time import Lagrange1time
from Lagrange2time import Lagrange2time

from LSFit1time import LSFit1time
from LSFit2time import LSFit2time

class TestLagrange1time(TestCase):
    def test_lagrange1time(self):
        x = np.array([1,2,3])
        y = np.array([4,7,11])
        self.assertAlmostEqual(Lagrange1time(x, y, 12), 47)

    def test_lagrange2time(self):
        x = np.array([1,2,3])
        y = np.array([4,7,11])
        self.assertAlmostEqual(Lagrange2time(x, y, 12), 92)

    def test_LSFit1time(self):       
        x = np.array([1,2,3])
        y = np.array([4,7,11])
        result = LSFit1time(x, y, 12)
        self.assertAlmostEqual(result[0], 42 + 1 / 3.0)

    def test_LSFit2time(self):       
        x = np.array([1,2,3])
        y = np.array([4,7,11])
        result = LSFit2time(x, y, 12)
        self.assertAlmostEqual(result[0], 92)
 
if __name__ == '__main__':
    unittest.main()
