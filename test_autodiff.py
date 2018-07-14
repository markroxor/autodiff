#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from autodiff import Constant, Function, Variable
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestAutodiff(unittest.TestCase):
    def setUp(self):
        t = np.arange(0.0, 1.0, 0.01)
        self.x = 2*np.pi*t

    def test_sin(self):
        exp = Function.Sin(Constant(self.x))

        self.assertTrue(np.allclose(exp.evaluate(), np.sin(self.x)))
        self.assertTrue(np.allclose(exp.grad(), np.cos(self.x)))

    def test_cos(self):
        exp = Function.Cos(Constant(self.x))

        self.assertTrue(np.allclose(exp.evaluate(), np.cos(self.x)))
        self.assertTrue(np.allclose(exp.grad(), -np.sin(self.x)))

    def test_tan(self):
        exp = Function.Tan(Constant(self.x))
         
        self.assertTrue(np.allclose(exp.evaluate(), np.tan(self.x)))
        self.assertTrue(np.allclose(exp.grad(), np.cos(self.x) ** -2))

    def test_complex_exp(self):
        exp = Function.Sin(Variable('x') ** Variable('x'))
        exp.update({'x':1})
        print(exp.evaluate())

        logger.warning("a")
        self.assertTrue(exp.evaluate(), 0.8414)
        self.assertTrue(exp.grad(), 0.5403)