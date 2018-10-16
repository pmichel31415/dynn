#!/usr/bin/env python3
import os
import unittest
from unittest import TestCase
import tempfile
import shutil

import numpy as np
import dynet as dy

from dynn import io


class TestUtil(TestCase):

    def setUp(self):
        self.max_dim = 50
        self.path = tempfile.mkdtemp()
        self.filename = os.path.join(self.path, "test.pc.npz")

    def tearDown(self):
        shutil.rmtree(self.path)

    def _dummy_pc(self):
        pc = dy.ParameterCollection()
        for d in range(1, self.max_dim):
            name = "a"*d
            subpc = pc.add_subcollection(name=f"{name}-pc")
            subpc.add_parameters((d, d+1), name=name)
            subpc.add_parameters((d+1, d), name=name)
            subpc.add_lookup_parameters((d+1, d+2), name=f"{name}-lookup")
            subpc.add_lookup_parameters((d+2, d+1), name=f"{name}-lookup")
        return pc

    def test_load(self):
        # Create collection and save
        pc1 = self._dummy_pc()
        io.save(pc1, self.filename)
        # Load
        pc2 = io.load(self.filename)
        # Compare parameter values
        for p1, p2 in zip(pc1.parameters_list(), pc2.parameters_list()):
            self.assertEqual(p1.name(), p2.name())
            self.assertTrue(np.allclose(p1.as_array(), p2.as_array()))
        # Compare lookup parameter values
        for lp1, lp2 in zip(
            pc1.lookup_parameters_list(), pc2.lookup_parameters_list()
        ):
            self.assertEqual(lp1.name(), lp2.name())
            self.assertTrue(np.allclose(lp1.as_array(), lp2.as_array()))

    def test_populate(self):
        # Create collection and save
        pc1 = self._dummy_pc()
        io.save(pc1, self.filename)
        # Populate new pc
        pc2 = self._dummy_pc()
        # Compare parameter values
        io.populate(pc2, self.filename)
        for p1, p2 in zip(pc1.parameters_list(), pc2.parameters_list()):
            self.assertEqual(p1.name(), p2.name())
            self.assertTrue(np.allclose(p1.as_array(), p2.as_array()))
        # Compare lookup parameter values
        for lp1, lp2 in zip(
            pc1.lookup_parameters_list(), pc2.lookup_parameters_list()
        ):
            self.assertEqual(lp1.name(), lp2.name())
            self.assertTrue(np.allclose(lp1.as_array(), lp2.as_array()))

    def test_name_error(self):
        # Wrong file
        wrong_params = {"___not_a_param___": np.zeros(10)}
        np.savez(self.filename, **wrong_params)
        # No error
        io.load(self.filename, ignore_invalid_names=True)
        # Error

        def will_raise(): io.load(self.filename)
        self.assertRaises(ValueError, will_raise)

    def test_shape_error(self):
        # Parameter collection
        pc1 = dy.ParameterCollection()
        pc1.add_parameters(10, name="this-one")
        # Save
        io.save(pc1, self.filename)
        # Second parameter collection with wrong param
        pc2 = dy.ParameterCollection()
        pc2.add_parameters(20, name="this-one")
        pc2.add_parameters(3, name="another-one")
        # No error
        io.populate(pc2, self.filename, ignore_shape_mismatch=True)
        # Error

        def will_raise(): io.populate(pc2, self.filename)
        self.assertRaises(ValueError, will_raise)


if __name__ == '__main__':
    unittest.main()
