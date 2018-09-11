#
import unittest
from unittest import TestCase
import dynet as dy
from dynn.layers import lstm


class TestLSTM(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.dh = 10
        self.di = 20
        self.dropout = 0.1

    def test_compactlstm(self):
        # Create compact lstm
        compact_lstm = lstm.CompactLSTM(
            self.di, self.dh, self.pc, self.dropout)
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        h0 = dy.random_uniform(self.dh, -1, 1)
        c0 = dy.random_uniform(self.dh, -1, 1)
        x = dy.random_uniform(self.di, -1, 1)
        # Initialize layer
        compact_lstm.init(test=False, update=True)
        # Run lstm cell
        h, c = compact_lstm(h0, c0, x)
        # Try forward/backward
        z = dy.dot_product(h, c)
        z.forward()
        z.backward()


if __name__ == '__main__':
    unittest.main()
