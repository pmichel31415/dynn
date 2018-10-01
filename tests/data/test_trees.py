#!/usr/bin/env python3

import unittest
from unittest import TestCase

from dynn.data.trees import Tree


class TestTree(TestCase):

    def setUp(self):
        self.test_tree_str = (
            "(4 (3 (3 (3 (3 (2 (2 The) (2 production)) (3 (2 has) (4 (2 been) "
            "(3 (2 made) (4 (2 with) (3 (2 (2 an) (2 (3 enormous) (2 amount)))"
            " (2 (2 of) (3 affection)))))))) (2 ,)) (2 so)) (2 (2 we) (2 (2 "
            "believe) (3 (2 (2 these) (2 characters)) (2 (4 love) (2 (2 each) "
            "(2 other))))))) (2 .))"
        )
        self.label = 4
        self.sentence = (
            "The production has been made with an enormous amount of affection"
            " , so we believe these characters love each other ."
        )

    def test_read_tree(self):
        tree = Tree.from_string(self.test_tree_str)
        self.assertEqual(str(tree), self.test_tree_str)
        self.assertEqual(tree.label, self.label)
        self.assertEqual(" ".join(tree.leaves()), self.sentence)


if __name__ == '__main__':
    unittest.main()
