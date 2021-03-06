#!/usr/bin/env python3
"""
Trees
^^^^^

Helper functions to handle tree-structured data
"""
import re


# The following is mostly copied from
# https://github.com/clab/dynet/blob/master/examples/treelstm/dataloader.py

linearized_tree_tokenizer = re.compile(r" +|[()]|[^ ()]+")


def _tokenize_linearized_tree(s):
    toks = [
        t for t in [
            match.group(0)
            for match in linearized_tree_tokenizer.finditer(s)
        ]
        if t[0] != " "]

    return toks


def _within_bracket(toks, labelled=True):
    label = next(toks) if labelled else 0
    children = []
    for tok in toks:
        if tok == "(":
            children.append(_within_bracket(toks, labelled=labelled))
        elif tok == ")":
            return Tree(label, children)
        else:
            children.append(Tree(tok, None))
    print(list(toks))
    raise RuntimeError('Error Parsing sexpr string')


class Tree(object):
    """Tree object for syntax trees"""

    def __init__(self, label, children=None):
        self.label = label if children is None else int(label)
        self.children = children

    @staticmethod
    def from_string(string, labelled=True):
        """Reads linearized tree from string

        Args:
            string (str): Linearized tree

        Returns:
            Tree: Tree object
        """
        string_toks = _tokenize_linearized_tree(string)
        # In some datasets (looking @ you SNLI) singel word sentences
        # don't have parses
        if len(string_toks) == 1:
            return Tree(string, None)
        toks = iter(string_toks)
        if next(toks) != "(":
            print(string)
            print(list(toks))
            raise RuntimeError('Error Parsing sexpr string')
        return _within_bracket(toks, labelled=labelled)

    def __str__(self):
        if self.children is None:
            return self.label

        children_str = " ".join([f"{child}" for child in self.children])
        return f"({self.label} {children_str})"

    def isleaf(self):
        return self.children is None

    def leaves_iter(self):
        if self.isleaf():
            yield self.label
        else:
            for c in self.children:
                for l in c.leaves_iter():
                    yield l

    def leaves(self):
        return list(self.leaves_iter())

    def nonterms_iter(self):
        if not self.isleaf():
            yield self
            for c in self.children:
                for n in c.nonterms_iter():
                    yield n

    def nonterms(self):
        return list(self.nonterms_iter())
