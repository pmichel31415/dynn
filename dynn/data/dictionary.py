#!/usr/bin/env python3
"""
Dictionary
^^^^^^^^^^

Dictionary object for holding string to index mappings
"""
from collections import defaultdict
import logging

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"


class Dictionary(object):

    def __init__(self, symbols=None, special_symbols=None):
        # Frozen means you can't add symbols
        self.frozen = False
        # Symbols map ints to, well, symbols
        self.symbols = []
        # Indices does the reverse (symbol to int)
        self.indices = {}
        # UNK (for unknown words)
        self.unk_idx = self.add(UNK_TOKEN)
        self.unk_tok = UNK_TOKEN
        # PAD (for padding)
        self.pad_idx = self.add(PAD_TOKEN)
        self.pad_tok = PAD_TOKEN
        # EOS (End Of Sentence)
        self.eos_idx = self.add(EOS_TOKEN)
        self.eos_tok = EOS_TOKEN
        # Additional special symbols
        if special_symbols is not None:
            for special_symbol in special_symbols:
                self.add(special_symbol)
        # Number of special symbols (easier to test for special symbols by
        # just checking whether idx < nspecials)
        self.nspecials = len(self.symbols)
        # Add symbols
        if symbols is not None:
            for symbol in symbols:
                self.add(symbol)

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, idx):
        return self.symbols[idx]

    def index(self, symbol, fail_if_unknown=False):
        """Returns the symbol's index

        By default this returns the ``<unk>`` index if the symbol is not in
        the dictionary

        Args:
            symbol (str): Symbol
            fail_if_unknown (bool): Fail with an error if the symbol is not in
                the dictionary (instead of returning the ``unk`` index)

        Returns:
            int: Symbol index
        """
        # Handle unknown symbols
        if symbol not in self.indices:
            # Either fail or return ``unk_idx``
            if fail_if_unknown:
                raise ValueError(f"{symbol} not in dictionary")
            else:
                return self.unk_idx
        # Otherwise return index
        return self.indices[symbol]

    def add(self, symbol):
        """Add a symbol to the dictionary and return its index

        If the dictionary is frozen or the symbol is already in it this
        doesn't do anything (except returning the index)

        Args:
            symbol (str): Symbol to add

        Returns:
            int: Symbol index (might be the ``unk`` index if the dictionary is
                frozen and the symbol isn't in the dictionary)
        """

        # Ignore existing symbols
        if not self.frozen and symbol not in self.indices:
            self.indices[symbol] = len(self)
            self.symbols.append(symbol)
        # Return index
        return self.index(symbol)

    def freeze(self):
        """Freeze the dictionary

        You can't add new symbols to a frozen dictionaries
        """
        self.frozen = True

    def thaw(self):
        """Un-freeze the dictionary

        Just like with food, it's not recommended to freeze/thaw a dictionary
        a lot. You should freeze it after reading the train data.
        """
        self.frozen = False

    def numberize(self, data):
        """Recursively descend into ``data`` and convert strings to indices.

        Args:
            data (list,str): Either a string or a list (of list)* of strings

        Returns:
            list,int: Same structure but with indices instead of strings
        """
        if isinstance(data, str):
            return self.index(data)
        if isinstance(data, dict):
            return {k: self.numberize(v) for k, v in data.items()}
        else:
            return [self.numberize(item) for item in data]

    def string(self, idxs, with_pad=False, with_eos=False, join_with=None):
        """Converts a list of indices to strings

        Depending on whether ``join_with`` is specified this either returns a
        string or a list of strings.

        Args:
            idxs (list): List of ints
            with_pad (bool, optional): Defaults to False. Include padding
                tokens
            with_eos (bool, optional): Defaults to False. Include EOS tokens
            join_with (str, optional): Defaults to None. Join the list into one
            string with this specific character

        Returns:
            str, list: Either a string or list of strings if ``join_with`` is
                ``None``
        """
        # Filter out special tokens
        idxs = [idx for idx in idxs
                if (idx != self.pad_idx or with_pad)
                and (idx != self.eos_idx or with_eos)]
        # Back to words
        words = [self[idx] for idx in idxs]
        # Join
        if join_with is not None:
            words = join_with.join(words)

        return words

    @staticmethod
    def from_data(
        data,
        min_count=1,
        max_size=-1,
        symbols=None,
        special_symbols=None
    ):
        """Build a dictionary from a dataset

        There is a variety of options to filter by frequency

        Args:
            data (list): List of list of strings
            min_count (int, optional): All symbols appearing less than
                ``min_count`` will be treated as ``unk`` s. (default: 1)
            max_size (int, optional): Only include the top ``max_size``
                most frequent symbols. Ignore if ``<=0``. (default: -1)
            symbols (list, optional): List of tokens to definitely include.
                (default: None)
            special_symbols (list, optional): Additional special tokens.
                (default: None)

        Returns:
            Dictionary: Brand new dictionary objects
        """
        counts = defaultdict(lambda: 0)
        # Count occurences
        for seq in data:
            for symbol in seq:
                counts[symbol] += 1

        # Filter by frequency
        most_freq = {
            symbol: count
            for symbol, count in counts.items() if count >= min_count
        }

        # Sort by frequency decreasing frequency
        most_freq = sorted(most_freq.items(), key=lambda x: -x[1])

        if max_size > 0:
            # Take top ``max_size`` most frequents
            n_forced_symbols = 0 if symbols is None else len(symbols)
            if n_forced_symbols > max_size:
                logging.warning(
                    f"You requested a maximum dictionary size of {max_size} "
                    f"but provided {len(symbols)} to include. The dictionary "
                    f"will have size {len(symbols)}."
                )
            if max_size < len(most_freq) + n_forced_symbols:
                most_freq = most_freq[:max(max_size - n_forced_symbols, 0)]

        # Add symbols
        symbols = symbols or []
        symbols.extend([symbol for symbol, _ in most_freq])

        # Actually create dictionary
        dic = Dictionary(symbols=symbols, special_symbols=special_symbols)

        return dic

    def save(self, filename, symbols_only=False):
        """Save the dictionary to a text file.

        Args:
            filename (str): Target filename.
            symbols_only (bool, optional): Defaults to False. If set to
                ``True``, only the symbols will be printed (and no header
                information about the size, frozen status and number of
                special symbols). Use this mostrly if you want to use the
                dictionary for other purposes.
        """

        with open(filename, "w", encoding="utf-8") as dic_file:
            if not symbols_only:
                header = f"{len(self)}\t{self.frozen}\t{self.nspecials}"
                print(header, file=dic_file)
            for symbol in self.symbols:
                print(symbol, file=dic_file)

    @staticmethod
    def load(filename):
        """Load the dictionary from a text file

        The text file can just contain a list of one symbol per line.

        Args:
            filename (str): File to load the dictionary from

        Returns:
            Dictionary: Resulting dictionary
        """

        symbols = []
        expected_size = None
        frozen = True
        nspecials = 0
        with open(filename, "r", encoding="utf-8") as dic_file:
            maybe_header = dic_file.readline().strip()
            if len(maybe_header.split("\t")) > 1:
                expected_size, frozen, nspecials = maybe_header.split("\t")
                expected_size = int(expected_size)
                frozen = frozen is "True"
                nspecials = int(nspecials)
            else:
                symbols.append(maybe_header)
            # Continue iterating
            for line in dic_file:
                symbols.append(line.strip())
        # check size
        if expected_size is not None and len(symbols) != expected_size:
            raise ValueError("Size mismatch in dictionary file")
        # Create dictionary
        special_symbols = symbols[:nspecials]
        symbols = symbols[nspecials:]
        dic = Dictionary(symbols=symbols, special_symbols=special_symbols)
        # Freeze maybe
        if frozen:
            dic.freeze()
        # Return
        return dic


__all__ = ["Dictionary"]
