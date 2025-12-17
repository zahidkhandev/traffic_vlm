import json
import os
import re

from config.dataset_config import DatasetConfig


class SimpleTokenizer:
    """
    A simple word-level tokenizer for the Traffic VLM.
    Handles vocabulary building, encoding, and decoding with special tokens.
    """

    def __init__(self):
        """Initialize the tokenizer with config and special tokens."""
        self.cfg = DatasetConfig()
        self.vocab = {}
        self.inverse_vocab = {}

        self.specials = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def build_vocabulary(self, command_files):
        """
        Scans the provided JSON command files to build a unique vocabulary.

        Args:
            command_files (list): List of filenames (e.g., ['train_commands.json'])
        """
        print("Building vocabulary...")
        unique_words = set()

        for file_path in command_files:
            full_path = os.path.join(self.cfg.output_dir, file_path)
            if not os.path.exists(full_path):
                print(f"Warning: {full_path} not found.")
                continue

            with open(full_path, "r") as f:
                commands = json.load(f)

            for item in commands:
                q_words = self._clean_and_split(item["q"])
                unique_words.update(q_words)

                a_words = self._clean_and_split(item["a"])
                unique_words.update(a_words)

        sorted_words = sorted(list(unique_words))

        self.vocab = {
            word: idx + len(self.specials) for idx, word in enumerate(sorted_words)
        }

        for idx, token in enumerate(self.specials):
            self.vocab[token] = idx

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        print(f"Vocabulary built: {len(self.vocab)} tokens.")
        self.save_vocab()

    def _clean_and_split(self, text):
        """
        Preprocesses text by lowercasing and removing punctuation.

        Returns:
            list: A list of cleaned words.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def encode(self, text, max_len=None):
        """
        Converts a string into a list of token IDs.
        Adds [SOS] at the start and [EOS] at the end.

        Args:
            text (str): The input sentence.
            max_len (int, optional): Max length for padding/truncation.

        Returns:
            list: List of integers (token IDs).
        """
        words = self._clean_and_split(text)
        ids = [self.vocab.get(w, self.unk_token_id) for w in words]

        ids = [self.sos_token_id] + ids + [self.eos_token_id]

        if max_len:
            if len(ids) > max_len:
                ids = ids[:max_len]
                ids[-1] = self.eos_token_id
            else:
                ids = ids + [self.pad_token_id] * (max_len - len(ids))

        return ids

    def decode(self, ids):
        """
        Converts a list of token IDs back into a string.
        Stops at [EOS] and ignores [PAD] and [SOS].

        Args:
            ids (list): List of integers.

        Returns:
            str: The decoded sentence.
        """
        words = []
        for idx in ids:
            if idx == self.pad_token_id:
                continue
            if idx == self.eos_token_id:
                break
            if idx == self.sos_token_id:
                continue

            words.append(self.inverse_vocab.get(idx, "[UNK]"))

        return " ".join(words)

    def save_vocab(self):
        """Saves the vocabulary dictionary to a JSON file."""
        path = os.path.join(self.cfg.output_dir, "vocab.json")
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)
        print(f"Saved vocabulary to {path}")

    def load_vocab(self):
        """Loads the vocabulary from the JSON file."""
        path = os.path.join(self.cfg.output_dir, "vocab.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.vocab = json.load(f)
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            print(f"Loaded vocabulary ({len(self.vocab)} tokens)")
        else:
            print("Vocab file not found. Run build_vocabulary() first.")


if __name__ == "__main__":
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocabulary(["train_commands.json", "val_commands.json"])

    test_str = "Is there a red car?"
    ids = tokenizer.encode(test_str, max_len=10)
    decoded = tokenizer.decode(ids)

    print(f"\nTest Sentence: '{test_str}'")
    print(f"Encoded IDs: {ids}")
    print(f"Decoded: '{decoded}'")
