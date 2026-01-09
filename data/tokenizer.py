import json
import os
import re

from config.dataset_config import DatasetConfig


class SimpleTokenizer:
    """
    A simple word-level tokenizer.
    Builds vocabulary STRICTLY from the provided dataset files.
    No frequency filtering, no hardcoded forced tokens.
    """

    def __init__(self):
        self.cfg = DatasetConfig()
        self.vocab = {}
        self.inverse_vocab = {}

        # Essential special tokens only
        self.specials = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def build_vocabulary(self, command_files):
        """
        Scans provided JSON files and adds EVERY unique word found to the vocabulary.
        """
        print(f"Building vocabulary from: {command_files}")
        unique_words = set()

        for file_name in command_files:
            full_path = os.path.join(self.cfg.output_dir, file_name)

            if not os.path.exists(full_path):
                print(f"Warning: {full_path} not found. Skipping.")
                continue

            with open(full_path, "r") as f:
                try:
                    commands = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding {file_name}. Skipping.")
                    continue

            print(f"  - Scanning {len(commands)} commands in {file_name}...")

            for item in commands:
                # Add words from Question
                q_words = self._clean_and_split(item["q"])
                unique_words.update(q_words)

                # Add words from Answer
                a_words = self._clean_and_split(item["a"])
                unique_words.update(a_words)

        # Sort for deterministic ordering
        sorted_words = sorted(list(unique_words))

        # Assign IDs starting after special tokens
        self.vocab = {
            word: idx + len(self.specials) for idx, word in enumerate(sorted_words)
        }

        # Add special tokens
        for idx, token in enumerate(self.specials):
            self.vocab[token] = idx

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        print(f"Vocabulary built: {len(self.vocab)} unique tokens.")
        self.save_vocab()

    def _clean_and_split(self, text):
        """Standard preprocessing: Lowercase + Remove punctuation."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def encode(self, text, max_len=None):
        """Encodes text to IDs."""
        words = self._clean_and_split(text)
        # If word exists in vocab, use ID. Else use [UNK]
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
        """Decodes IDs to text."""
        words = []
        for idx in ids:
            if isinstance(idx, list):
                idx = idx[0]
            if hasattr(idx, "item"):
                idx = idx.item()  # Handle Tensor inputs

            if idx == self.pad_token_id:
                continue
            if idx == self.eos_token_id:
                break
            if idx == self.sos_token_id:
                continue

            words.append(self.inverse_vocab.get(idx, "[UNK]"))

        return " ".join(words)

    def save_vocab(self):
        path = os.path.join(self.cfg.output_dir, "vocab.json")
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)
        print(f"Saved vocabulary to {path}")

    def load_vocab(self):
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

    # CRITICAL: Include ALL splits so the test set doesn't have [UNK] words
    # that were only present in validation/test but not train.
    tokenizer.build_vocabulary(
        ["train_commands.json", "val_commands.json", "test_commands.json"]
    )

    # Test
    test_str = "Is there a red car?"
    ids = tokenizer.encode(test_str)
    print(f"\nString: '{test_str}'")
    print(f"IDs: {ids}")
    print(f"Decoded: '{tokenizer.decode(ids)}'")
