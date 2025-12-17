# Traffic Scene VLM - ULTRA-DETAILED Tasks 6-10

---

## **PHASE 2: DATA PIPELINE (Continued)**

---

### **Task 6: tokenizer.py - Custom Vocabulary Builder**

#### **6.1 Tokenizer Architecture Design**

**6.1.1 Tokenization Strategy Selection**

```
Decision Matrix:

Option A: Character-Level Tokenization
  Pros: No OOV words, small vocab
  Cons: Long sequences, poor semantics
  Verdict: ❌ Too granular for commands

Option B: Subword Tokenization (BPE/WordPiece)
  Pros: Handles rare words, flexible vocab
  Cons: Complex to implement, harder to interpret
  Verdict: ❌ Overkill for limited domain

Option C: Word-Level Tokenization
  Pros: Simple, interpretable, fast
  Cons: OOV problem for new words
  Verdict: ✅ BEST for traffic commands (controlled vocab)

Choice: Word-Level Tokenization with Special Token Handling
```

**6.1.2 Tokenizer Class Structure**

```
Class: TrafficCommandTokenizer

Attributes:
  - vocab: dict[str, int]           # word → token_id
  - inverse_vocab: dict[int, str]   # token_id → word
  - special_tokens: dict[str, int]  # special token names → ids
  - vocab_size: int                 # total vocabulary size
  - pad_token_id: int               # padding token ID
  - unk_token_id: int               # unknown token ID
  - sos_token_id: int               # start-of-sequence token ID
  - eos_token_id: int               # end-of-sequence token ID
  - max_length: int                 # maximum sequence length (128)

Methods:
  1. __init__(vocab_file=None)
  2. build_vocab(commands: List[str], min_freq=1)
  3. encode(text: str) → List[int]
  4. decode(token_ids: List[int]) → str
  5. batch_encode(texts: List[str]) → dict
  6. batch_decode(token_ids: List[List[int]]) → List[str]
  7. save_vocab(path: str)
  8. load_vocab(path: str)
  9. get_vocab_size() → int
  10. tokenize(text: str) → List[str]  # text → words
  11. convert_tokens_to_ids(tokens: List[str]) → List[int]
  12. convert_ids_to_tokens(ids: List[int]) → List[str]
```

#### **6.2 Vocabulary Construction**

**6.2.1 Special Tokens Definition**

```
Special Token Design:

Token           | ID    | Purpose
----------------|-------|----------------------------------
[PAD]           | 0     | Padding shorter sequences
[UNK]           | 1     | Unknown/out-of-vocabulary words
[SOS]           | 2     | Start of sequence (for generation)
[EOS]           | 3     | End of sequence (for generation)
[YES]           | 4     | Classification answer: YES
[NO]            | 5     | Classification answer: NO
[MAYBE]         | 6     | Classification answer: MAYBE
[SEP]           | 7     | Separator (if multi-sentence)

Reserved IDs: 0-9 (for future special tokens)
Regular vocab starts from ID 10

Why these specific IDs?
- [PAD]=0: PyTorch ignores ID 0 in loss by default
- [UNK]=1: Common convention
- Answer tokens (4-6): Direct mapping to class labels
```

**6.2.2 Vocabulary Building Pipeline**

```
Function: build_vocab_from_commands(commands, config)

Input:
  commands: List[str] - All training commands
  config: VocabConfig

Steps:

Step 1: Text Preprocessing
  for command in commands:
    # Lowercase
    command = command.lower()

    # Remove extra whitespace
    command = ' '.join(command.split())

    # Handle punctuation (keep ? for questions)
    command = command.replace(',', ' , ')
    command = command.replace('.', ' . ')
    command = command.replace('?', ' ? ')

    preprocessed_commands.append(command)

Step 2: Tokenization (Split into Words)
  word_counter = Counter()

  for command in preprocessed_commands:
    words = command.split()  # Simple whitespace split
    word_counter.update(words)

Step 3: Frequency Filtering
  # Remove rare words (min_freq threshold)
  filtered_words = {word: count for word, count in word_counter.items()
                    if count >= config.min_word_frequency}

  # Sort by frequency (most frequent first)
  sorted_words = sorted(filtered_words.items(),
                       key=lambda x: x[1],
                       reverse=True)

Step 4: Vocabulary Assignment
  vocab = {}

  # Add special tokens first (IDs 0-9)
  for token_name, token_id in special_tokens.items():
    vocab[token_name] = token_id

  # Add regular words (IDs from 10)
  next_id = 10
  for word, freq in sorted_words:
    if next_id >= config.max_vocab_size:
      break
    vocab[word] = next_id
    next_id += 1

  # Create inverse mapping
  inverse_vocab = {id: word for word, id in vocab.items()}

Step 5: Vocabulary Statistics
  stats = {
    "total_words_seen": len(word_counter),
    "vocab_size": len(vocab),
    "special_tokens_count": len(special_tokens),
    "most_common_words": word_counter.most_common(20),
    "coverage": calculate_coverage(vocab, word_counter)
  }

  # Coverage = % of tokens in corpus covered by vocab
  # Target: > 99%

Step 6: Save Vocabulary
  vocab_data = {
    "vocab": vocab,
    "special_tokens": special_tokens,
    "stats": stats,
    "config": config.__dict__
  }

  save_json(vocab_data, "data/processed/vocab.json")

Return: vocab, inverse_vocab, stats
```

**6.2.3 Traffic Domain Vocabulary Design**

```
Predefined Core Vocabulary (Mandatory Words):

Traffic Objects (30 words):
  ["car", "vehicle", "pedestrian", "person", "traffic", "light",
   "signal", "sign", "bus", "truck", "motorcycle", "bike", "bicycle",
   "rider", "driver", "lane", "road", "street", "intersection",
   "crosswalk", "crossing", "highway", "obstacle", "barrier"]

Spatial/Direction (20 words):
  ["left", "right", "ahead", "front", "rear", "back", "side",
   "near", "far", "close", "nearby", "opposite", "next",
   "above", "below", "top", "bottom", "center", "middle", "around"]

Actions (20 words):
  ["turn", "go", "stop", "proceed", "move", "drive", "walk",
   "cross", "change", "overtake", "pass", "wait", "yield",
   "merge", "exit", "enter", "approach", "avoid", "follow"]

States/Attributes (25 words):
  ["red", "green", "yellow", "clear", "blocked", "safe", "unsafe",
   "allowed", "forbidden", "visible", "hidden", "fast", "slow",
   "moving", "stopped", "parked", "empty", "occupied", "open",
   "closed", "active", "inactive", "on", "off", "working"]

Question Words (10 words):
  ["is", "are", "can", "could", "should", "will", "would",
   "what", "which", "where"]

Articles/Prepositions/Conjunctions (25 words):
  ["the", "a", "an", "in", "on", "at", "to", "from", "by",
   "with", "without", "of", "for", "and", "or", "if", "when",
   "there", "here", "any", "no", "yes", "not"]

Sentence Structure (10 words):
  ["?", ".", "it", "that", "this", "do", "does", "have", "has"]

Total Core Words: ~140 words
Plus special tokens: 8
Plus generated/rare words: ~150-300

Target Vocabulary Size: 300-500 tokens
```

**6.2.4 Vocabulary Coverage Analysis**

```
Function: calculate_vocab_coverage(vocab, all_commands)

Purpose: Ensure vocabulary covers most command tokens

Algorithm:
  total_tokens = 0
  covered_tokens = 0
  oov_words = Counter()

  for command in all_commands:
    tokens = tokenize(command)
    total_tokens += len(tokens)

    for token in tokens:
      if token in vocab:
        covered_tokens += 1
      else:
        oov_words[token] += 1

  coverage = covered_tokens / total_tokens

  report = {
    "coverage_percentage": coverage * 100,
    "total_tokens": total_tokens,
    "covered_tokens": covered_tokens,
    "oov_tokens": total_tokens - covered_tokens,
    "unique_oov_words": len(oov_words),
    "top_oov_words": oov_words.most_common(20)
  }

  return report

Success Criteria:
  - Coverage > 99% ✓
  - OOV words < 10 unique ✓
  - Top OOV words should be typos/errors

If coverage < 99%:
  - Lower min_frequency threshold
  - Add missing important words manually
  - Check preprocessing (maybe keeping contractions)
```

#### **6.3 Encoding Implementation**

**6.3.1 Single Text Encoding**

```
Function: encode(text: str, add_special_tokens=True,
                 padding=False, max_length=None) → dict

Input: "Can the car safely turn left?"

Step-by-Step Processing:

Step 1: Preprocessing
  text = text.lower().strip()
  → "can the car safely turn left?"

Step 2: Tokenization (word split)
  tokens = self.tokenize(text)
  → ["can", "the", "car", "safely", "turn", "left", "?"]

Step 3: Add Special Tokens (if requested)
  if add_special_tokens:
    tokens = ["[SOS]"] + tokens + ["[EOS]"]
  → ["[SOS]", "can", "the", "car", "safely", "turn", "left", "?", "[EOS]"]

Step 4: Convert to IDs
  token_ids = []
  for token in tokens:
    if token in self.vocab:
      token_ids.append(self.vocab[token])
    else:
      token_ids.append(self.unk_token_id)  # [UNK] token
      logging.warning(f"Unknown token: {token}")

  Example mapping (hypothetical IDs):
  ["[SOS]", "can",  "the", "car", "safely", "turn", "left", "?", "[EOS]"]
  [   2,     45,    12,    23,     87,      56,     34,    140,    3   ]

Step 5: Create Attention Mask
  attention_mask = [1] * len(token_ids)  # 1 = not padding
  → [1, 1, 1, 1, 1, 1, 1, 1, 1]

Step 6: Padding (if requested)
  if padding and max_length:
    if len(token_ids) < max_length:
      # Pad to max_length
      pad_length = max_length - len(token_ids)
      token_ids += [self.pad_token_id] * pad_length
      attention_mask += [0] * pad_length
    elif len(token_ids) > max_length:
      # Truncate
      token_ids = token_ids[:max_length]
      attention_mask = attention_mask[:max_length]

  After padding to max_length=16:
  token_ids =      [2, 45, 12, 23, 87, 56, 34, 140, 3, 0, 0, 0, 0, 0, 0, 0]
  attention_mask = [1,  1,  1,  1,  1,  1,  1,   1, 1, 0, 0, 0, 0, 0, 0, 0]

Step 7: Return Dictionary
  return {
    "input_ids": token_ids,
    "attention_mask": attention_mask,
    "length": original_length  # 9 (before padding)
  }
```

**6.3.2 Batch Encoding**

```
Function: batch_encode(texts: List[str], **kwargs) → dict

Input:
  texts = [
    "Is there a pedestrian?",
    "Can the car turn left?",
    "What color is the traffic light?"
  ]

Step 1: Encode Each Text
  encoded_list = []
  for text in texts:
    encoded = self.encode(text, add_special_tokens=True, padding=False)
    encoded_list.append(encoded)

Step 2: Find Maximum Length
  max_seq_len = max(len(enc["input_ids"]) for enc in encoded_list)

  # Round up to multiple of 8 (GPU efficiency)
  max_seq_len = ((max_seq_len + 7) // 8) * 8

  # Cap at model's max_length
  max_seq_len = min(max_seq_len, self.max_length)

Step 3: Pad All Sequences to Same Length
  batch_input_ids = []
  batch_attention_mask = []
  batch_lengths = []

  for encoded in encoded_list:
    # Pad
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    pad_length = max_seq_len - len(input_ids)

    input_ids += [self.pad_token_id] * pad_length
    attention_mask += [0] * pad_length

    batch_input_ids.append(input_ids)
    batch_attention_mask.append(attention_mask)
    batch_lengths.append(encoded["length"])

Step 4: Convert to Tensors (if PyTorch integration)
  import torch

  batch = {
    "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
    "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
    "lengths": torch.tensor(batch_lengths, dtype=torch.long)
  }

  # Shapes:
  # input_ids: [batch_size, max_seq_len] e.g., [3, 16]
  # attention_mask: [3, 16]
  # lengths: [3]

Return: batch dictionary
```

#### **6.4 Decoding Implementation**

**6.4.1 Single Sequence Decoding**

```
Function: decode(token_ids: List[int], skip_special_tokens=True) → str

Input: [2, 45, 12, 23, 87, 56, 34, 140, 3, 0, 0, 0]

Step 1: Convert IDs to Tokens
  tokens = []
  for token_id in token_ids:
    if token_id in self.inverse_vocab:
      tokens.append(self.inverse_vocab[token_id])
    else:
      tokens.append("[UNK]")

  → ["[SOS]", "can", "the", "car", "safely", "turn", "left", "?",
     "[EOS]", "[PAD]", "[PAD]", "[PAD]"]

Step 2: Remove Special Tokens (if requested)
  if skip_special_tokens:
    special_token_list = ["[SOS]", "[EOS]", "[PAD]", "[SEP]"]
    tokens = [t for t in tokens if t not in special_token_list]

  → ["can", "the", "car", "safely", "turn", "left", "?"]

Step 3: Join Tokens
  text = " ".join(tokens)
  → "can the car safely turn left ?"

Step 4: Post-processing (Optional)
  # Remove space before punctuation
  text = text.replace(" ?", "?")
  text = text.replace(" .", ".")
  text = text.replace(" ,", ",")

  → "can the car safely turn left?"

Step 5: Capitalize First Letter
  text = text[0].upper() + text[1:] if text else ""
  → "Can the car safely turn left?"

Return: text
```

**6.4.2 Batch Decoding**

```
Function: batch_decode(token_ids_batch, skip_special_tokens=True) → List[str]

Input:
  token_ids_batch: torch.Tensor [batch_size, seq_len]
  or List[List[int]]

Step 1: Convert Tensor to List (if needed)
  if isinstance(token_ids_batch, torch.Tensor):
    token_ids_batch = token_ids_batch.cpu().numpy().tolist()

Step 2: Decode Each Sequence
  decoded_texts = []
  for token_ids in token_ids_batch:
    text = self.decode(token_ids, skip_special_tokens=skip_special_tokens)
    decoded_texts.append(text)

Return: decoded_texts
```

#### **6.5 Tokenization Utils**

**6.5.1 Basic Tokenization**

```
Function: tokenize(text: str) → List[str]

Purpose: Split text into word tokens

Implementation:
  # Lowercase
  text = text.lower()

  # Add space around punctuation
  text = re.sub(r'([?.!,])', r' \1 ', text)

  # Multiple spaces to single
  text = re.sub(r'\s+', ' ', text)

  # Split on whitespace
  tokens = text.strip().split()

  return tokens

Example:
  "Is there a pedestrian?"
  → ["is", "there", "a", "pedestrian", "?"]
```

**6.5.2 Advanced Tokenization Options**

```
For future improvements:

Option 1: Handle Contractions
  "can't" → ["can", "'", "t"] or keep as "can't"
  Decision: Keep as single token for simplicity

Option 2: Numbers
  "5 cars" → ["5", "cars"] or ["[NUM]", "cars"]
  Decision: Keep numbers as-is

Option 3: Compound Words
  "traffic light" → single token or two?
  Decision: Keep as two tokens (more flexible)

Option 4: Punctuation Handling
  "left?" vs "left ?"
  Decision: Separate punctuation as token
```

#### **6.6 Vocabulary Persistence**

**6.6.1 Save Vocabulary**

```
Function: save_vocab(path: str)

Format: JSON

Structure:
{
  "vocab_version": "1.0",
  "creation_date": "2025-12-17",
  "vocab_size": 487,
  "special_tokens": {
    "[PAD]": 0,
    "[UNK]": 1,
    "[SOS]": 2,
    "[EOS]": 3,
    "[YES]": 4,
    "[NO]": 5,
    "[MAYBE]": 6,
    "[SEP]": 7
  },
  "vocab": {
    "the": 10,
    "a": 11,
    "car": 12,
    "is": 13,
    ...
  },
  "config": {
    "min_word_frequency": 2,
    "max_vocab_size": 500,
    "lowercase": true
  },
  "statistics": {
    "total_words_seen": 1245,
    "coverage": 99.6,
    "most_common": [["the", 1523], ["is", 1204], ...]
  }
}

Save to: data/processed/vocab.json
```

**6.6.2 Load Vocabulary**

```
Function: load_vocab(path: str)

Steps:
  1. Load JSON file
  2. Validate structure
  3. Reconstruct vocab dict
  4. Reconstruct inverse_vocab dict
  5. Set special token IDs
  6. Validate vocab_size matches

Error Handling:
  - File not found → raise FileNotFoundError
  - Invalid JSON → raise JSONDecodeError
  - Missing keys → raise ValueError
  - Version mismatch → warning

Example:
  tokenizer = TrafficCommandTokenizer()
  tokenizer.load_vocab("data/processed/vocab.json")
  assert tokenizer.vocab_size == 487
```

#### **6.7 Tokenizer Integration with Dataset**

**6.7.1 Dataset Integration Points**

```
Where tokenizer is used:

1. During Command Generation (Task 5):
   - Build vocab from all generated commands
   - Validate all commands are encodable
   - Check coverage

2. During Data Loading (Task 7):
   - Encode commands on-the-fly
   - Cache encoded commands for speed
   - Apply padding per batch

3. During Training (Task 18):
   - Decode predictions for logging
   - Decode for visualization

4. During Inference:
   - Encode input commands
   - Decode model outputs
```

**6.7.2 Caching Encoded Commands**

```
Optimization: Pre-encode all commands

Function: create_encoded_cache(commands, tokenizer, output_path)

Steps:
  1. Encode all commands
  2. Save to disk (pickle or numpy)
  3. Load during training (faster than on-the-fly)

Tradeoff:
  - Memory: Encoded data is larger (int arrays vs strings)
  - Speed: 10x faster data loading
  - Decision: Cache for training, encode on-the-fly for inference

Cache Structure:
{
  "command_text": "Is there a car?",
  "input_ids": [2, 45, 67, 12, 140, 3],
  "attention_mask": [1, 1, 1, 1, 1, 1],
  "length": 6
}

Save as: data/processed/train_encoded_commands.pkl
```

#### **6.8 Testing & Validation**

**6.8.1 Unit Tests**

```
Tests to implement:

1. test_vocab_building():
   commands = ["is there a car?", "can the car turn left?"]
   tokenizer.build_vocab(commands)
   assert tokenizer.vocab_size > len(special_tokens)
   assert "[PAD]" in tokenizer.vocab
   assert "car" in tokenizer.vocab

2. test_encode_decode_consistency():
   text = "Is there a pedestrian?"
   encoded = tokenizer.encode(text)
   decoded = tokenizer.decode(encoded["input_ids"])
   assert decoded.lower() == text.lower()

3. test_batch_encoding_shape():
   texts = ["Is there a car?", "Can the car turn left?"]
   batch = tokenizer.batch_encode(texts, padding=True, max_length=16)
   assert batch["input_ids"].shape == (2, 16)
   assert batch["attention_mask"].shape == (2, 16)

4. test_special_tokens():
   encoded = tokenizer.encode("test", add_special_tokens=True)
   assert encoded["input_ids"][0] == tokenizer.sos_token_id
   assert encoded["input_ids"][-1] == tokenizer.eos_token_id

5. test_unknown_token_handling():
   tokenizer.build_vocab(["is there a car"])
   encoded = tokenizer.encode("xyz unknown word")
   assert tokenizer.unk_token_id in encoded["input_ids"]

6. test_padding():
   encoded = tokenizer.encode("test", padding=True, max_length=10)
   assert len(encoded["input_ids"]) == 10
   assert encoded["input_ids"][-1] == tokenizer.pad_token_id
   assert encoded["attention_mask"][-1] == 0

7. test_save_load_vocab():
   tokenizer1.build_vocab(commands)
   tokenizer1.save_vocab("temp_vocab.json")

   tokenizer2 = TrafficCommandTokenizer()
   tokenizer2.load_vocab("temp_vocab.json")

   assert tokenizer1.vocab == tokenizer2.vocab
   assert tokenizer1.vocab_size == tokenizer2.vocab_size

8. test_coverage_99_percent():
   tokenizer.build_vocab(train_commands)
   coverage = tokenizer.calculate_coverage(all_commands)
   assert coverage > 0.99
```

**6.8.2 Edge Cases to Handle**

```
1. Empty String:
   tokenizer.encode("") → should return [SOS, EOS]

2. Only Punctuation:
   tokenizer.encode("???") → should handle gracefully

3. Very Long Command:
   text = "a" * 1000
   encoded = tokenizer.encode(text, max_length=128)
   assert len(encoded["input_ids"]) == 128  # truncated

4. Special Characters:
   tokenizer.encode("car@#$%") → handle or filter

5. Numbers:
   tokenizer.encode("5 cars") → should tokenize correctly

6. Repeated Words:
   tokenizer.encode("is is is") → should work

7. Unicode Characters:
   tokenizer.encode("café") → normalize or keep
```

#### **6.9 Performance Optimization**

**6.9.1 Encoding Speed**

```
Optimization strategies:

1. Vectorized Operations:
   # Instead of loop:
   for token in tokens:
     ids.append(vocab[token])

   # Use numpy:
   ids = np.array([vocab.get(t, unk_id) for t in tokens])

2. Caching:
   # Cache frequent commands
   encode_cache = {}
   def encode_cached(text):
     if text in encode_cache:
       return encode_cache[text]
     result = encode(text)
     encode_cache[text] = result
     return result

3. Batch Processing:
   # Process all commands at once, not one-by-one
   batch_encode(all_commands) instead of loop

Target Speed:
  - Encode 1000 commands in < 1 second
  - Benchmark on your machine
```

#### **6.10 Documentation & Logging**

**6.10.1 Docstrings**

```
Every function should have:

def encode(self, text: str, add_special_tokens: bool = True) -> dict:
    """
    Encode a single text command into token IDs.

    Args:
        text (str): Input command text
        add_special_tokens (bool): Whether to add [SOS] and [EOS]

    Returns:
        dict: {
            "input_ids": List[int],
            "attention_mask": List[int],
            "length": int
        }

    Example:
        >>> tokenizer.encode("Is there a car?")
        {"input_ids": [2, 45, 67, 12, 140, 3], ...}
    """
```

**6.10.2 Logging**

```
Log important events:

import logging

# Vocabulary building
logging.info(f"Built vocabulary with {vocab_size} tokens")
logging.info(f"Coverage: {coverage:.2%}")
logging.info(f"Top 10 words: {most_common[:10]}")

# Unknown tokens
logging.warning(f"Unknown token encountered: '{token}'")
logging.info(f"Total unknown tokens: {unk_count}")

# Performance
logging.debug(f"Encoded {len(texts)} texts in {elapsed:.3f}s")
```

---

### **Task 7: data_loader.py - PyTorch Dataset & DataLoader**

#### **7.1 Dataset Class Architecture**

**7.1.1 TrafficVLMDataset Design**

```
Class: TrafficVLMDataset(torch.utils.data.Dataset)

Purpose: Load image-command-answer triplets for training

Inheritance: torch.utils.data.Dataset
  - Must implement __len__() and __getitem__()

Attributes:
  - split: str ("train", "val", "test")
  - image_h5_path: str
  - commands_json_path: str
  - tokenizer: TrafficCommandTokenizer
  - transform: torchvision.transforms
  - h5_file: h5py.File (opened in __init__)
  - commands_data: List[dict]
  - config: DatasetConfig

Methods:
  1. __init__(split, config, tokenizer, transform=None)
  2. __len__() → int
  3. __getitem__(idx) → dict
  4. _load_image(idx) → np.ndarray
  5. _load_command(idx) → dict
  6. _apply_transform(image) → torch.Tensor
  7. close() → None
  8. __del__() → None
```

**7.1.2 Data Item Structure**

```
Each __getitem__(idx) returns:

{
  # Image data
  "image": torch.Tensor [3, 224, 224],        # CHW format, float32
  "image_id": str,                             # e.g., "b1c66a42-6f7d68ca"

  # Command data
  "input_ids": torch.Tensor [seq_len],        # Tokenized command
  "attention_mask": torch.Tensor [seq_len],   # 1=real, 0=padding
  "command_text": str,                         # Original text
  "command_length": int,                       # Length before padding

  # Answer data
  "label": torch.Tensor [],                   # Class label (0=NO, 1=YES)
  "answer_text": str,                          # "YES" or "NO"

  # Metadata
  "idx": int,                                  # Dataset index
  "scene_type": str,                           # "city street", etc.
  "num_objects": int,                          # Object count in scene
  "command_type": str                          # "safety", "detection", etc.
}
```

#### **7.2 Dataset Initialization**

**7.2.1 **init** Implementation Details**

```
def __init__(self, split, config, tokenizer, transform=None):

Step 1: Save Attributes
  self.split = split
  self.config = config
  self.tokenizer = tokenizer
  self.transform = transform

Step 2: Construct File Paths
  self.image_h5_path = os.path.join(
    config.processed_root,
    f"{split}_images.h5"
  )
  self.commands_json_path = os.path.join(
    config.processed_root,
    f"{split}_commands.json"
  )

Step 3: Validate Files Exist
  if not os.path.exists(self.image_h5_path):
    raise FileNotFoundError(f"HDF5 file not found: {self.image_h5_path}")
  if not os.path.exists(self.commands_json_path):
    raise FileNotFoundError(f"Commands file not found: {self.commands_json_path}")

Step 4: Open HDF5 File
  # Important: Use 'r' mode (read-only) for multiprocessing safety
  self.h5_file = h5py.File(self.image_h5_path, 'r')

  # Access datasets
  self.images_dataset = self.h5_file['images']
  self.image_ids_dataset = self.h5_file['image_ids']
  self.annotations_dataset = self.h5_file['annotations']

Step 5: Load Commands JSON
  with open(self.commands_json_path, 'r') as f:
    self.commands_data = json.load(f)

  # commands_data structure:
  # [
  #   {
  #     "image_id": "b1c66a42-6f7d68ca",
  #     "command": "Is there a pedestrian?",
  #     "answer": "YES",
  #     "command_type": "detection",
  #     "metadata": {...}
  #   },
  #   ...
  # ]

Step 6: Validate Data Consistency
  num_images = len(self.images_dataset)
  num_commands = len(self.commands_data)

  if num_images != num_commands:
    logging.warning(f"Image count ({num_images}) != command count ({num_commands})")

  # For this project, should be 1:1 mapping
  assert num_images == num_commands

Step 7: Create Label Mapping
  self.label_to_id = {
    "NO": 0,
    "YES": 1,
    "MAYBE": 2  # if using 3-class
  }
  self.id_to_label = {v: k for k, v in self.label_to_id.items()}

Step 8: Cache Normalizer Stats (if needed)
  if hasattr(self.h5_file, 'metadata'):
    self.mean = self.h5_file['metadata/mean'][:]
    self.std = self.h5_file['metadata/std'][:]
  else:
    # Default ImageNet stats
    self.mean = np.array([0.485, 0.456, 0.406])
    self.std = np.array([0.229, 0.224, 0.225])

Step 9: Setup Transforms (if none provided)
  if self.transform is None:
    self.transform = self._get_default_transform()

Step 10: Log Dataset Info
  logging.info(f"Loaded {split} dataset:")
  logging.info(f"  - Images: {num_images}")
  logging.info(f"  - Commands: {num_commands}")
  logging.info(f"  - HDF5 path: {self.image_h5_path}")
```

**7.2.2 Windows/Multiprocessing Considerations**

```
Critical for Windows + A3000:

Issue: HDF5 files don't work well with multiprocessing on Windows

Solution 1: Open HDF5 in worker_init_fn
  def worker_init_fn(worker_id):
    # Each worker opens its own HDF5 file handle
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.h5_file = h5py.File(dataset.image_h5_path, 'r')
    dataset.images_dataset = dataset.h5_file['images']

  # In __init__:
  self.h5_file = None  # Will be opened in worker

Solution 2: Use num_workers=0 (Single process)
  # Simplest for debugging, slower training
  dataloader = DataLoader(dataset, num_workers=0)

Solution 3: Load all images to RAM (if fits)
  # For 5000 images × 224×224×3 × 4 bytes ≈ 2.8GB
  if config.load_to_ram:
    self.images = self.h5_file['images'][:]  # Load all
    self.h5_file.close()

Recommendation for A3000:
  - Use num_workers=2 with worker_init_fn
  - Or num_workers=0 for stability
  - Persistent workers for efficiency
```

#### **7.3 Core Dataset Methods**

**7.3.1 **len** Implementation**

```
def __len__(self) -> int:
    """Return total number of samples in dataset."""
    return len(self.commands_data)

Simple but critical for DataLoader to know dataset size.
```

**7.3.2 **getitem** Implementation**

```
def __getitem__(self, idx: int) -> dict:
    """
    Load and return a single sample.

    Args:
        idx: Sample index (0 to len-1)

    Returns:
        dict: Sample with image, command, label
    """

Step 1: Validate Index
  if idx < 0 or idx >= len(self):
    raise IndexError(f"Index {idx} out of range [0, {len(self)-1}]")

Step 2: Load Image from HDF5
  try:
    image = self.images_dataset[idx]  # [224, 224, 3] uint8
  except Exception as e:
    logging.error(f"Failed to load image at index {idx}: {e}")
    raise

Step 3: Convert Image to Float and Normalize
  image = image.astype(np.float32) / 255.0  # [0, 255] → [0, 1]

  # Apply normalization
  image = (image - self.mean) / self.std

Step 4: Convert to Tensor and Transpose to CHW
  image = torch.from_numpy(image).float()  # [H, W, C]
  image = image.permute(2, 0, 1)            # [C, H, W]

Step 5: Apply Transforms (if any)
  if self.transform is not None:
    image = self.transform(image)

Step 6: Load Command Data
  command_item = self.commands_data[idx]
  command_text = command_item["command"]
  answer_text = command_item["answer"]

Step 7: Encode Command Text
  encoded = self.tokenizer.encode(
    command_text,
    add_special_tokens=True,
    padding=False  # Will pad in collate_fn
  )

  input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
  attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)

Step 8: Convert Answer to Label
  label = self.label_to_id[answer_text]
  label = torch.tensor(label, dtype=torch.long)

Step 9: Gather Metadata
  metadata = command_item.get("metadata", {})

Step 10: Create Sample Dictionary
  sample = {
    # Image
    "image": image,                              # [3, 224, 224]
    "image_id": command_item["image_id"],

    # Command
    "input_ids": input_ids,                      # [seq_len]
    "attention_mask": attention_mask,            # [seq_len]
    "command_text": command_text,
    "command_length": len(input_ids),

    # Label
    "label": label,                              # [] scalar
    "answer_text": answer_text,

    # Metadata
    "idx": idx,
    "command_type": command_item.get("command_type", "unknown"),
    "scene_type": metadata.get("scene_type", "unknown"),
    "num_objects": metadata.get("num_objects", 0)
  }

Step 11: Return Sample
  return sample

Error Handling:
  - Wrap in try-except
  - If loading fails, log and return None or dummy sample
  - DataLoader can skip None samples with custom collate_fn
```

#### **7.4 Data Collation (Batching)**

**7.4.1 Custom Collate Function**

```
def collate_fn(batch: List[dict]) -> dict:
    """
    Collate list of samples into a batch.

    Args:
        batch: List of sample dicts from __getitem__

    Returns:
        dict: Batched tensors
    """

Purpose:
  - Stack images into [B, C, H, W]
  - Pad commands to same length
  - Stack labels
  - Handle variable-length sequences

Step 1: Filter Out None Samples (if any)
  batch = [sample for sample in batch if sample is not None]

  if len(batch) == 0:
    return None  # Empty batch

Step 2: Stack Images
  images = torch.stack([sample["image"] for sample in batch])
  # Shape: [batch_size, 3, 224, 224]

Step 3: Pad Commands to Same Length
  # Find max length in batch
  max_len = max(sample["command_length"] for sample in batch)

  # Round up to multiple of 8 (GPU efficiency)
  max_len = ((max_len + 7) // 8) * 8

  # Pad each sequence
  input_ids_list = []
  attention_mask_list = []

  for sample in batch:
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Pad
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
      input_ids = torch.cat([
        input_ids,
        torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
      ])
      attention_mask = torch.cat([
        attention_mask,
        torch.zeros(pad_len, dtype=torch.long)
      ])

    input_ids_list.append(input_ids)
    attention_mask_list.append(attention_mask)

  # Stack
  input_ids = torch.stack(input_ids_list)          # [B, max_len]
  attention_mask = torch.stack(attention_mask_list)  # [B, max_len]

Step 4: Stack Labels
  labels = torch.stack([sample["label"] for sample in batch])
  # Shape: [batch_size]

Step 5: Collect Metadata (for logging/visualization)
  image_ids = [sample["image_id"] for sample in batch]
  command_texts = [sample["command_text"] for sample in batch]
  answer_texts = [sample["answer_text"] for sample in batch]
  command_types = [sample["command_type"] for sample in batch]

Step 6: Create Batch Dictionary
  batch_dict = {
    # Model inputs
    "images": images,                    # [B, 3, 224, 224]
    "input_ids": input_ids,              # [B, max_len]
    "attention_mask": attention_mask,    # [B, max_len]

    # Targets
    "labels": labels,                    # [B]

    # Metadata (not used by model, for logging)
    "image_ids": image_ids,
    "command_texts": command_texts,
    "answer_texts": answer_texts,
    "command_types": command_types
  }

Step 7: Return Batch
  return batch_dict
```

**7.4.2 Efficient Padding Strategy**

```
Problem: Padding to max sequence length in entire dataset wastes memory

Solution: Dynamic padding per batch

Benefits:
  - Shorter commands → less padding → faster computation
  - Batch 1 might have max_len=12
  - Batch 2 might have max_len=20
  - Saves ~30% memory and compute

Implementation:
  Use custom collate_fn (as shown above)
  Do NOT pad in __getitem__

Tradeoff:
  - Pros: Memory efficient, faster
  - Cons: Batches have different shapes (fine for PyTorch)
```

#### **7.5 Data Augmentation**

**7.5.1 Image Augmentation Pipeline**

```
For Training:

import torchvision.transforms as T

train_transform = T.Compose([
  # Already resized in HDF5, so start from [3, 224, 224]

  # Spatial augmentations
  T.RandomHorizontalFlip(p=0.5),
  T.RandomRotation(degrees=5),

  # Color augmentations
  T.ColorJitter(
    brightness=0.1,
    contrast=0.1,
    saturation=0.1,
    hue=0.05
  ),

  # Normalize (if not done in __getitem__)
  # T.Normalize(mean=[0.485, 0.456, 0.406],
  #             std=[0.229, 0.224, 0.225])
])

For Validation/Test:
  - NO augmentation
  - Only normalization

Note: Apply augmentation in __getitem__ AFTER loading image
```

**7.5.2 Augmentation Strategy**

```
Which augmentations to use for traffic scenes?

✅ Safe:
  - Horizontal flip (road can be mirrored)
  - Color jitter (different lighting conditions)
  - Small rotations (±5°, camera tilt)

❌ Avoid:
  - Vertical flip (cars don't drive upside down)
  - Large rotations (>10°, unrealistic)
  - Random crop (might crop out important objects)
  - Cutout/Erasing (need full scene)

Special Consideration:
  - Horizontal flip changes "left"/"right" labels!
  - If command is "Can the car turn left?", flip makes it wrong
  - Solution: Don't flip for directional commands, OR:
    - Flip image AND swap "left"↔"right" in command
    - Complex, not recommended for first version
  - Recommendation: Disable horizontal flip, or only for non-directional commands
```

#### **7.6 DataLoader Configuration**

**7.6.1 Training DataLoader Setup**

```
from torch.utils.data import DataLoader

train_dataset = TrafficVLMDataset(
  split="train",
  config=dataset_config,
  tokenizer=tokenizer,
  transform=train_transform
)

train_dataloader = DataLoader(
  dataset=train_dataset,
  batch_size=4,                    # A3000: 4
  shuffle=True,                    # Shuffle for training
  num_workers=2,                   # Windows: 0-2
  pin_memory=True,                 # Faster GPU transfer
  drop_last=True,                  # Drop incomplete last batch
  collate_fn=collate_fn,           # Custom padding
  persistent_workers=True,         # Keep workers alive (faster)
  prefetch_factor=2,               # Prefetch 2 batches per worker
  worker_init_fn=worker_init_fn    # For HDF5 multiprocessing
)

Configuration Explanations:

batch_size=4:
  - A3000 has 6GB VRAM
  - Model ~40M params ≈ 160MB
  - Activations per sample ≈ 50MB
  - Total per batch: 160 + 4*50 = 360MB (safe)
  - With mixed precision: can fit batch=8

shuffle=True:
  - Randomize order each epoch
  - Improves generalization

num_workers=2:
  - Windows limitation: 0-2 workers
  - Linux: can use 4-8
  - 0 = single process (debugging)
  - 2 = balanced (speed vs stability)

pin_memory=True:
  - Allocate page-locked memory
  - Faster CPU→GPU transfer
  - Uses more RAM

drop_last=True:
  - If dataset size not divisible by batch_size
  - Drop last incomplete batch
  - Ensures consistent batch size

persistent_workers=True:
  - Don't kill workers between epochs
  - Faster (no worker respawn overhead)
  - Higher memory usage

prefetch_factor=2:
  - Each worker prefetches N batches
  - Reduces GPU wait time
  - Higher = more memory
```

**7.6.2 Validation DataLoader Setup**

```
val_dataset = TrafficVLMDataset(
  split="val",
  config=dataset_config,
  tokenizer=tokenizer,
  transform=val_transform  # No augmentation
)

val_dataloader = DataLoader(
  dataset=val_dataset,
  batch_size=8,              # Can be larger (no backprop)
  shuffle=False,             # Don't shuffle validation
  num_workers=2,
  pin_memory=True,
  drop_last=False,           # Keep all validation samples
  collate_fn=collate_fn,
  persistent_workers=True
)

Key Differences from Training:
  - No shuffle (consistent ordering)
  - Can use larger batch (no gradients)
  - Don't drop last (want all samples)
```

**7.6.3 Worker Init Function (for HDF5)**

```
def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker.
    Opens HDF5 file in each worker process.
    """
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:  # Single-process
        return

    dataset = worker_info.dataset

    # Re-open HDF5 file in this worker
    dataset.h5_file = h5py.File(dataset.image_h5_path, 'r')
    dataset.images_dataset = dataset.h5_file['images']
    dataset.image_ids_dataset = dataset.h5_file['image_ids']
    dataset.annotations_dataset = dataset.h5_file['annotations']

    # Set random seed per worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    logging.debug(f"Worker {worker_id} initialized with seed {worker_seed}")
```

#### **7.7 Memory Optimization**

**7.7.1 Memory-Efficient Loading**

```
Problem: Loading 5000 images (224×224×3) = ~2.8GB RAM

Option 1: Load All to RAM (if fits)
  Pros: Fastest data loading
  Cons: Uses 2.8GB RAM
  When: If you have 16GB+ RAM

Option 2: Load from HDF5 on-the-fly
  Pros: Minimal RAM usage
  Cons: Slower (disk I/O)
  When: Limited RAM

Option 3: Memory-mapped HDF5
  HDF5 supports memory mapping
  OS manages caching
  Balance between speed and memory

Recommendation for A3000 + 16GB RAM:
  - Load validation set to RAM (500 images ≈ 280MB)
  - Load training from HDF5 (4000 images, too large)
  - Use SSD for fast disk reads
```

**7.7.2 Gradient Accumulation Support**

```
When using gradient accumulation:

effective_batch_size = batch_size × accumulation_steps
4 × 4 = 16

DataLoader still yields batch_size=4
Training loop accumulates gradients over 4 batches

No change to DataLoader code
```

#### **7.8 Testing & Validation**

**7.8.1 Dataset Tests**

```
1. test_dataset_length():
   dataset = TrafficVLMDataset("train", config, tokenizer)
   assert len(dataset) == 4000

2. test_getitem_shapes():
   sample = dataset[0]
   assert sample["image"].shape == (3, 224, 224)
   assert sample["input_ids"].ndim == 1
   assert sample["label"].ndim == 0  # scalar

3. test_getitem_types():
   sample = dataset[0]
   assert sample["image"].dtype == torch.float32
   assert sample["input_ids"].dtype == torch.long
   assert sample["label"].dtype == torch.long

4. test_no_data_leakage():
   train_dataset = TrafficVLMDataset("train", ...)
   val_dataset = TrafficVLMDataset("val", ...)

   train_ids = set(train_dataset.commands_data[i]["image_id"]
                   for i in range(len(train_dataset)))
   val_ids = set(val_dataset.commands_data[i]["image_id"]
                 for i in range(len(val_dataset)))

   assert len(train_ids & val_ids) == 0  # No overlap

5. test_collate_fn():
   batch = [dataset[i] for i in range(4)]
   collated = collate_fn(batch)
   assert collated["images"].shape == (4, 3, 224, 224)
   assert collated["labels"].shape == (4,)

6. test_dataloader_iteration():
   dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
   batch = next(iter(dataloader))
   assert batch is not None
   assert "images" in batch
   assert "labels" in batch

7. test_image_normalization():
   sample = dataset[0]
   image = sample["image"]
   # Check roughly normalized (mean ≈ 0, std ≈ 1)
   assert -3 < image.mean() < 3
   assert 0.5 < image.std() < 2

8. test_command_encoding():
   sample = dataset[0]
   # Check [SOS] and [EOS] present
   assert sample["input_ids"][0] == tokenizer.sos_token_id
   assert sample["input_ids"][-1] == tokenizer.eos_token_id or \
          tokenizer.eos_token_id in sample["input_ids"]
```

#### **7.9 Performance Profiling**

**7.9.1 Dataloader Speed Test**

```
import time
from tqdm import tqdm

def profile_dataloader(dataloader, num_batches=100):
    """Measure data loading speed."""

    start = time.time()

    for i, batch in enumerate(tqdm(dataloader)):
        if i >= num_batches:
            break
        # Simulate GPU transfer
        _ = batch["images"].to("cuda", non_blocking=True)

    elapsed = time.time() - start

    samples_per_sec = (num_batches * dataloader.batch_size) / elapsed

    print(f"Data loading speed: {samples_per_sec:.1f} samples/sec")
    print(f"Time per batch: {elapsed/num_batches*1000:.1f} ms")

    return samples_per_sec

Target Performance:
  - >100 samples/sec (good)
  - >200 samples/sec (excellent)
  - If <50 samples/sec: increase num_workers or optimize __getitem__
```

**7.9.2 Bottleneck Identification**

```
Use cProfile:

import cProfile

profiler = cProfile.Profile()
profiler.enable()

for i, batch in enumerate(dataloader):
    if i >= 10:
        break

profiler.disable()
profiler.print_stats(sort='cumulative')

Look for:
  - Slow HDF5 reads
  - Slow tokenization
  - Slow transforms

Optimize accordingly
```

---

## **PHASE 3: VISION ENCODER (Day 3)**

---

### **Task 8: siglip_encoder.py - Main Vision Transformer**

#### **8.1 SigLip Vision Encoder Architecture**

**8.1.1 Architecture Overview**

```
SigLip Vision Encoder (inspired by CLIP/SigLip paper):

Input: [B, 3, 224, 224] RGB images

↓ Patch Embedding
[B, 196, 768] patch tokens (224/16 = 14, 14×14 = 196)

↓ Add Position Embeddings
[B, 196, 768]

↓ Transformer Encoder Layers (N=12)
[B, 196, 768]

↓ Final Layer Norm
[B, 196, 768]

Output: Visual tokens for projection layer

Key Differences from Standard ViT:
  - No [CLS] token (use all patch tokens)
  - Pre-normalization (norm before attention/FFN)
  - Trained with SigLip contrastive loss (sigmoid, not softmax)
  - For this project: Train from scratch OR load pre-trained
```

**8.1.2 Model Configuration**

```
VisionEncoderConfig:
  # Input
  image_size: int = 224
  patch_size: int = 16
  num_channels: int = 3

  # Architecture
  hidden_size: int = 768          # Embedding dimension
  num_hidden_layers: int = 12     # Transformer layers
  num_attention_heads: int = 12   # Heads per layer
  intermediate_size: int = 3072   # FFN intermediate (4x hidden)

  # Activation
  hidden_act: str = "gelu"

  # Regularization
  attention_dropout: float = 0.0
  hidden_dropout: float = 0.0

  # Normalization
  layer_norm_eps: float = 1e-6

  # Initialization
  initializer_range: float = 0.02

Memory Footprint Calculation:
  Patch embeddings: 3 × 16² × 768 = 590K params
  Position embeddings: 196 × 768 = 150K params
  12 layers × (
    Attention: 4 × 768² = 2.36M params/layer
    FFN: 2 × 768 × 3072 = 4.72M params/layer
    Total: ~7M params/layer
  )
  Total Vision Encoder: ~85M params

For A3000 (6GB):
  Too large! Need to reduce.

Optimized Config for A3000:
  hidden_size: 384
  num_hidden_layers: 6
  num_attention_heads: 6
  intermediate_size: 1536
  → Total: ~22M params ✓
```

**8.1.3 Class Structure**

```
Class: SigLipVisionEncoder(nn.Module)

Attributes:
  - config: VisionEncoderConfig
  - embeddings: VisionEmbeddings (Task 9)
  - encoder: VisionTransformerEncoder
  - post_layernorm: nn.LayerNorm

Methods:
  1. __init__(config)
  2. forward(pixel_values, output_attentions=False,
             output_hidden_states=False) → dict
  3. get_input_embeddings() → VisionEmbeddings
  4. _init_weights(module)

Nested Classes:
  - VisionTransformerEncoder
    - List of VisionEncoderLayer
```

#### **8.2 Vision Transformer Encoder Implementation**

**8.2.1 VisionTransformerEncoder**

```
Class: VisionTransformerEncoder(nn.Module)

Purpose: Stack of N transformer encoder layers

Structure:
  self.layers = nn.ModuleList([
    VisionEncoderLayer(config)
    for _ in range(config.num_hidden_layers)
  ])

Forward Pass:
  def forward(self, hidden_states, output_attentions=False):
    """
    Args:
      hidden_states: [B, 196, 768] patch embeddings
      output_attentions: Whether to return attention weights

    Returns:
      hidden_states: [B, 196, 768] final representations
      all_attentions: List of attention tensors (if requested)
    """

    all_attentions = [] if output_attentions else None

    for i, layer in enumerate(self.layers):
      layer_outputs = layer(
        hidden_states,
        output_attentions=output_attentions
      )

      hidden_states = layer_outputs[0]  # Updated hidden states

      if output_attentions:
        all_attentions.append(layer_outputs[1])

    return hidden_states, all_attentions
```

**8.2.2 VisionEncoderLayer**

```
Class: VisionEncoderLayer(nn.Module)

Purpose: Single transformer encoder block

Components:
  1. Layer Norm 1
  2. Self-Attention
  3. Residual Connection 1
  4. Layer Norm 2
  5. Feed-Forward Network
  6. Residual Connection 2

Architecture Choice: Pre-Normalization
  x → LayerNorm → Attention → Add(x, ·)
  Better gradient flow for deep models

Structure:
  def __init__(self, config):
    self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.self_attn = VisionAttention(config)  # Task 10
    self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.mlp = VisionMLP(config)  # Feed-forward

Forward Pass:
  def forward(self, hidden_states, output_attentions=False):
    # Save input for residual
    residual = hidden_states

    # Pre-norm + Self-Attention
    hidden_states = self.layer_norm1(hidden_states)
    attn_output, attn_weights = self.self_attn(
      hidden_states,
      output_attentions=output_attentions
    )

    # Residual connection 1
    hidden_states = residual + attn_output

    # Save for residual 2
    residual = hidden_states

    # Pre-norm + FFN
    hidden_states = self.layer_norm2(hidden_states)
    mlp_output = self.mlp(hidden_states)

    # Residual connection 2
    hidden_states = residual + mlp_output

    return (hidden_states, attn_weights) if output_attentions else (hidden_states,)
```

**8.2.3 Feed-Forward Network (MLP)**

```
Class: VisionMLP(nn.Module)

Purpose: Two-layer FFN with nonlinearity

Structure:
  FC1: hidden_size → intermediate_size (768 → 3072)
  Activation: GELU
  FC2: intermediate_size → hidden_size (3072 → 768)
  Dropout

Implementation:
  def __init__(self, config):
    self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.activation = nn.GELU()
    self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout)

  def forward(self, hidden_states):
    # Shape: [B, 196, 768]

    hidden_states = self.fc1(hidden_states)
    # Shape: [B, 196, 3072]

    hidden_states = self.activation(hidden_states)
    # Shape: [B, 196, 3072]

    hidden_states = self.fc2(hidden_states)
    # Shape: [B, 196, 768]

    hidden_states = self.dropout(hidden_states)
    # Shape: [B, 196, 768]

    return hidden_states

Why 4x expansion?
  - Standard Transformer design
  - More capacity for feature extraction
  - Bottleneck creates compression
```

#### **8.3 Complete Vision Encoder Forward Pass**

**8.3.1 Full Forward Method**

```
def forward(
  self,
  pixel_values: torch.Tensor,
  output_attentions: bool = False,
  output_hidden_states: bool = False
) -> dict:
  """
  Args:
    pixel_values: [B, 3, 224, 224] input images
    output_attentions: Return attention weights from all layers
    output_hidden_states: Return hidden states from all layers

  Returns:
    dict: {
      "last_hidden_state": [B, 196, 768],
      "hidden_states": List[[B, 196, 768]] (if requested),
      "attentions": List[[B, num_heads, 196, 196]] (if requested)
    }
  """

  # Step 1: Patch Embedding + Position Embedding
  hidden_states = self.embeddings(pixel_values)
  # Shape: [B, 196, 768]

  # Step 2: Pass through Transformer encoder layers
  encoder_outputs = self.encoder(
    hidden_states,
    output_attentions=output_attentions
  )

  hidden_states = encoder_outputs[0]  # [B, 196, 768]
  all_attentions = encoder_outputs[1] if output_attentions else None

  # Step 3: Final Layer Normalization
  hidden_states = self.post_layernorm(hidden_states)
  # Shape: [B, 196, 768]

  # Step 4: Prepare output dictionary
  output = {
    "last_hidden_state": hidden_states,
  }

  if output_attentions:
    output["attentions"] = all_attentions

  if output_hidden_states:
    # Would need to collect from each layer
    # For simplicity, just return final
    output["hidden_states"] = [hidden_states]

  return output

Tensor Shapes Summary:
  Input:  [B, 3, 224, 224]
  Patches: [B, 196, 768]
  Layer 1-12: [B, 196, 768]
  Output: [B, 196, 768]
```

#### **8.4 Weight Initialization**

**8.4.1 Initialization Strategy**

```
def _init_weights(self, module):
  """Initialize weights following transformer conventions."""

  if isinstance(module, nn.Linear):
    # Xavier/Glorot uniform initialization
    torch.nn.init.trunc_normal_(
      module.weight,
      mean=0.0,
      std=self.config.initializer_range
    )
    if module.bias is not None:
      torch.nn.init.zeros_(module.bias)

  elif isinstance(module, nn.LayerNorm):
    # Layer norm: gamma=1, beta=0
    torch.nn.init.ones_(module.weight)
    torch.nn.init.zeros_(module.bias)

  elif isinstance(module, nn.Embedding):
    # Position embeddings
    torch.nn.init.trunc_normal_(
      module.weight,
      mean=0.0,
      std=self.config.initializer_range
    )

  elif isinstance(module, nn.Conv2d):
    # Patch embedding convolution
    torch.nn.init.trunc_normal_(
      module.weight,
      mean=0.0,
      std=self.config.initializer_range
    )
    if module.bias is not None:
      torch.nn.init.zeros_(module.bias)

Why truncated normal?
  - Standard for Transformers
  - Avoids extreme values
  - initializer_range=0.02 is typical

Call in __init__:
  self.apply(self._init_weights)
```

#### **8.5 Model Inspection & Utilities**

**8.5.1 Parameter Counting**

```
def count_parameters(model):
  """Count trainable parameters."""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Check vision encoder size
vision_encoder = SigLipVisionEncoder(config)
num_params = count_parameters(vision_encoder)
print(f"Vision Encoder Parameters: {num_params:,}")

Target: <25M for A3000
```

**8.5.2 Model Summary**

```
def print_model_summary(model):
  """Print model architecture summary."""
  print(model)
  print("\nParameter counts per module:")
  for name, module in model.named_children():
    num_params = count_parameters(module)
    print(f"  {name}: {num_params:,}")

Example output:
  embeddings: 740,352
  encoder.layers.0: 7,087,872
  encoder.layers.1: 7,087,872
  ...
  post_layernorm: 1,536
  Total: 22,451,712
```

**8.5.3 Forward Pass Test**

```
def test_vision_encoder():
  """Test forward pass with dummy data."""
  config = VisionEncoderConfig(
    image_size=224,
    hidden_size=384,
    num_hidden_layers=6
  )

  model = SigLipVisionEncoder(config)

  # Dummy input
  batch_size = 2
  pixel_values = torch.randn(batch_size, 3, 224, 224)

  # Forward pass
  outputs = model(pixel_values, output_attentions=True)

  # Check shapes
  assert outputs["last_hidden_state"].shape == (2, 196, 384)
  assert len(outputs["attentions"]) == 6  # num_layers

  print("✓ Vision encoder forward pass successful")

Run this before proceeding to training!
```

#### **8.6 Loading Pre-trained Weights (Optional)**

**8.6.1 Transfer Learning Setup**

```
If using pre-trained SigLip weights:

Option A: Load from Hugging Face
  from transformers import SiglipVisionModel

  pretrained = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")

  # Copy weights to your model
  your_model.load_state_dict(pretrained.state_dict(), strict=False)

Option B: Train from scratch
  - Faster for first iteration
  - Learn architecture before transfer learning
  - Recommendation: Start from scratch for this project

If using pretrained:
  - Freeze encoder during initial training
  - Unfreeze after projection/decoder converge
```

**8.6.2 Freezing/Unfreezing Layers**

```
def freeze_vision_encoder(model):
  """Freeze all vision encoder parameters."""
  for param in model.embeddings.parameters():
    param.requires_grad = False
  for param in model.encoder.parameters():
    param.requires_grad = False
  # post_layernorm can stay trainable

def unfreeze_vision_encoder(model):
  """Unfreeze all vision encoder parameters."""
  for param in model.parameters():
    param.requires_grad = True

Usage:
  # Phase 1: Train only projection + decoder
  freeze_vision_encoder(vlm_model.vision_encoder)
  train(epochs=5)

  # Phase 2: Fine-tune entire model
  unfreeze_vision_encoder(vlm_model.vision_encoder)
  train(epochs=15, lower_lr=True)
```

---

### **Task 9: vision_embeddings.py - Patch Embedding Layer**

#### **9.1 Vision Embeddings Overview**

**9.1.1 Purpose**

```
Convert input images to patch embeddings:
  [B, 3, 224, 224] → [B, num_patches, hidden_size]

For 224×224 image with 16×16 patches:
  num_patches = (224/16) × (224/16) = 14 × 14 = 196

Each 16×16×3 patch becomes a single embedding vector
```

**9.1.2 Implementation Approaches**

```
Approach 1: Reshape + Linear
  - Flatten each patch to vector
  - Project with linear layer
  - Simple but less efficient

Approach 2: Convolution (Standard ViT)
  - Use Conv2d with kernel=patch_size, stride=patch_size
  - One operation, more efficient
  - **This is what we'll use**

Approach 3: Overlapping Patches
  - stride < patch_size
  - More patches, more computation
  - Not used in standard ViT
```

#### **9.2 Patch Embedding with Convolution**

**9.2.1 PatchEmbedding Class**

```
Class: PatchEmbedding(nn.Module)

Purpose: Convert image to patch tokens using convolution

Attributes:
  - image_size: int (224)
  - patch_size: int (16)
  - num_channels: int (3)
  - hidden_size: int (768 or 384)
  - num_patches: int (196)
  - projection: nn.Conv2d

Methods:
  - __init__(config)
  - forward(pixel_values) → patch_embeds

Implementation:
  def __init__(self, config):
    super().__init__()

    self.image_size = config.image_size
    self.patch_size = config.patch_size
    self.num_channels = config.num_channels
    self.hidden_size = config.hidden_size

    # Calculate number of patches
    self.num_patches = (self.image_size // self.patch_size) ** 2

    # Patch projection using convolution
    self.projection = nn.Conv2d(
      in_channels=self.num_channels,      # 3 (RGB)
      out_channels=self.hidden_size,      # 768
      kernel_size=self.patch_size,        # 16
      stride=self.patch_size,             # 16 (non-overlapping)
      padding=0,
      bias=True
    )

  def forward(self, pixel_values):
    """
    Args:
      pixel_values: [B, 3, 224, 224]

    Returns:
      patch_embeds: [B, 196, 768]
    """
    batch_size = pixel_values.shape[0]

    # Apply convolution
    # Input: [B, 3, 224, 224]
    # Output: [B, 768, 14, 14]  (224/16 = 14)
    embeddings = self.projection(pixel_values)

    # Flatten spatial dimensions
    # [B, 768, 14, 14] → [B, 768, 196]
    embeddings = embeddings.flatten(2)

    # Transpose to [B, 196, 768]
    embeddings = embeddings.transpose(1, 2)

    return embeddings

Tensor Shape Flow:
  [B, 3, 224, 224]         Input image
       ↓ Conv2d(kernel=16, stride=16)
  [B, 768, 14, 14]         Feature map
       ↓ flatten(2)
  [B, 768, 196]            Flattened (196 = 14×14)
       ↓ transpose(1,2)
  [B, 196, 768]            Patch embeddings
```

**9.2.2 Why Convolution Works**

```
Convolution with kernel=patch_size, stride=patch_size is equivalent to:

1. Divide image into non-overlapping patches
2. Flatten each patch
3. Apply same linear transformation to each

But convolution is:
  - Faster (GPU-optimized)
  - More memory efficient
  - Easier to implement

Mathematical Equivalence:
  patch = image[i:i+16, j:j+16]  # [16, 16, 3]
  flat_patch = patch.reshape(-1)   # [768]
  embedding = W @ flat_patch + b   # [hidden_size]

  ≡

  embedding = Conv2d(image)[i//16, j//16]
```

#### **9.3 Position Embeddings**

**9.3.1 Why Position Embeddings?**

```
Problem:
  - Self-attention is permutation-invariant
  - Patch order doesn't matter without position info
  - Need to encode spatial location

Solution:
  Add learned position embeddings to patch embeddings

Types:
  1. Learned 1D positions (ViT original)
  2. Learned 2D positions
  3. Sinusoidal positions (Transformer original)
  4. Rotary positions (RoPE - for decoder, not vision)

For Vision Encoder: Use Learned 1D (simplest, works well)
```

**9.3.2 PositionEmbedding Class**

```
Class: PositionEmbedding(nn.Module)

Purpose: Add learnable position information to patches

Implementation:
  def __init__(self, config):
    super().__init__()

    self.num_patches = (config.image_size // config.patch_size) ** 2
    self.hidden_size = config.hidden_size

    # Learnable position embeddings
    # One embedding vector per patch position
    self.position_embeddings = nn.Parameter(
      torch.zeros(1, self.num_patches, self.hidden_size)
    )

    # Initialize with truncated normal
    nn.init.trunc_normal_(self.position_embeddings, std=0.02)

  def forward(self, patch_embeds):
    """
    Args:
      patch_embeds: [B, 196, 768]

    Returns:
      embeddings: [B, 196, 768]
    """
    # Add position embeddings (broadcasted across batch)
    embeddings = patch_embeds + self.position_embeddings

    return embeddings

Shape:
  patch_embeds:          [B, 196, 768]
  position_embeddings:   [1, 196, 768]  (broadcasted)
  result:                [B, 196, 768]
```

# Traffic Scene VLM - ULTRA-DETAILED Tasks 9-10 (Continued)

---

### **Task 9: vision_embeddings.py - Patch Embedding Layer (Continued)**

#### **9.3.3 2D Position Embeddings (Alternative) - Continued**

```
Class: PositionEmbedding2D(nn.Module):
  def __init__(self, config):
    self.grid_size = config.image_size // config.patch_size  # 14

    # Separate embeddings for row and column
    self.row_embeddings = nn.Parameter(
      torch.zeros(1, self.grid_size, 1, config.hidden_size // 2)
    )
    self.col_embeddings = nn.Parameter(
      torch.zeros(1, 1, self.grid_size, config.hidden_size // 2)
    )

    # Initialize
    nn.init.trunc_normal_(self.row_embeddings, std=0.02)
    nn.init.trunc_normal_(self.col_embeddings, std=0.02)

  def forward(self, patch_embeds):
    # patch_embeds: [B, 196, 768]
    B = patch_embeds.shape[0]

    # Reshape to 2D grid
    embeddings = patch_embeds.view(B, self.grid_size, self.grid_size, -1)
    # [B, 14, 14, 768]

    # Add row embeddings (broadcasted across columns)
    # row_embeddings: [1, 14, 1, 384] → broadcasts to [B, 14, 14, 384]
    # Add col embeddings (broadcasted across rows)
    # col_embeddings: [1, 1, 14, 384] → broadcasts to [B, 14, 14, 384]

    row_emb = self.row_embeddings.expand(B, -1, self.grid_size, -1)
    col_emb = self.col_embeddings.expand(B, self.grid_size, -1, -1)

    # Concatenate row and col embeddings
    pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # [B, 14, 14, 768]

    # Add to patch embeddings
    embeddings = embeddings + pos_emb

    # Flatten back to sequence
    embeddings = embeddings.view(B, self.grid_size * self.grid_size, -1)
    # [B, 196, 768]

    return embeddings

Advantage of 2D:
  - Encodes spatial structure explicitly
  - Can generalize to different resolutions
  - More parameters but more expressive

For this project: Use 1D (simpler, standard ViT)
```

#### **9.4 Complete VisionEmbeddings Module**

**9.4.1 VisionEmbeddings Class**

```
Class: VisionEmbeddings(nn.Module)

Purpose: Combine patch embedding + position embedding

Structure:
  - PatchEmbedding: Conv2d for patches
  - PositionEmbedding: Learned positions
  - Dropout (optional)

Implementation:
  class VisionEmbeddings(nn.Module):
    def __init__(self, config):
      super().__init__()

      self.config = config

      # Patch embedding
      self.patch_embedding = PatchEmbedding(config)

      # Position embedding
      self.position_embedding = PositionEmbedding(config)

      # Dropout (optional, usually 0.0 for vision)
      self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, pixel_values):
      """
      Args:
        pixel_values: [B, 3, 224, 224]

      Returns:
        embeddings: [B, 196, 768]
      """
      # Step 1: Convert image to patches
      patch_embeds = self.patch_embedding(pixel_values)
      # [B, 196, 768]

      # Step 2: Add position embeddings
      embeddings = self.position_embedding(patch_embeds)
      # [B, 196, 768]

      # Step 3: Apply dropout (usually not needed in vision)
      embeddings = self.dropout(embeddings)
      # [B, 196, 768]

      return embeddings
```

**9.4.2 Full Forward Pass Example**

```
Example walkthrough with actual tensors:

Input Image:
  pixel_values = torch.randn(2, 3, 224, 224)
  # Batch of 2 RGB images, 224x224

Step 1: Patch Embedding
  projection = nn.Conv2d(3, 768, kernel_size=16, stride=16)
  output = projection(pixel_values)
  # Shape: [2, 768, 14, 14]

  # Flatten spatial dims
  output = output.flatten(2)  # [2, 768, 196]
  output = output.transpose(1, 2)  # [2, 196, 768]

Step 2: Position Embedding
  position_embeddings = nn.Parameter(torch.randn(1, 196, 768))
  output = output + position_embeddings
  # Shape: [2, 196, 768]

Step 3: Dropout (p=0.0 typically)
  output = dropout(output)
  # Shape: [2, 196, 768]

Final Output:
  Each of 196 patches is now represented by a 768-dim vector
  Contains both content (from convolution) and position info
```

#### **9.5 Alternative: Including CLS Token**

**9.5.1 CLS Token Implementation (Optional)**

```
Standard ViT includes a [CLS] token:
  - Prepended to patch sequence
  - Used for classification
  - Final representation of entire image

For PaliGemma style:
  - NO [CLS] token
  - Use all patch tokens
  - Reason: Want fine-grained spatial info for cross-attention

If you wanted to add CLS token:

class VisionEmbeddingsWithCLS(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.patch_embedding = PatchEmbedding(config)

    # CLS token (learnable)
    self.cls_token = nn.Parameter(
      torch.zeros(1, 1, config.hidden_size)
    )

    # Position embedding for patches + CLS
    num_patches = (config.image_size // config.patch_size) ** 2
    self.position_embedding = nn.Parameter(
      torch.zeros(1, num_patches + 1, config.hidden_size)
    )

    self.dropout = nn.Dropout(config.hidden_dropout)

    # Initialize
    nn.init.trunc_normal_(self.cls_token, std=0.02)
    nn.init.trunc_normal_(self.position_embedding, std=0.02)

  def forward(self, pixel_values):
    batch_size = pixel_values.shape[0]

    # Patch embeddings
    patch_embeds = self.patch_embedding(pixel_values)
    # [B, 196, 768]

    # Expand CLS token for batch
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    # [B, 1, 768]

    # Prepend CLS token
    embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)
    # [B, 197, 768]  (1 CLS + 196 patches)

    # Add position embeddings
    embeddings = embeddings + self.position_embedding

    # Dropout
    embeddings = self.dropout(embeddings)

    return embeddings

For this project: DON'T use CLS token
  - Follow PaliGemma design
  - All patches attend to all patches
  - Cross-attention uses all visual tokens
```

#### **9.6 Interpolation for Different Image Sizes**

**9.6.1 Handling Variable Resolutions**

```
Problem:
  Position embeddings are fixed for specific resolution
  If trained on 224x224, can't use 384x384 without adjustment

Solution: Interpolate position embeddings

Function: interpolate_pos_encoding(pos_embed, grid_size)

Implementation:
  def interpolate_pos_encoding(self, pos_embed, h, w):
    """
    Interpolate position embeddings for different resolutions.

    Args:
      pos_embed: [1, num_patches, hidden_size]
      h, w: New grid height and width

    Returns:
      new_pos_embed: [1, h*w, hidden_size]
    """
    npatch = pos_embed.shape[1]
    N = h * w

    if npatch == N:
      return pos_embed

    # Get original grid size
    orig_size = int(npatch ** 0.5)

    # Reshape to 2D
    pos_embed = pos_embed.reshape(1, orig_size, orig_size, -1)
    pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, hidden, H, W]

    # Interpolate using bilinear
    pos_embed = F.interpolate(
      pos_embed,
      size=(h, w),
      mode='bilinear',
      align_corners=False
    )

    # Reshape back
    pos_embed = pos_embed.permute(0, 2, 3, 1)  # [1, h, w, hidden]
    pos_embed = pos_embed.reshape(1, N, -1)

    return pos_embed

When to use:
  - Fine-tuning on different resolution
  - Inference on higher resolution
  - For this project: Not needed (fixed 224x224)
```

#### **9.7 Testing & Validation**

**9.7.1 Unit Tests for Vision Embeddings**

```
1. test_patch_embedding_shape():
   config = VisionEncoderConfig(image_size=224, patch_size=16, hidden_size=768)
   patch_emb = PatchEmbedding(config)

   input = torch.randn(2, 3, 224, 224)
   output = patch_emb(input)

   assert output.shape == (2, 196, 768)

2. test_position_embedding_shape():
   pos_emb = PositionEmbedding(config)

   patch_embeds = torch.randn(2, 196, 768)
   output = pos_emb(patch_embeds)

   assert output.shape == (2, 196, 768)

3. test_position_embedding_adds_variation():
   pos_emb = PositionEmbedding(config)

   # Same input patches
   patch_embeds = torch.ones(1, 196, 768)
   output = pos_emb(patch_embeds)

   # Should be different due to position info
   assert not torch.allclose(output[0, 0], output[0, 1])

4. test_vision_embeddings_full():
   embeddings_module = VisionEmbeddings(config)

   pixel_values = torch.randn(4, 3, 224, 224)
   output = embeddings_module(pixel_values)

   assert output.shape == (4, 196, 768)

5. test_different_patch_sizes():
   # 8x8 patches
   config8 = VisionEncoderConfig(patch_size=8)
   emb8 = VisionEmbeddings(config8)
   out8 = emb8(torch.randn(1, 3, 224, 224))
   assert out8.shape == (1, 784, 768)  # (224/8)^2 = 784

   # 32x32 patches
   config32 = VisionEncoderConfig(patch_size=32)
   emb32 = VisionEmbeddings(config32)
   out32 = emb32(torch.randn(1, 3, 224, 224))
   assert out32.shape == (1, 49, 768)  # (224/32)^2 = 49

6. test_batch_independence():
   embeddings_module = VisionEmbeddings(config)

   img1 = torch.randn(1, 3, 224, 224)
   img2 = torch.randn(1, 3, 224, 224)

   out1 = embeddings_module(img1)
   out2 = embeddings_module(img2)
   out_batch = embeddings_module(torch.cat([img1, img2]))

   assert torch.allclose(out_batch[0], out1[0])
   assert torch.allclose(out_batch[1], out2[0])
```

**9.7.2 Visual Inspection Tests**

```
def visualize_patch_embeddings():
  """Visualize how image is split into patches."""
  import matplotlib.pyplot as plt

  # Load sample image
  image = torch.randn(1, 3, 224, 224)

  # Get patch embeddings
  patch_emb = PatchEmbedding(config)
  patches = patch_emb(image)  # [1, 196, 768]

  # Visualize first few patch embeddings as heatmaps
  fig, axes = plt.subplots(4, 4, figsize=(10, 10))

  for i, ax in enumerate(axes.flat):
    if i < 16:
      # Take first 64 dims of each patch embedding
      patch_vec = patches[0, i, :64].detach().numpy()
      patch_map = patch_vec.reshape(8, 8)

      ax.imshow(patch_map, cmap='viridis')
      ax.set_title(f'Patch {i}')
      ax.axis('off')

  plt.tight_layout()
  plt.savefig('outputs/visualizations/patch_embeddings.png')

def visualize_position_embeddings():
  """Visualize learned position embeddings."""
  import matplotlib.pyplot as plt

  pos_emb = PositionEmbedding(config)

  # Get position embeddings [1, 196, 768]
  pos = pos_emb.position_embeddings[0].detach().numpy()

  # Reshape to 2D grid
  grid_size = 14

  # Visualize different dimensions
  fig, axes = plt.subplots(2, 3, figsize=(12, 8))

  for idx, ax in enumerate(axes.flat):
    dim_idx = idx * 128  # Sample different dimensions

    pos_map = pos[:, dim_idx].reshape(grid_size, grid_size)

    im = ax.imshow(pos_map, cmap='RdBu')
    ax.set_title(f'Dimension {dim_idx}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

  plt.suptitle('Position Embeddings Visualization')
  plt.tight_layout()
  plt.savefig('outputs/visualizations/position_embeddings.png')

Run these to verify embeddings are working correctly!
```

#### **9.8 Memory & Performance Considerations**

**9.8.1 Memory Footprint**

```
PatchEmbedding (Conv2d):
  Parameters: in_channels × out_channels × kernel_size²
  = 3 × 768 × 16² = 589,824 params
  Memory: 589,824 × 4 bytes (FP32) = 2.36 MB

PositionEmbedding:
  Parameters: num_patches × hidden_size
  = 196 × 768 = 150,528 params
  Memory: 150,528 × 4 bytes = 0.60 MB

Total Vision Embeddings: ~3 MB (negligible)

Activations (per sample):
  Input: 3 × 224 × 224 = 150,528 values
  Patch embeds: 196 × 768 = 150,528 values
  With batch=4: 4 × 150,528 × 4 bytes = 2.4 MB

Conclusion: Vision embeddings are very lightweight
```

**9.8.2 Computational Cost**

```
Patch Embedding Convolution:
  FLOPs = batch_size × out_h × out_w × in_channels × kernel_h × kernel_w × out_channels

  = 1 × 14 × 14 × 3 × 16 × 16 × 768
  = 1 × 196 × 3 × 256 × 768
  ≈ 115 MFLOPs per image

  Very fast on GPU: < 1ms

Position Embedding Addition:
  Element-wise addition: 196 × 768 = 150,528 ops
  Negligible

Total: Vision embeddings are extremely fast
```

#### **9.9 Advanced: Sinusoidal Position Embeddings**

**9.9.1 Sinusoidal Implementation (Alternative)**

```
Used in original Transformer paper
Not learned, deterministic

class SinusoidalPositionEmbedding(nn.Module):
  def __init__(self, config):
    super().__init__()

    num_patches = (config.image_size // config.patch_size) ** 2
    hidden_size = config.hidden_size

    # Create position encoding matrix
    position = torch.arange(num_patches).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, hidden_size, 2) *
      (-math.log(10000.0) / hidden_size)
    )

    pos_enc = torch.zeros(1, num_patches, hidden_size)
    pos_enc[0, :, 0::2] = torch.sin(position * div_term)
    pos_enc[0, :, 1::2] = torch.cos(position * div_term)

    # Register as buffer (not trained)
    self.register_buffer('pos_enc', pos_enc)

  def forward(self, patch_embeds):
    return patch_embeds + self.pos_enc

Advantage:
  - No parameters to learn
  - Can extrapolate to longer sequences

Disadvantage:
  - Usually performs worse than learned
  - Less flexibility

For vision: Learned embeddings are standard
```

---

### **Task 10: vision_attention.py - Multi-Head Self-Attention for Vision**

#### **10.1 Multi-Head Attention Overview**

**10.1.1 Attention Mechanism Recap**

```
Self-Attention computes how each patch attends to all other patches

Formula:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Where:
  Q (Query): "What am I looking for?"
  K (Key): "What information do I have?"
  V (Value): "What information do I provide?"

Multi-Head Attention:
  - Split Q, K, V into H heads
  - Each head learns different patterns
  - Concatenate results

Example:
  8 heads might learn to attend to:
    - Head 1: Nearby patches (local patterns)
    - Head 2: Distant patches (global context)
    - Head 3: Vertical alignment
    - Head 4: Horizontal alignment
    - Head 5-8: Other patterns
```

**10.1.2 Vision vs Language Attention Differences**

```
Vision Self-Attention:
  - NO causal mask (all patches can attend to all)
  - Symmetric attention matrix [196 x 196]
  - Bi-directional
  - Example: Patch 50 can attend to patch 100 AND 100 to 50

Language Self-Attention (decoder):
  - Causal mask (can't see future tokens)
  - Triangular attention matrix
  - Uni-directional in decoder

For vision encoder: Use full attention (no mask)
```

#### **10.2 VisionAttention Implementation**

**10.2.1 VisionAttention Class Structure**

```
Class: VisionAttention(nn.Module)

Purpose: Multi-head self-attention for vision patches

Attributes:
  - hidden_size: int (768)
  - num_attention_heads: int (12)
  - attention_head_size: int (hidden_size // num_heads = 64)
  - all_head_size: int (num_heads × head_size = 768)
  - query: nn.Linear
  - key: nn.Linear
  - value: nn.Linear
  - dropout: nn.Dropout
  - output: nn.Linear

Methods:
  1. __init__(config)
  2. transpose_for_scores(x) → reshaped tensor
  3. forward(hidden_states, output_attentions) → (output, attn_weights)
```

**10.2.2 Initialization**

```
def __init__(self, config):
  super().__init__()

  self.hidden_size = config.hidden_size  # 768
  self.num_attention_heads = config.num_attention_heads  # 12

  # Each head processes a smaller dimension
  self.attention_head_size = self.hidden_size // self.num_attention_heads
  # 768 // 12 = 64

  self.all_head_size = self.num_attention_heads * self.attention_head_size
  # Should equal hidden_size (768)

  # Verify divisibility
  if self.hidden_size % self.num_attention_heads != 0:
    raise ValueError(
      f"hidden_size ({self.hidden_size}) must be divisible by "
      f"num_attention_heads ({self.num_attention_heads})"
    )

  # Query, Key, Value projections
  self.query = nn.Linear(self.hidden_size, self.all_head_size)
  self.key = nn.Linear(self.hidden_size, self.all_head_size)
  self.value = nn.Linear(self.hidden_size, self.all_head_size)

  # Attention dropout
  self.dropout = nn.Dropout(config.attention_dropout)

  # Output projection
  self.output = nn.Linear(self.all_head_size, self.hidden_size)
  self.output_dropout = nn.Dropout(config.hidden_dropout)

Parameter Count (for 768 dim, 12 heads):
  Query: 768 × 768 = 589,824 params
  Key: 768 × 768 = 589,824 params
  Value: 768 × 768 = 589,824 params
  Output: 768 × 768 = 589,824 params
  Total: ~2.36M params per attention layer
```

**10.2.3 Transpose for Scores Helper**

```
def transpose_for_scores(self, x):
  """
  Reshape for multi-head attention.

  Args:
    x: [B, num_patches, all_head_size]
       e.g., [2, 196, 768]

  Returns:
    x: [B, num_heads, num_patches, head_size]
       e.g., [2, 12, 196, 64]
  """
  batch_size = x.size(0)
  num_patches = x.size(1)

  # Reshape to [B, num_patches, num_heads, head_size]
  new_shape = (
    batch_size,
    num_patches,
    self.num_attention_heads,
    self.attention_head_size
  )
  x = x.view(*new_shape)

  # Transpose to [B, num_heads, num_patches, head_size]
  x = x.permute(0, 2, 1, 3)

  return x

Why this reshaping?
  - Allows parallel computation across heads
  - Each head operates on 64-dim subspace
  - GPU efficiently processes [B, H, N, d_k] tensors
```

**10.2.4 Forward Pass - Step by Step**

```
def forward(self, hidden_states, output_attentions=False):
  """
  Args:
    hidden_states: [B, 196, 768] - Patch embeddings
    output_attentions: bool - Return attention weights

  Returns:
    attention_output: [B, 196, 768]
    attention_probs: [B, 12, 196, 196] (if output_attentions)
  """

  batch_size = hidden_states.size(0)

  # Step 1: Linear projections for Q, K, V
  # Each projection: [B, 196, 768] → [B, 196, 768]
  query_layer = self.query(hidden_states)
  key_layer = self.key(hidden_states)
  value_layer = self.value(hidden_states)

  # Step 2: Reshape for multi-head attention
  # [B, 196, 768] → [B, 12, 196, 64]
  query_layer = self.transpose_for_scores(query_layer)
  key_layer = self.transpose_for_scores(key_layer)
  value_layer = self.transpose_for_scores(value_layer)

  # Step 3: Compute attention scores
  # Q @ K^T: [B, 12, 196, 64] @ [B, 12, 64, 196] = [B, 12, 196, 196]
  attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

  # Step 4: Scale by sqrt(d_k)
  # Prevents large values that cause vanishing gradients in softmax
  attention_scores = attention_scores / math.sqrt(self.attention_head_size)

  # Step 5: Apply softmax to get attention probabilities
  # Softmax over last dimension (keys)
  # Each query attends to all keys with probs summing to 1
  attention_probs = nn.functional.softmax(attention_scores, dim=-1)
  # Shape: [B, 12, 196, 196]

  # Interpretation of attention_probs[0, 0, i, j]:
  #   How much patch i attends to patch j in head 0

  # Step 6: Apply dropout to attention probabilities
  attention_probs = self.dropout(attention_probs)

  # Step 7: Apply attention to values
  # [B, 12, 196, 196] @ [B, 12, 196, 64] = [B, 12, 196, 64]
  context_layer = torch.matmul(attention_probs, value_layer)

  # Step 8: Reshape back to concatenate heads
  # [B, 12, 196, 64] → [B, 196, 12, 64]
  context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

  # Concatenate heads: [B, 196, 12, 64] → [B, 196, 768]
  new_shape = (batch_size, -1, self.all_head_size)
  context_layer = context_layer.view(*new_shape)

  # Step 9: Final linear projection
  attention_output = self.output(context_layer)

  # Step 10: Output dropout
  attention_output = self.output_dropout(attention_output)

  # Step 11: Return
  outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

  return outputs

Tensor Shapes Summary:
  Input:           [B, 196, 768]
  Q, K, V proj:    [B, 196, 768]
  Q, K, V reshaped:[B, 12, 196, 64]
  Attn scores:     [B, 12, 196, 196]
  Attn probs:      [B, 12, 196, 196]
  Context:         [B, 12, 196, 64]
  Output:          [B, 196, 768]
```

#### **10.3 Detailed Attention Computation Example**

**10.3.1 Numerical Example**

```
Let's trace through with small numbers:

Assume:
  - Batch size = 1
  - 4 patches (instead of 196 for clarity)
  - Hidden size = 8
  - 2 attention heads
  - Head size = 4

Input:
  hidden_states = [1, 4, 8]

Step 1: Project to Q, K, V
  Q = W_q @ hidden_states  # [1, 4, 8]
  K = W_k @ hidden_states  # [1, 4, 8]
  V = W_v @ hidden_states  # [1, 4, 8]

Step 2: Reshape for heads
  Q = [1, 2, 4, 4]  # [B, heads, patches, head_dim]
  K = [1, 2, 4, 4]
  V = [1, 2, 4, 4]

Step 3: Attention scores for head 0
  Q[0, 0] = [[q00, q01, q02, q03],  # Patch 0 query
             [q10, q11, q12, q13],  # Patch 1 query
             [q20, q21, q22, q23],  # Patch 2 query
             [q30, q31, q32, q33]]  # Patch 3 query

  K[0, 0]^T = [[k00, k10, k20, k30],  # Keys transposed
               [k01, k11, k21, k31],
               [k02, k12, k22, k32],
               [k03, k13, k23, k33]]

  Scores = Q @ K^T  # [4, 4]

  scores[i, j] = dot(query_i, key_j)

  Example scores matrix:
  [[15.2,  3.4,  -1.2,  5.6],   # How patch 0 scores all keys
   [2.1,   18.4,  6.3,  -0.9],  # How patch 1 scores all keys
   [-0.5,  7.2,   12.8,  4.1],  # How patch 2 scores all keys
   [4.3,  -2.1,   3.6,   16.9]] # How patch 3 scores all keys

Step 4: Scale
  scores = scores / sqrt(4) = scores / 2

Step 5: Softmax (row-wise)
  For patch 0: softmax([15.2, 3.4, -1.2, 5.6])
  → [0.92, 0.05, 0.00, 0.03]

  Interpretation: Patch 0 mostly attends to itself (0.92)
                  and slightly to patch 3 (0.03)

  Full attention matrix (after softmax):
  [[0.92, 0.05, 0.00, 0.03],
   [0.02, 0.94, 0.03, 0.01],
   [0.01, 0.15, 0.82, 0.02],
   [0.15, 0.00, 0.02, 0.83]]

Step 6: Apply to values
  V[0, 0] = [[v00, v01, v02, v03],
             [v10, v11, v12, v13],
             [v20, v21, v22, v23],
             [v30, v31, v32, v33]]

  Output for patch 0:
  = 0.92 * [v00, v01, v02, v03]
  + 0.05 * [v10, v11, v12, v13]
  + 0.00 * [v20, v21, v22, v23]
  + 0.03 * [v30, v31, v32, v33]

  = Weighted average of all value vectors

This happens for all heads in parallel!
```

**10.3.2 Attention Pattern Interpretation**

```
After training, attention patterns emerge:

Pattern 1: Local Attention
  Patch at position (i, j) attends to neighbors
  Example: Patch 50 attends to [40, 41, 49, 50, 51, 59, 60]
  Use: Capture local textures (edges, corners)

Pattern 2: Global Attention
  Patch attends to distant patches
  Example: Patch 10 attends to patch 180 (opposite corner)
  Use: Capture global context (scene layout)

Pattern 3: Row/Column Attention
  Patch attends to same row or column
  Example: Patch (2, 5) attends to all (2, *) or (*, 5)
  Use: Capture horizontal/vertical structures (lanes, buildings)

Pattern 4: Semantic Attention
  Patches attend based on content, not position
  Example: All "car" patches attend to each other
  Use: Group similar objects

Different heads learn different patterns!
```

#### **10.4 Attention Mask Support (Optional for Vision)**

**10.4.1 Adding Mask Parameter**

```
def forward(self, hidden_states, attention_mask=None, output_attentions=False):
  """
  Args:
    attention_mask: [B, 1, 1, num_patches] or None
                    0 = attend, -inf = don't attend
  """

  # ... (Q, K, V computation as before)

  # Compute attention scores
  attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
  attention_scores = attention_scores / math.sqrt(self.attention_head_size)

  # Apply attention mask (if provided)
  if attention_mask is not None:
    # attention_mask: [B, 1, 1, num_patches]
    # Expand to [B, num_heads, num_patches, num_patches]
    attention_scores = attention_scores + attention_mask

    # Where mask is -inf, softmax will be 0

  # Softmax
  attention_probs = nn.functional.softmax(attention_scores, dim=-1)

  # ... (rest of forward pass)

When to use mask in vision?
  - Padding (if images have different sizes)
  - Ignoring certain patches
  - For this project: NOT NEEDED (fixed size 224x224)
```

#### **10.5 Optimizations**

**10.5.1 Scaled Dot-Product Attention (PyTorch 2.0+)**

```
PyTorch 2.0 has optimized attention:

import torch.nn.functional as F

# Instead of manual implementation:
attention_output = F.scaled_dot_product_attention(
  query_layer,
  key_layer,
  value_layer,
  attn_mask=None,
  dropout_p=self.dropout.p if self.training else 0.0,
  is_causal=False  # Not causal for vision
)

Benefits:
  - Flash Attention under the hood
  - 2-4x faster
  - Lower memory usage
  - Automatic kernel selection

Requirements:
  - PyTorch >= 2.0
  - CUDA GPU with compute capability >= 7.5 (A3000 ✓)

For this project:
  - Use if PyTorch 2.0+ available
  - Otherwise, manual implementation fine
```

**10.5.2 Memory-Efficient Attention**

```
Problem: Attention matrix is [B, H, N, N]
  For 196 patches, 12 heads: [B, 12, 196, 196]
  = B × 12 × 38,416 values × 4 bytes (FP32)
  = B × 1.84 MB
  For batch=4: ~7.4 MB per layer
  For 12 layers: ~89 MB just for attention!

Solutions:
1. Don't store attention scores (only needed for visualization)
2. Use Flash Attention (fuses operations)
3. Use checkpointing (recompute in backward pass)

Implementation for training:
  # Only compute attention when needed
  if output_attentions or (not self.training):
    # Store attention for visualization/eval
    attention_output, attention_probs = self.attention(...)
  else:
    # Don't store, save memory
    attention_output = F.scaled_dot_product_attention(...)
```

#### **10.6 Testing & Validation**

**10.6.1 Attention Tests**

```
1. test_attention_shape():
   config = VisionEncoderConfig(hidden_size=768, num_attention_heads=12)
   attention = VisionAttention(config)

   hidden_states = torch.randn(2, 196, 768)
   output = attention(hidden_states)[0]

   assert output.shape == (2, 196, 768)

2. test_attention_with_output():
   output, attn_weights = attention(hidden_states, output_attentions=True)

   assert output.shape == (2, 196, 768)
   assert attn_weights.shape == (2, 12, 196, 196)

3. test_attention_probabilities_sum_to_one():
   _, attn_weights = attention(hidden_states, output_attentions=True)

   # Sum over last dimension (keys)
   sums = attn_weights.sum(dim=-1)

   # Should be all 1.0
   assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

4. test_attention_is_non_negative():
   _, attn_weights = attention(hidden_states, output_attentions=True)

   assert (attn_weights >= 0).all()
   assert (attn_weights <= 1).all()

5. test_attention_symmetry():
   # Vision attention should be symmetric (no causal mask)
   # Not testing that attn[i,j] == attn[j,i] (that's content-dependent)
   # But shape should be square
   _, attn_weights = attention(hidden_states, output_attentions=True)
   assert attn_weights.shape[-2] == attn_weights.shape[-1]

6. test_attention_gradient_flow():
   hidden_states = torch.randn(1, 196, 768, requires_grad=True)
   output = attention(hidden_states)[0]

   # Compute loss and backprop
   loss = output.sum()
   loss.backward()

   # Gradients should exist
   assert hidden_states.grad is not None
   assert not torch.isnan(hidden_states.grad).any()

7. test_different_num_heads():
   # 4 heads
   config4 = VisionEncoderConfig(hidden_size=768, num_attention_heads=4)
   attn4 = VisionAttention(config4)
   out4, weights4 = attn4(hidden_states, output_attentions=True)
   assert weights4.shape[1] == 4  # num_heads dimension

   # 16 heads
   config16 = VisionEncoderConfig(hidden_size=768, num_attention_heads=16)
   attn16 = VisionAttention(config16)
   out16, weights16 = attn16(hidden_states, output_attentions=True)
   assert weights16.shape[1] == 16
```

**10.6.2 Attention Visualization Tests**

```
def visualize_attention_patterns():
  """Visualize attention patterns."""
  import matplotlib.pyplot as plt

  config = VisionEncoderConfig()
  attention = VisionAttention(config)

  # Random input
  hidden_states = torch.randn(1, 196, 768)

  # Get attention weights
  _, attn_weights = attention(hidden_states, output_attentions=True)
  # Shape: [1, 12, 196, 196]

  # Visualize attention from one patch
  patch_idx = 98  # Center patch (14*7 = 98)

  fig, axes = plt.subplots(3, 4, figsize=(12, 9))

  for head_idx in range(12):
    ax = axes[head_idx // 4, head_idx % 4]

    # Get attention from patch_idx to all patches
    attn = attn_weights[0, head_idx, patch_idx, :].detach().numpy()

    # Reshape to 2D grid
    attn_map = attn.reshape(14, 14)

    im = ax.imshow(attn_map, cmap='hot', vmin=0, vmax=attn.max())
    ax.set_title(f'Head {head_idx}')
    ax.axis('off')

    # Mark center patch
    ax.plot(7, 7, 'b*', markersize=10)

  plt.suptitle(f'Attention from Center Patch (idx={patch_idx})')
  plt.tight_layout()
  plt.savefig('outputs/visualizations/attention_heads.png')

  print("Attention visualization saved!")

Expected patterns (after training):
  - Some heads show local attention (bright around center)
  - Some heads show global attention (distributed)
  - Some heads show row/column patterns
```

#### **10.7 Attention Head Pruning (Advanced)**

**10.7.1 Identifying Important Heads**

```
After training, some heads may be more important than others

Function: compute_head_importance()

def compute_head_importance(model, dataloader):
  """
  Compute importance of each attention head.
  Based on gradient magnitude.
  """
  head_importance = torch.zeros(
    model.config.num_hidden_layers,
    model.config.num_attention_heads
  )

  model.eval()

  for batch in dataloader:
    images = batch["images"]
    labels = batch["labels"]

    # Forward pass
    outputs = model(images, labels=labels, output_attentions=True)
    loss = outputs["loss"]

    # Backward pass
    loss.backward()

    # Accumulate gradient norms for each head
    for layer_idx, layer in enumerate(model.vision_encoder.encoder.layers):
      # Get attention output gradients
      if layer.self_attn.output.weight.grad is not None:
        grad = layer.self_attn.output.weight.grad

        # Average gradient norm per head
        grad_per_head = grad.view(
          model.config.num_attention_heads,
          -1
        )
        head_importance[layer_idx] += grad_per_head.norm(dim=1)

    model.zero_grad()

  # Normalize
  head_importance /= len(dataloader)

  return head_importance

Usage:
  importance = compute_head_importance(model, val_dataloader)

  # Find least important heads
  least_important = importance.view(-1).topk(20, largest=False)
f
  # Can prune these heads to reduce model size
```

---

## Summary of Tasks 6-10

### **Completed Components:**

1. **Task 6: Tokenizer** ✓

   - Word-level tokenization
   - Special tokens ([PAD], [SOS], [EOS], [YES], [NO])
   - Vocabulary building from traffic commands
   - Encoding/decoding with padding
   - ~300-500 token vocabulary

2. **Task 7: Data Loader** ✓

   - PyTorch Dataset class
   - HDF5 image loading
   - Custom collate function with dynamic padding
   - Windows multiprocessing support
   - Batch size 4 with gradient accumulation

3. **Task 8: SigLip Vision Encoder** ✓

   - 6-layer transformer encoder
   - Pre-normalization architecture
   - ~22M parameters (optimized for A3000)
   - Returns [B, 196, 768] visual tokens

4. **Task 9: Vision Embeddings** ✓

   - Patch embedding via convolution (16×16 patches)
   - Learned position embeddings
   - 196 patches from 224×224 image
   - [B, 3, 224, 224] → [B, 196, 768]

5. **Task 10: Vision Attention** ✓
   - Multi-head self-attention (12 heads)
   - Scaled dot-product attention
   - Attention weight extraction for visualization
   - [B, 12, 196, 196] attention maps

### **Key Design Decisions:**

- Image size: 224×224 (standard)
- Patch size: 16×16 (196 patches)
- Hidden size: 384-768 (memory tradeoff)
- Vocabulary: 300-500 tokens (domain-specific)
- No CLS token (PaliGemma style)
- Learned position embeddings (not sinusoidal)
