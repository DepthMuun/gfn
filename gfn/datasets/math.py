from torch.utils.data import IterableDataset
import random
import torch

class MathDataset(IterableDataset):
    """
    Infinite stream of arithmetic problems "A op B = C".
    """
    def __init__(self, size=None, max_digits=8, file_path=None):
        """
        Args:
            size: Ignored, kept for compatibility.
            max_digits: Maximum digits for operands.
            file_path: Optional path to .txt file with fixed examples.
        """
        self.max_digits = max_digits
        self.file_path = file_path
        self.samples = []
        
        if self.file_path:
            with open(self.file_path, 'r') as f:
                self.samples = [line.strip() for line in f if line.strip()]
            print(f"[{self.__class__.__name__}] Loaded {len(self.samples)} fixed samples from {self.file_path}")

        # Vocabulary: 0-9, +, -, *, =, <PAD>, <EOS>
        self.char_to_id = {str(i): i for i in range(10)}
        self.char_to_id['+'] = 10
        self.char_to_id['-'] = 11
        self.char_to_id['*'] = 12
        self.char_to_id['='] = 13
        self.char_to_id['<PAD>'] = 14
        self.char_to_id['<EOS>'] = 15
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        
    def _generate_problem(self):
        ops = ['+', '-', '*']
        op = random.choice(ops)
        
        if op == '*':
            # Limit multiplication complexity
            a = random.randint(0, 10**min(3, self.max_digits) - 1)
            b = random.randint(0, 10**min(3, self.max_digits) - 1)
            c = a * b
        elif op == '-':
            a = random.randint(0, 10**self.max_digits - 1)
            b = random.randint(0, a)  # Positive result
            c = a - b
        else:
            a = random.randint(0, 10**self.max_digits - 1)
            b = random.randint(0, 10**self.max_digits - 1)
            c = a + b
            
        s = f"{a}{op}{b}={c}"
        return s

    def __iter__(self):
        if self.file_path and self.samples:
            # Fixed Dataset Mode (Cyclic)
            while True:
                # Shuffle for better training dynamics
                indices = list(range(len(self.samples)))
                random.shuffle(indices)
                for i in indices:
                    s = self.samples[i]
                    ids = [self.char_to_id[c] for c in s]
                    ids.append(self.char_to_id['<EOS>'])
                    yield torch.tensor(ids, dtype=torch.long)
        else:
            # Infinite Stream Mode
            while True:
                s = self._generate_problem()
                ids = [self.char_to_id[c] for c in s]
                ids.append(self.char_to_id['<EOS>'])
                yield torch.tensor(ids, dtype=torch.long)
            
    def collate_fn(self, batch):
        # Pad to max length in batch
        max_len = max(len(x) for x in batch)
        padded_batch = []
        for x in batch:
            pad_len = max_len - len(x)
            padded = torch.cat([x, torch.full((pad_len,), self.char_to_id['<PAD>'], dtype=torch.long)])
            padded_batch.append(padded)
        return torch.stack(padded_batch)

    def decode(self, ids):
        chars = []
        for i in ids:
            i = i.item() if isinstance(i, torch.Tensor) else i
            if i == self.char_to_id['<EOS>']:
                break
            if i == self.char_to_id['<PAD>']:
                continue
            chars.append(self.id_to_char.get(i, '?'))
        return "".join(chars)
