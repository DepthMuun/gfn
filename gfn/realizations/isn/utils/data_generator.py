"""
Simple arithmetic data generator for training.

Generates problems like "2 + 3 = 5"
"""

import random
from typing import List, Tuple, Dict
import torch


class ArithmeticDataGenerator:
    """
    Generates simple arithmetic problems.
    
    Format: "num1 op num2 = result"
    """
    
    def __init__(
        self,
        min_digits: int = 1,
        max_digits: int = 2,
        operations: List[str] = ['add'],
        seed: int = 42
    ):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.operations = operations
        
        random.seed(seed)
        
        # Vocabulary
        self.vocab = self._build_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
    
    def _build_vocab(self) -> List[str]:
        """Build vocabulary."""
        vocab = ['<PAD>', '<START>', '<END>', '<UNK>']
        
        # Add digits
        for i in range(10):
            vocab.append(str(i))
        
        # Add operations
        vocab.extend(['+', '-', '*', '/', '='])
        
        # Add numbers up to 999
        for i in range(1000):
            vocab.append(str(i))
        
        return vocab
    
    def generate_problem(self) -> Tuple[str, str, int]:
        """
        Generate a single arithmetic problem.
        
        Returns:
            (input_str, output_str, result)
        """
        # Generate numbers
        min_val = 10 ** (self.min_digits - 1)
        max_val = 10 ** self.max_digits - 1
        
        num1 = random.randint(min_val, max_val)
        num2 = random.randint(min_val, max_val)
        
        # Choose operation
        op = random.choice(self.operations)
        
        # Compute result
        if op == 'add' or op == '+':
            result = num1 + num2
            op_symbol = '+'
        elif op == 'subtract' or op == '-':
            result = num1 - num2
            op_symbol = '-'
        elif op == 'multiply' or op == '*':
            result = num1 * num2
            op_symbol = '*'
        elif op == 'divide' or op == '/':
            if num2 == 0:
                num2 = 1
            result = num1 // num2  # Integer division
            op_symbol = '/'
        else:
            result = num1 + num2
            op_symbol = '+'
        
        # Format as string
        input_str = f"{num1} {op_symbol} {num2}"
        output_str = str(result)
        
        return input_str, output_str, result
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to IDs."""
        tokens = text.split()
        ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id['<UNK>'])
        
        return ids
    
    def detokenize(self, ids: List[int]) -> str:
        """Convert IDs back to text."""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if token not in ['<PAD>', '<START>', '<END>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def generate_dataset(
        self,
        n_samples: int
    ) -> List[Dict[str, any]]:
        """
        Generate a dataset of problems.
        
        Returns:
            List of dicts with 'input', 'output', 'result'
        """
        dataset = []
        
        for _ in range(n_samples):
            input_str, output_str, result = self.generate_problem()
            
            # Tokenize
            input_ids = [self.token_to_id['<START>']] + self.tokenize(input_str)
            output_ids = [self.token_to_id['<START>']] + self.tokenize(output_str) + [self.token_to_id['<END>']]
            
            dataset.append({
                'input': input_str,
                'output': output_str,
                'result': result,
                'input_ids': input_ids,
                'output_ids': output_ids
            })
        
        return dataset
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch for DataLoader.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched tensors
        """
        # Find max lengths
        max_input_len = max(len(sample['input_ids']) for sample in batch)
        max_output_len = max(len(sample['output_ids']) for sample in batch)
        
        # Pad sequences
        input_ids = []
        output_ids = []
        
        for sample in batch:
            # Pad input
            padded_input = sample['input_ids'] + [self.token_to_id['<PAD>']] * (max_input_len - len(sample['input_ids']))
            input_ids.append(padded_input)
            
            # Pad output
            padded_output = sample['output_ids'] + [self.token_to_id['<PAD>']] * (max_output_len - len(sample['output_ids']))
            output_ids.append(padded_output)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'inputs': [sample['input'] for sample in batch],
            'outputs': [sample['output'] for sample in batch],
            'results': [sample['result'] for sample in batch]
        }
