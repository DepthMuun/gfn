import torch
from torch.utils.data import Dataset
import random

class MathDataset(Dataset):
    """
    Generador sintético de operaciones aritméticas básicas.
    Ejemplo: '15+04=19' -> [1, 5, 10, 0, 4, 12, 1, 9]
    Vocabulario: 0-9 (0-9), + (10), - (11), = (12), <PAD> (13), * (14), / (15)
    """
    def __init__(self, num_samples=10000, operations=['+']):
        self.num_samples = num_samples
        self.operations = operations
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({'+': 10, '-': 11, '=': 12, 'PAD': 13, '*': 14, '/': 15})
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self.samples = []
        for _ in range(num_samples):
            self.samples.append(self._generate_sample())

    def _generate_sample(self):
        op = random.choice(self.operations)
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        
        if op == '+':
            res = a + b
        elif op == '-':
            # Aseguramos resultado positivo para simplificar vocabulario inicial
            if a < b: a, b = b, a
            res = a - b
        
        # Formatear como string de longitud fija: "XX+YY=ZZZ" (L=8 o 9)
        # Usamos 2 dígitos para operandos y hasta 3 para el resultado (Suma de 99+99=198)
        # Formato: AAopBB=CCC
        expr = f"{a:02d}{op}{b:02d}={res:03d}"
        
        token_ids = [self.vocab[char] for char in expr]
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Para entrenamiento sequence-to-sequence autoregresivo:
        # Input: "AAopBB="
        # Target: "CCC" (o el shift completo para que aprenda todo si queremos)
        full_seq = self.samples[idx]
        return full_seq

    def decode(self, ids):
        return "".join([self.inv_vocab[int(i)] for i in ids])

if __name__ == "__main__":
    ds = MathDataset(num_samples=5, operations=['+', '-'])
    for i in range(len(ds)):
        sample = ds[i]
        print(f"Sample: {sample} -> {ds.decode(sample)}")
