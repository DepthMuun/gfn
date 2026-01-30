import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import random

class MixedHFDataset(IterableDataset):
    """
    Streams data from multiple HF datasets:
    1. Multilingual Text (Wikipedia En/Es)
    2. Mathematics (OpenWebMath or comparable)
    """
    def __init__(self, tokenizer=None, seq_len=128):
        self.seq_len = seq_len
        
        if tokenizer is None:
            # Use GPT-2 tokenizer as standard
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # GPT2 has no PAD token usually, set it to EOS
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
            
        print("Loading Streaming Datasets...")
        # Wikipedia (multilingual) -> Switched to Wikitext for stability on Cloud
        self.wiki_en = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        
        # Math Thinking Dataset (Higher quality reasoning)
        print("Loading MathInstruct...")
        self.math_thinking = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
        
        
        self.vocab_size = len(self.tokenizer)
        
    def __iter__(self):
        # Create iterators
        iter_wiki = iter(self.wiki_en)
        iter_math = iter(self.math_thinking)
        
        while True:
            # Mix: 33% Wiki, 33% MathThinking, 33% Synthetic Arithmetic
            r = random.random()
            if r < 0.33:
                try:
                    item = next(iter_wiki)
                    text = item['text']
                except StopIteration:
                    iter_wiki = iter(self.wiki_en)
                    continue
            elif r < 0.66:
                try:
                    item = next(iter_math)
                    text = f"Q: {item['instruction']}\nA: {item['output']}"
                except StopIteration:
                    iter_math = iter(self.math_thinking)
                    continue
            else:
                a = random.randint(0, 10**8 - 1)
                b = random.randint(0, 10**8 - 1)
                c = a + b
                text = f"Math: {a} + {b} = {c}"
            
            if not text.strip(): continue
            yield text

def get_collate_fn(tokenizer, seq_len):
    def collate_fn(batch):
        # batch is a list of raw strings
        enc = tokenizer(batch, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
        return enc['input_ids'], enc['attention_mask']
    return collate_fn
