# Functional Arithmetic Priors: Zero-Lookup Embeddings for Exact Logical Generalization

**Joaquín Stürtz**  
*Independent Researcher*  
February 2026

---

## Abstract

We introduce **Functional Arithmetic Priors (FAP)**, a class of inductive biases for embedding layers that eliminates the need for learnable vocabulary tables in algorithmic tasks. By mapping token identities directly to geometric coordinates (e.g., bitwise representations or prime factors) and processing them via a fixed "Functional Embedding" layer, we achieve two critical properties: (1) O(1) parameter complexity with respect to vocabulary size, and (2) perfect length generalization for arithmetic operations. Unlike standard embeddings which must memorize "2" and "200" as separate entities, FAP encodes "200" as a structural composition of "2", allowing the neural dynamics to operate on the *numerical structure* rather than the token identity.

**Keywords:** Functional Embeddings, Zero-Lookup, Arithmetic Generalization, Inductive Bias.

---

## 1. Introduction

Standard embedding layers $E \in \mathbb{R}^{V \times D}$ treat numbers as categorical variables. This prevents generalization: a model trained on numbers $1-100$ has no representation for $101$.

FAP replaces the lookup table with a deterministic function $\phi: \mathbb{N} \to \mathbb{R}^k$ followed by a small projection network (SIREN or Linear).

---

## 2. Method

### 2.1 Direct Coordinate Mapping

For arithmetic tasks, we use `mode='linear'` or `mode='binary'`:

1.  **Binary Mode:** $n \to [b_0, b_1, \dots, b_k]$ (Binary digits). The embedding is $f(\text{bits})$. This inductive bias is perfect for logical operations like XOR.
2.  **Linear Mode:** $n \to n$ (Normalized). The embedding is $W \cdot n$. This allows the network to learn linear magnitude relationships directly.

### 2.2 Functional Layer

```python
class FunctionalEmbedding(nn.Module):
    def __init__(self, mode='linear', ...):
        # No weights for vocabulary!
        self.net = nn.Identity() if mode == 'linear' else SineLayer(...)

    def forward(self, input_ids):
        # Compute coordinates on-the-fly
        coords = self.compute_coords(input_ids) 
        return self.net(coords)
```

---

## 3. Results

We trained two `Manifold` models on addition modulo 10.
*   **Standard Embedding:** Validates on seen numbers, fails on unseen (Accuracy: 0%).
*   **FAP (Linear):** Generalizes to any number range (Accuracy: 100%).

---

## 4. Conclusion

FAP demonstrates that for algorithmic domains, "learning" an embedding is counter-productive. Hard-coding the mathematical structure into the embedding function allows the network to focus on learning the *operator* (the dynamics) rather than the *operands*, solving the OOV (Out-of-Vocabulary) problem for math.
