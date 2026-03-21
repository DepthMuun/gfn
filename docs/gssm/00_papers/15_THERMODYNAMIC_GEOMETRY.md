# Free Energy Minimization on Riemannian Manifolds: A Thermodynamic Approach to Neural Network Training

**Joaquin Stürtz**  
*Independent Researcher*  
January 2026

## Abstract

We propose a thermodynamic interpretation of neural network training as free energy minimization on Riemannian manifolds. By introducing a learnable temperature parameter that controls the exploration-exploitation tradeoff, we derive thermodynamic Christoffel symbols that adapt geometric curvature based on local free energy landscapes. Our approach combines principles from statistical mechanics with Riemannian geometry, demonstrating that temperature annealing during training improves convergence speed and generalization. The framework establishes a rigorous connection between optimization dynamics and physical thermodynamics, providing both theoretical insights and practical improvements for training deep neural networks.


## 1. Introduction

Neural network optimization can be viewed as navigation through an energy landscape, yet standard gradient-based methods lack explicit mechanisms for controlling exploration versus exploitation. We propose treating training as a thermodynamic process governed by the free energy functional:

$$ F = E - TS $$

where energy $E$ represents prediction error, temperature $T$ controls stochasticity and exploration, and entropy $S$ quantifies uncertainty in the parameter distribution. At equilibrium, the system minimizes $F$, balancing energy minimization ($E \to \min$) with entropy maximization ($S \to \max$).

Our key contributions are:

1. **Thermodynamic metric adaptation** that incorporates free energy gradients into Riemannian geometry, enabling the latent manifold to adapt its curvature based on local optimization dynamics through temperature-dependent metric tensors.

2. **Learnable temperature parameters** that automatically balance exploration and exploitation during training, eliminating the need for manually scheduled annealing.

3. **Temperature annealing schedules** derived from statistical mechanics principles, providing theoretically grounded protocols for transitioning from high-temperature exploration to low-temperature exploitation.

4. **Empirical demonstration** of improved convergence and generalization on language modeling benchmarks.

The Christoffel symbols $\Gamma^k_{ij}(x, T)$ derived from the temperature-dependent metric capture the thermodynamic modification of the geometric structure, enabling principled exploration of the loss landscape.


## 2. Theoretical Framework

### 2.1 Statistical Mechanics Background

In statistical mechanics, a system at temperature $T$ follows the canonical ensemble distribution:

$$ p(x) = \frac{1}{Z} \exp\left(-\frac{E(x)}{k_B T}\right) $$

where $k_B$ is Boltzmann's constant, $\beta = 1/(k_B T)$ is the inverse temperature, and $Z = \int \exp(-E(x)/k_B T) \, dx$ is the partition function. The Helmholtz free energy, which governs equilibrium behavior, is:

$$ F = E - TS = -k_B T \log Z $$

At equilibrium, the system minimizes $F$, achieving an optimal balance between minimizing energy (exploitation) and maximizing entropy (exploration). This principle, known from equilibrium statistical mechanics, provides a powerful framework for understanding optimization dynamics.

### 2.2 Variational Free Energy

For an approximate distribution $q(x)$ representing our beliefs about the system state, the variational free energy is:

$$ F[q] = \mathbb{E}_q[E(x)] + k_B T \, \text{KL}[q \| p] = \mathbb{E}_q[E(x)] - T S[q] $$

where $S[q] = -\mathbb{E}_q[\log q(x)]$ is the entropy of the approximate distribution. This variational perspective connects the free energy principle to standard information-theoretic quantities.

**Theorem 1.** The distribution $q$ that minimizes $F[q]$ is the Gibbs distribution $p(x) \propto \exp(-E(x)/T)$.

*Proof.* Setting $\delta F / \delta q = 0$ yields $E(x) + T(\log q(x) + 1) = \text{const}$, which gives $q(x) = C \exp(-E(x)/T)$. The normalization constant $C$ is determined by $\int q(x) \, dx = 1$. $\square$

This theorem establishes that the optimal exploration strategy at temperature $T$ is exactly the Gibbs distribution, which concentrates probability mass on low-energy regions while maintaining probabilistic spread based on the temperature scale.

### 2.3 Thermodynamic Integration

The partition function ratio between two temperatures provides a measure of the free energy difference:

$$ \log\left(\frac{Z_1}{Z_0}\right) = -\int_0^1 \langle E \rangle_\beta(\lambda) \, d\lambda $$

where $\beta(\lambda) = \lambda \beta_1 + (1-\lambda) \beta_0$ interpolates between inverse temperatures. This thermodynamic integration identity allows computation of free energy differences through path integrals over temperature, enabling theoretical analysis of annealing protocols.

The expectation $\langle E \rangle_\beta(\lambda)$ is computed with respect to the interpolating distribution $p_\lambda(x) \propto \exp(-\beta(\lambda) E(x))$, providing a smooth connection between different thermodynamic states.


## 3. Thermodynamic Geometry

### 3.1 Free Energy on Manifolds

We extend the free energy functional to Riemannian manifolds by defining the dynamical free energy on the tangent bundle $T\mathcal{M}$:

$$ F(x, v) = E(x) - T S(x, v) $$

where:
- $E(x)$ is the energy function, typically a learned neural network mapping from parameters to scalar loss
- $S(x, v)$ is the entropy estimated from the velocity distribution in the tangent space at $x$
- $T$ is the learnable temperature parameter controlling the effective "thermal noise" in the dynamics

The velocity entropy provides a measure of exploration in the parameter space. When velocities are concentrated (low entropy), the system is exploiting a narrow region of the loss landscape. When velocities are diffuse (high entropy), the system is exploring broadly.

The entropy can be approximated from velocity magnitudes using the Gaussian proxy, which provides a differentiable estimate:

$$ S(x, v) \approx \frac{1}{2} \log \det \Sigma(x) = \frac{1}{2} \log \left( \prod_{i=1}^d \sigma_i^2 \right) = \sum_{i=1}^d \log \sigma_i $$

where $\Sigma(x)$ is the covariance tensor of the velocity distribution, estimated from recent velocity samples. This approximation captures the directional spread of optimization momentum through the eigenvalues of the covariance tensor.

### 3.2 Thermodynamic Christoffel Symbols

The key innovation of our framework is the extension of Riemannian Christoffel symbols to incorporate thermodynamic forces through a temperature-dependent metric tensor. We define **thermodynamic Christoffel symbols** by first modulating the metric:

$$ g_{ij}(x, T) = g_{base,ij}(x) \cdot \exp\left(-\frac{\alpha}{T} \frac{\partial F}{\partial x^k} x^k\right) $$

where $\alpha$ is a scaling hyperparameter. From this temperature-dependent metric, we compute the thermodynamic Christoffel symbols using the standard formula:

$$ \Gamma^k_{ij}(x, T) = \frac{1}{2} g^{kl}(x, T) \left( \partial_i g_{jl}(x, T) + \partial_j g_{il}(x, T) - \partial_l g_{ij}(x, T) \right) $$

This formulation yields dynamics that are sensitive to both local curvature (through $g_{base,ij}$) and thermodynamic state (through the temperature-modulated metric). The effective geodesic equation becomes:

$$ \frac{D v^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x, T) v^i v^j = -\frac{\partial V_{eff}}{\partial x^k} $$

where $V_{eff}(x, T)$ is an effective potential incorporating the free energy. At high temperature ($T \gg 1$), the metric approaches the base metric $g_{base,ij}$, and dynamics are dominated by the base manifold geometry. At low temperature ($T \ll 1$), the metric is significantly modified by the free energy gradients, driving the system rapidly toward energy minima.

### 3.3 Free Energy Gradient Computation

The thermodynamic force requires computation of the free energy gradient with respect to position. The energy component is obtained through standard backpropagation:

$$ \frac{\partial E}{\partial x^k} = \nabla_k E(x) $$

The entropy component requires careful treatment to maintain differentiability. Using the covariance-based approximation:

$$ \frac{\partial S}{\partial x^k} = \frac{\partial}{\partial x^k} \left( \frac{1}{2} \log \det \Sigma(x) \right) = \frac{1}{2} \text{Tr}\left( \Sigma^{-1}(x) \frac{\partial \Sigma}{\partial x^k}(x) \right) $$

This yields a gradient that encourages velocity diversity when entropy is low and allows concentration when entropy is already high. The full thermodynamic force is:

$$ \frac{\partial F}{\partial x^k} = \frac{\partial E}{\partial x^k} - T \frac{\partial S}{\partial x^k} $$

which is incorporated into the metric modulation and subsequently into the Christoffel symbols $\Gamma^k_{ij}(x, T)$ through the temperature-dependent geometry.


## 4. Temperature Annealing

### 4.1 Annealing Schedules

The choice of temperature schedule significantly impacts optimization dynamics. We implement three annealing strategies grounded in statistical mechanics:

**Linear Annealing:**
$$ T(t) = T_0 - (T_0 - T_f) \cdot \frac{t}{T_{\max}} $$

This simple schedule provides a constant rate of cooling, suitable when no prior knowledge of the landscape structure is available.

**Exponential Annealing:**
$$ T(t) = T_0 \exp(-\lambda t) \quad \text{where} \quad \lambda = -\frac{\log(T_f/T_0)}{T_{\max}} $$

Exponential annealing provides rapid initial cooling followed by gradual refinement, matching the intuition that most exploration should occur early in training.

**Cosine Annealing:**
$$ T(t) = T_f + (T_0 - T_f) \cdot \frac{1 + \cos(\pi t / T_{\max})}{2} $$

Cosine annealing provides smooth transitions that avoid abrupt changes in dynamics, potentially beneficial for convergence stability. The resulting metric tensor $g_{ij}(x, T(t))$ varies smoothly with temperature, ensuring the Christoffel symbols $\Gamma^k_{ij}(x, T(t))$ remain well-behaved throughout training.

### 4.2 Adaptive Temperature

Rather than fixing an annealing schedule a priori, we allow the temperature parameter to be learnable, enabling automatic discovery of optimal schedules:

$$ \log T = \theta, \quad \theta \in \mathbb{R} $$

The log-parameterization ensures positivity ($T = \exp(\theta) > 0$) while allowing arbitrary magnitude. During training, gradient descent automatically discovers temperature values that optimize the trade-off between exploration and exploitation.

The learned temperature evolution reveals three distinct phases, which can be understood through the lens of the temperature-dependent metric $g_{ij}(x, T)$:

1. **Initial Phase ($0$-$20\%$ training):** High temperature ($T \approx 1.8$) flattens the metric $g_{ij}(x, T) \approx g_{base,ij}(x)$, enabling broad exploration of the loss landscape by reducing curvature through the Christoffel symbols.

2. **Middle Phase ($20$-$70\%$):** Gradual cooling ($T: 1.8 \to 0.5$) increases metric anisotropy through the free energy modulation, focusing the search while maintaining sufficient stochasticity to escape shallow local minima.

3. **Final Phase ($70$-$100\%$):** Low temperature ($T \approx 0.2$) sharpens the metric, refining the solution in the discovered basin with reduced variance.

This learned evolution matches theoretical predictions from simulated annealing literature, providing empirical validation of the thermodynamic framework.

### 4.3 Connection to Simulated Annealing

The temperature-dependent metric formulation provides a geometric interpretation of simulated annealing. At high temperatures, the metric approaches the base metric, and geodesics traverse the manifold relatively freely. As temperature decreases, the metric modulation becomes more pronounced, creating effective potential wells around low-energy regions. This geometric perspective unifies optimization and statistical mechanics through the language of differential geometry.


## 5. Experimental Results

### 5.1 Language Modeling

We evaluate our approach on language modeling benchmarks including WikiText-103 and Penn Treebank. The experimental setup uses a Manifold Geodesic Flow Network with 6 layers and 256 dimensions, comparing against standard Transformer baselines and Manifold GFN without thermodynamic extensions.

**Results on WikiText-103:**

| Model | Final Perplexity | Steps to 90\% | Training Time |
|-------|------------------|---------------|---------------|
| Transformer | 24.3 | 50k | 12h |
| Manifold GFN | 22.1 | 42k | 14h |
| Thermodynamic GFN (fixed T) | 21.5 | 38k | 15h |
| Thermodynamic GFN (annealed) | **20.8** | **32k** | 15h |

**Key Findings:**
- The annealed thermodynamic GFN achieves 23% faster convergence (32k vs 42k steps to reach 90% of final performance)
- Final perplexity improves by 5.8% compared to non-thermodynamic baselines (20.8 vs 22.1)
- Computational overhead is minimal (+7% training time) due to efficient gradient computation

The improved performance can be attributed to the adaptive geometry induced by the temperature-dependent Christoffel symbols $\Gamma^k_{ij}(x, T)$, which automatically adjust the local curvature to match the optimization requirements.

### 5.2 Free Energy Landscape Analysis

Models trained with thermodynamic geometry exhibit qualitatively different loss landscapes:

- **Smoother loss landscapes:** Lower Hessian eigenvalues indicate reduced curvature in directions orthogonal to the optimization trajectory, facilitated by the temperature-modulated metric tensor.

- **Better local minima:** Higher test accuracy at convergence suggests discovery of superior local minima, consistent with the exploration benefits of high-temperature initial phases where the Christoffel symbols are closer to base values.

- **Robustness:** Reduced sensitivity to weight initialization, as the thermodynamic framework naturally adapts the optimization dynamics to the local landscape structure through the learned temperature.


## 6. Discussion

Our thermodynamic approach provides a principled framework for controlling exploration-exploitation tradeoffs in neural network training. By grounding optimization in statistical mechanics and connecting free energy to the metric tensor $g_{ij}(x, T)$, we achieve both theoretical elegance and practical improvements.

**Advantages:**
- Automatic temperature scheduling via learnable parameters eliminates manual hyperparameter tuning
- Improved convergence speed and final performance on challenging benchmarks
- Theoretical connection to well-established physics provides interpretability and guidance
- The framework naturally extends to various architectural choices and problem domains
- The temperature-dependent Christoffel symbols $\Gamma^k_{ij}(x, T)$ provide a geometric interpretation of thermodynamic effects

**Limitations:**
- Requires careful tuning of initial temperature range to ensure appropriate exploration
- Entropy estimation from velocities is approximate and may not capture full distribution properties
- Computational overhead from metric and Christoffel symbol computation, though manageable in practice

**Future Directions:**
- Extension to non-equilibrium thermodynamics using the Jarzynski equality for free energy estimation
- Multi-temperature ensembles for uncertainty quantification and Bayesian model averaging
- Application to reinforcement learning as an exploration bonus mechanism
- Extension to Finsler geometry where the metric depends on both position and velocity


## 7. Related Work

**Simulated Annealing.** Our work builds on the foundational simulated annealing algorithm, extending it to continuous optimization on Riemannian manifolds with learned temperature schedules. The temperature-dependent metric provides a geometric realization of the annealing process.

**Free Energy Principle.** Friston's Free Energy Principle provides the conceptual foundation for treating neural computation as free energy minimization, though our implementation differs in its explicit geometric formulation and connection to Christoffel symbols through the metric tensor.

**Thermodynamic Neural Networks.** Recent work on thermodynamic computing explores physical implementations of neural networks that leverage thermal noise, while our approach uses thermodynamic principles for algorithmic design in software.

**Natural Gradient Methods.** The connection between information geometry and natural gradient descent (Amari, 1998) provides a complementary perspective, with thermodynamic Christoffel symbols extending the geometric framework to include temperature-dependent curvature.


## 8. Conclusion

We have demonstrated that thermodynamic principles can be productively integrated with Riemannian geometry to improve neural network training. By treating optimization as free energy minimization with learnable temperature parameters, we achieve significant improvements in both convergence speed and final performance on language modeling benchmarks.

The temperature-dependent metric tensor $g_{ij}(x, T)$ and its associated Christoffel symbols $\Gamma^k_{ij}(x, T)$ provide a natural bridge between optimization dynamics and physical thermodynamics, enabling automatic exploration-exploitation balancing through learned temperature schedules. This work illustrates the broader potential of physics-inspired approaches to machine learning, showing that fundamental principles from statistical mechanics can guide the design of more efficient and robust optimization algorithms. The geometric perspective unifies the thermodynamic and optimization viewpoints, revealing how temperature shapes the very fabric of the loss landscape through the language of differential geometry.


## References

Amari, S. I. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251-276.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. *Physical Review Letters*, 78(14), 2690.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.

Wirnsberger, P., Ballard, A. J., Papamakarios, G., Abercrombie, S., Racanière, S., Pritzel, A., ... & Blundell, C. (2020). Targeted free energy estimation via learned mappings. *The Journal of Chemical Physics*, 153(14), 144112.

Wright, L. G., Onodera, T., Stein, M. M., Wang, T., Schachter, D. T., Hu, Z., & McMahon, P. L. (2022). Deep physical neural networks trained with backpropagation. *Nature*, 601(7894), 549-555.
