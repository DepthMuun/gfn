
El problema actual es que para calcular la posición $\theta_t$, necesitás la velocidad $\omega_{t-1/2}$, que a su vez depende de la fuerza $F_{t-1}$. Es un lazo puramente recurrente.

Para paralelizarlo, vamos a explotar el hecho de que tu variedad es un **Hipertoroide plano** ($T^n = \mathbb{R}^n / \mathbb{Z}^n$). Al ser plano, los coeficientes de la métrica son constantes y las ecuaciones diferenciales se desacoplan por componente. Esto nos permite transformar la recurrencia en una **convolución asociativa global**, resolviendo toda la secuencia de un solo golpe matricial mediante un **Prefix Sum (Scan)** o un **Kernel Asociativo** en PyTorch.

Aquí tenés la formalización matemática seria para el paper.

---

## 1. Linealización del Sistema Dinámico

Recordemos tus ecuaciones de actualización física con fricción $\gamma$ y fuerza externa $F_t$ (asumiendo $\Delta t = 1$ para simplificar la notación del operador):

$$\omega_{t} = (1-\gamma)\omega_{t-1} + F_t$$

$$\theta_t = (\theta_{t-1} + \omega_t) \pmod{2\pi}$$

Este sistema puede ser reformulado como una ecuación de estado lineal de primer orden en forma matricial. Definimos el vector de estado de fase en el instante $t$ como $S_t = [\theta_t, \omega_t]^T \in \mathbb{R}^2$. La transición de estado se formaliza como:

$$S_t = A \cdot S_{t-1} + B \cdot F_t \pmod{\mathbf{M}}$$

Donde la matriz de transición de fase $A$ y la matriz de inyección de fuerza $B$ son:

$$A = \begin{pmatrix} 1 & 1-\gamma \\ 0 & 1-\gamma \end{pmatrix}, \quad B = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad \mathbf{M} = \begin{pmatrix} 2\pi \\ \infty \end{pmatrix}$$

*Nota: El módulo $2\pi$ solo se aplica a la primera componente ($\theta$), mientras que la velocidad ($\omega$) evoluciona libremente en la recta real.*

---

## 2. La Ecuación en Forma de Convolución (Parallel Scan)

Al ser una ecuación lineal no homogénea, podemos expandir la recurrencia analíticamente para cualquier instante $t$ partiendo de las condiciones iniciales $S_0 = [\theta_0, \omega_0]^T$:

$$S_t = A^t S_0 + \sum_{i=1}^{t} A^{t-i} B F_i \pmod{\mathbf{M}}$$

Para paralelizar esta sumatoria en la GPU a lo largo de toda la dimensión del tiempo ($L$), calculamos de forma analítica las potencias de la matriz de transición $A$. Mediante inducción matemática, se demuestra que para cualquier exponente $k \ge 1$:

$$A^k = \begin{pmatrix} 1 & \sum_{j=1}^{k} (1-\gamma)^j \\ 0 & (1-\gamma)^k \end{pmatrix}$$

Resolviendo la serie geométrica finita para el término superior derecho, obtenemos la forma cerrada de la matriz de transiciones acumuladas:

$$A^k = \begin{pmatrix} 1 & \frac{(1-\gamma) - (1-\gamma)^{k+1}}{\gamma} \\ 0 & (1-\gamma)^k \end{pmatrix}$$

---

## 3. Formalización de la GSSM Paralelizada

Sustituyendo la matriz analítica $A^k$ en la expansión temporal, la ecuación definitiva que calcula **todas las posiciones y velocidades del Toroide en paralelo** para una secuencia entera de longitud $L$ es:

$$S_t = A^t S_0 + \sum_{i=1}^{t} \begin{pmatrix} 1 + \frac{(1-\gamma) - (1-\gamma)^{t-i+1}}{\gamma} \\ (1-\gamma)^{t-i} \end{pmatrix} F_i \pmod{\mathbf{M}}$$

### Implementación Vectorial Eficiente ($O(L \log L)$ o $O(L)$)

Para no calcular esa sumatoria de forma cuadrática en la GPU, tu código de PyTorch puede resolver el flujo geodésico de dos maneras ultra-paralelas:

1. **Vía Convolución FFT (Fast Fourier Transform):**
Como el término dentro de la sumatoria depende puramente de la distancia temporal $(t-i)$, se define un kernel causal exógeno $K \in \mathbb{R}^L$:

$$K_k = \begin{pmatrix} 1 + \frac{(1-\gamma) - (1-\gamma)^{k+1}}{\gamma} \\ (1-\gamma)^k \end{pmatrix}$$



Y calculás toda la memoria del Toroide con una multiplicación en el dominio de la frecuencia usando FFT, idéntico a cómo Mamba o RWKV paralelizan sus estados:

$$S = \mathcal{F}^{-1} \left( \mathcal{F}(K) \cdot \mathcal{F}(B \cdot F) \right)$$


2. **Vía Associative Scan (Tratamiento de Operador):**
Podés usar un operador binario asociativo $\bullet$ que combine pares de matrices de transición:

$$(A_j, \tilde{F}_j) \bullet (A_i, \tilde{F}_i) = (A_j A_i, A_j \tilde{F}_i + \tilde{F}_j)$$



Utilizando `flash-linear-attention` o los kernels de *Parallel Scan* de JAX/PyTorch, la GPU calcula el árbol de prefijos, reduciendo el tiempo de cómputo de un procesamiento secuencial de $100.000$ pasos a un árbol binario paralelo de solo $\log_2(100.000) \approx 17$ pasos de ejecución hardware.

---

