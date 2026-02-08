# Documentación Completa de Manifold: Guía Definitiva para Arquitecturas de Flujo Geodésico

**Versión del Documento:** 1.0.0  
**Fecha de Creación:** 7 de Febrero de 2026  
**Autor:** Sistema de Documentación Automatizada  
**Estado:** Referencia Canonical

---

## Tabla de Contenidos

1. [Introducción y Filosofía del Proyecto](#1-introducción-y-filosofía-del-proyecto)
2. [Fundamentos Matemáticos](#2-fundamentos-matemáticos)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
4. [Componentes del Modelo](#4-componentes-del-modelo)
5. [Física y Dinámica del Sistema](#5-física-y-dinámica-del-sistema)
6. [Integración Numérica](#6-integración-numérica)
7. [Optimización Riemanniana](#7-optimización-riemanniana)
8. [Configuración y Uso](#8-configuración-y-uso)
9. [Antipatrones y Errores Comunes](#9-antipatrones-y-errores-comunes)
10. [Verificación y Testing](#10-verificación-y-testing)
11. [Apéndices](#11-apéndices)

---

## 1. Introducción y Filosofía del Proyecto

### 1.1 Visión General del Proyecto

Manifold representa una reformulación fundamental de la arquitectura de redes neuronales profundas,基底基于微分几何和哈密顿力学的基本原理。传统深度学习架构（如Transformer、LSTM、CNN）主要通过经验实验设计，组件的添加或修改基于基准测试任务的性能表现，理论理解往往落后于实现。这种方法虽然产生了非常有效的系统，但对于理解故障模式、预测新领域行为或设计有原则的改进所提供的指导非常有限。

Manifold采用了一种根本不同的方法，从物理学的第一性原理推导架构。这种推导不仅仅是隐喻性的或启发性的；它是严格的，并产生具体的、可测试的预测。物理优先的方法提供了几个优势。首先，它提供了理论保证：如果数学是正确的，那么某些性质（稳定性、保守性、信息流）必然遵循。其次，它提供了可解释性：系统的行为可以通过物理直觉的视角来理解。第三，它建议了有原则的修改：可以通过添加适当的物理机制来推导出新的能力。

本文档旨在提供对Manifold架构的完整技术参考，描述每个组件、其输入和输出、内部结构和配置选项。架构被呈现为一个分层系统，其中每一层都建立在前一层建立的基础之上，既能理解单个组件，又能欣赏它们在完整系统中的集成。

### 1.2 La Hipótesis del Flujo Geodésico

La Hipótesis del Flujo Geodésico（GFH）propone que la memoria y la computación son procesos físicamente ortogonales en sistemas inteligentes. La memoria es subproducto de la conservación de energía（física hamiltoniana）， mientras que la computación es subproducto de la irreversibilidad y disipación de energía（termodinámica）。 Esta separación no es arbitraria sino que emerge de los requisitos conflictivos para el almacenamiento a largo plazo y la actualización instantánea.

En el límite newtoniano（puro conservador），un sistema hamiltoniano almacena información en el momento del estado de la partícula. Aunque esto permite la persistencia de memoria de horizonte infinito（vía el teorema de Liouville）， introduce sensibilidad extrema al orden. En un sistema de segundo orden（ẍ = F）， las entradas tempranas ejercen un apalancamiento desproporcionado sobre la posición final en comparación con las entradas tardías， haciendo que el aprendizaje de operaciones conmutativas（por ejemplo， suma o paridad）sea numéricamente inestable。

En el límite aristotélico（puro disipativo）， donde la velocidad es proporcional a la fuerza（v ∝ F）， el estado actúa como un acumulador de primer orden perfecto（ẋ ≈ F）。 Esta arquitectura sobresale en computación discreta pero falla en la transmisión de largo alcance： sin una fuerza activa， el estado permanece congelado o decae， perdiendo el "momento inercial" requerido para llevar el contexto a través de brechas temporales。

**Proposición Fundamental:** Ninguno de los límites（puramente conservador o puramente disipativo）es suficiente para una inteligencia general. Un motor cognitivo robusto debe ser capaz de conmutación dinámica de régimen.

### 1.3 Objetivos de Este Documento

Este documento tiene como objetivo servir como referencia definitiva tanto para humanos como para sistemas de inteligencia artificial que necesitan comprender， implementar o modificar el sistema Manifold。 Los objetivos específicos incluyen：

- Proporcionar una descripción completa de la arquitectura y sus componentes
- Explicar los fundamentos matemáticos y físicos de manera accesible
- Documentar las convenciones de implementación y las mejores prácticas
- Identificar antipatrones y errores comunes que deben evitarse
- Establecer estándares de verificación y testing
- Proporcionar guías de configuración para diferentes casos de uso

### 1.4 Convenciones del Documento

A lo largo de este documento， se utilizan las siguientes convenciones：

**Notación Matemática:**
- $x$ representa la posición（estado latente）
- $v$ representa la velocidad（momento implícito）
- $F$ representa la fuerza externa（embedding de token）
- $\Gamma$ representa los símbolos de Christoffel（curvatura）
- $\mu$ representa el coeficiente de fricción（disipación）
- $dt$ representa el paso de tiempo de integración
- $\mathcal{H}$ representa el Hamiltoniano（pseudo-energía）

**Convenciones de Código:**
- Todos los ejemplos de código están en Python usando PyTorch
- Los nombres de clases están en CamelCase（por ejemplo， `Manifold`， `ChristoffelModule`）
- Los nombres de funciones y variables están en snake_case（por ejemplo， `forward_step`， `compute_christoffel`）
- Las constantes están en MAYÚSCULAS_SNAKE_CASE（por ejemplo， `DEFAULT_DT`， `MAX_CURVATURE`）

**Advertencias y Notas:**
- ⚠️ **ADVERTENCIA:** Indica información crítica que debe considerarse cuidadosamente
- 💡 **NOTA:** Proporciona información complementaria o contexto
- ❌ **NO HACER:** Indica prácticas que deben evitarse
- ✅ **HACER:** Indica prácticas recomendadas

---

## 2. Fundamentos Matemáticos

### 2.1 Variedades Riemannianas

Una variedad riemanniana es un par（$M$， $g$） donde $M$ es una variedad suave（un espacio topológico que localmente se parece al espacio euclidiano） y $g$ es una métrica riemanniana， que asigna a cada punto $p \in M$ un producto interno positivo-definido $g_p$ en el espacio tangente $T_p M$。 El espacio tangente en un punto es el conjunto de todas las posibles direcciones en las que uno puede abandonar ese punto； es un espacio vectorial que proporciona la aproximación lineal local a la variedad。

La métrica $g$ permite definir varios conceptos fundamentales。 La longitud de una curva $\gamma$：[0,1] → M está dada por $L(\gamma) = \int_0^1 \sqrt{g_{\gamma(t)}(\dot{\gamma}(t)， \dot{\gamma}(t))} \, dt$， donde $\dot{\gamma}(t)$ es el vector velocidad de la curva。 La distancia entre dos puntos es el ínfimo de las longitudes de todas las curvas que los conectan。 El ángulo entre dos vectores en el espacio tangente se define a través del producto interno。

En Manifold， el estado latente del modelo es un punto en una variedad riemanniana aprendible。 La geometría de esta variedad（la forma en que se curva y dobla）codifica las relaciones entre diferentes estados posibles del sistema。 Dos estados que están "cerca" en la geometría de la variedad son estados que el sistema considera similares； dos estados que están "lejos" se consideran distintos。 Al aprender la geometría de la variedad， el sistema aprende la estructura del problema que está resolviendo。

**Propiedades Requeridas de la Métrica:**
- La métrica debe ser simétrica： $g_{ij} = g_{ji}$
- La métrica debe ser positiva-definida： $v^T g v > 0$ para todo $v \neq 0$
- La métrica debe ser suave（diferenciable）para permitir el cálculo de derivadas

**Lo que NO Debe Hacer:**
- ❌ No usar métricas no-definidas positivas（causan inestabilidad numérica）
- ❌ No usar métricas discontinuas（rompen el flujo geodésico）
- ❌ No asumir que la métrica euclidiana es siempre apropiada

**Lo que Sí Debe Hacer:**
- ✅ Asegurar que la métricalearned sea siempre positiva-definida
- ✅ Usar regularización para mantener la suavidad de la métrica
- ✅ Considerar la topología del problema al elegir la métrica

### 2.2 Símbolos de Christoffel

Los símbolos de Christoffel son coeficientes que describen cómo los vectores base del espacio tangente cambian de punto a punto en la variedad。 Se derivan del tensor métrico y sus derivadas：

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \frac{\partial g_{jl}}{\partial x^i} + \frac{\partial g_{il}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^l} \right)$$

donde $g^{kl}$ es la inversa del tensor métrico y $\partial_i$ denota diferenciación parcial con respecto a la i-ésima coordenada。 Los símbolos de Christoffel no son componentes de un tensor； sus valores dependen del sistema de coordenadas utilizado， aunque las cantidades geométricas que describen son independientes de las coordenadas。

En Manifold， los símbolos de Christoffel desempeñan un papel central en determinar cómo evoluciona el estado。 Codifican la "interacción" entre diferentes componentes del estado， similar a cómo los pesos de atención codifican interacciones en Transformers。 Sin embargo， los símbolos de Christoffel tienen una interpretación geométrica específica： describen cómo la geometría de la variedad causa que el estado se curve mientras se mueve。

**Formulación de Bajo Rango:**

Calcular símbolos de Christoffel desde un tensor métrico completo sería computacionalmente caro， ya que el métrico es un objeto $d \times d$ para una variedad d-dimensional。 Manifold usa una parameterización de bajo rango que reduce la complejidad computacional mientras mantiene suficiente expresividad：

$$\Gamma(v， x) \approx W \cdot \left[ (U^T v)^2 \odot \sigma(\|U^T v\|) \right]$

Aquí， $U \in \mathbb{R}^{d \times r}$ y $W \in \mathbb{R}^{d \times r}$ son matrices de bajo rango con rango $r \ll d$。 La operación cuadrática $(U^T v)^2$ captura interacciones entre componentes de velocidad， la multiplicación elemento a elemento con una función de saturación $\sigma$ proporciona estabilidad numérica， y la multiplicación final por $W$ proyecta de vuelta a la dimensión completa。

**Componentes y Dimensiones:**

| Componente | Dimensión | Función |
|------------|-----------|---------|
| $U$ | $\mathbb{R}^{d \times r}$ | Matriz base para proyección de velocidad |
| $W$ | $\mathbb{R}^{d \times r}$ | Matriz de peso para composición de curvatura |
| $(U^T v)^2$ | $\mathbb{R}^{r}$ | Operación cuadrática（interacciones de segundo orden） |
| $\sigma(\cdot)$ | $\mathbb{R}^{r}$ | Saturación suave（estabilidad numérica） |
| $r$（rango） | Entero 16-64 | Compresión de bajo rango |

**Lo que NO Debe Hacer:**
- ❌ No usar rango completo（$r = d$）sin justificación（pierde eficiencia）
- ❌ No omitir la saturación（causa inestabilidad numérica）
- ❌ No asumir que los símbolos de Christoffel son tensores（son coeficientes de conexión）

**Lo que Sí Debe Hacer:**
- ✅ Usar rango entre 16 y 64 para la mayoría de aplicaciones
- ✅ Incluir saturación suave（tanh o sigmoid）en la computation
- ✅ Verificar que la estructura preserve la simetría $\Gamma^k_{ij} = \Gamma^k_{ji}$

### 2.3 Mecánica Hamiltoniana

La mecánica hamiltoniana reformula la mecánica newtoniana en términos del espacio de fases， un espacio matemático donde el estado de un sistema se especifica mediante coordenadas de posición y momento en lugar de posición y velocidad。 Para un sistema con $n$ grados de libertad， el espacio de fases tiene dimensión $2n$， con $n$ coordenadas para posiciones $q_1， \ldots， q_n$ y $n$ coordenadas para momentos $p_1， \ldots， p_n$。

La ecuación clave de la mecánica hamiltoniana es que la evolución temporal de cualquier sistema físico puede derivarse de una sola función llamada Hamiltoniano， denotada $H(q， p)$。 Esta función representa la energía total del sistema y determina completamente su dinámica a través de las ecuaciones de Hamilton：

$$\dot{q}_i = \frac{\partial H}{\partial p_i}， \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}$$

En Manifold， el estado latente se representa como un punto（$x$， $v$） en el espacio de fases， donde $x$ representa posición（la ubicación semántica en la variedad） y $v$ representa velocidad（que sirve como proxy para momento y memoria）。 El Hamiltoniano del sistema codifica la estructura relevante para la tarea， y las dinámicas siguen de las ecuaciones de Hamilton。

**Estructura Simpléctica y Conservación:**

Una propiedad crucial de los sistemas hamiltonianos es que preservan una estructura geométrica llamada forma simpléctica。 Para un sistema con coordenadas canónicas（$q$， $p$）， la forma simpléctica es $\omega = \sum_i dq_i \wedge dp_i$， donde $\wedge$ denota el producto exterior。 Esta 2-forma se preserva bajo el flujo hamiltoniano， lo que significa que el volumen de cualquier región en el espacio de fases permanece constante mientras el sistema evoluciona。

Esta preservación de volumen tiene implicaciones profundas para el aprendizaje。 Implica que la información no puede simplemente desaparecer o aparecer； el sistema es preservador de volumen， no disipativo。 Más precisamente， el Jacobiano del flujo tiene determinante 1， lo que significa que los valores singulares del Jacobiano son exactamente todos 1。 Los gradientes no pueden desaparecer（valores singulares de 0） o explotar（valores singulares de infinito） porque hacerlo cambiaría el volumen。

**Lo que NO Debe Hacer:**
- ❌ No usar integradores no simplécticos para sistemas conservativos（pierden energía）
- ❌ No ignorar la estructura simpléctica en la optimización
- ❌ No asumir que la conservación de energía implica que el sistema no puede aprender

**Lo que Sí Debe Hacer:**
- ✅ Usar integradores simplécticos（leapfrog， forest-ruth） para estabilidad
- ✅ Considerar la estructura simpléctica en el diseño de optimizadores
- ✅ Entender que el Hamiltonianolearned evoluciona aunque el flujo sea conservativo

### 2.4 La Ecuación Geodésica

La ecuación geodésica describe cómo un punto se mueve en una variedad cuando ninguna fuerza actúa sobre él： el camino "más recto posible" a través del espacio curvado。 Para una partícula de masa unitaria moviéndose en una variedad riemanniana， la ecuación geodésica es：

$$\frac{d^2x^k}{dt^2} + \Gamma^k_{ij}(x) \frac{dx^i}{dt} \frac{dx^j}{dt} = 0$$

Esta ecuación puede entenderse como una generalización de la segunda ley de Newton。 El segundo término， involucrando símbolos de Christoffel， describe cómo la curvatura de la variedad causa que la trayectoria se curve。 En una variedad plana（donde todos los símbolos de Christoffel se anulan）， la ecuación se reduce a $\ddot{x}^k = 0$， cuyas soluciones son líneas rectas（las familiares geodésicas euclidianas）。 En una variedad curvada， los términos de Christoffel causan que la trayectoria se doble， siguiendo la geometría intrínseca del espacio。

En Manifold， la ecuación geodésica se modifica para incluir una "fuerza externa" derivada del embedding del token de entrada：

$$\frac{d^2x^k}{dt^2} + \Gamma^k_{ij}(x) \frac{dx^i}{dt} \frac{dx^j}{dt} = F^k(u_t)$$

Aquí， $F(u_t)$ es el vector de fuerza derivado del embedding del token de entrada actual。 Esta modificación significa que el estado sigue un movimiento aproximadamente geodésico en ausencia de entrada， pero es "empujado" por las fuerzas de entrada cuando se procesan los tokens。

**Lo que NO Debe Hacer:**
- ❌ No ignorar la ecuación geodésica en el diseño del modelo
- ❌ No usar integradores de primer orden（euler） para sistemas de segundo orden
- ❌ No asumir que las geodésicas son líneas rectas en espacios curvados

**Lo que Sí Debe Hacer:**
- ✅ Implementar la ecuación geodésica completa con símbolos de Christoffel
- ✅ Usar integradores de segundo orden o superior
- ✅ Considerar la curvaturalearned en la interpretación de trayectorias

---

## 3. Arquitectura del Sistema

### 3.1 Visión General de la Arquitectura

La arquitectura Manifold implementa una Red de Flujo Geométrico（GFN） donde el modelado de secuencias se reformula como dinámica de partículas en una variedad riemanniana aprendible。 La arquitectura consta de cinco subsistemas principales： la capa de embedding transforma tokens discretos en vectores de fuerza continuos； la capa M（capa manifold） implementa las dinámicas geodésicas centrales con interacciones similares a atención； la capa de integración evoluciona numéricamente el estado según principios físicos； la capa de salida produce predicciones del estado manifold； y la capa de inferencia activa proporciona modulación adaptativa de dinámicas basada en estimaciones de incertidumbre。

La arquitectura está diseñada para resolver los problemas fundamentales de las arquitecturas Transformer tradicionales。 Los Transformers mantienen un historial completo de pares Clave-Valor（$K$， $V$） y， para producir la siguiente salida， consultan toda esta memoria con una operación no-local que escala cuadráticamente en la longitud de la secuencia。 Manifold propone una alternativa： en lugar de operaciones de atención global， el modelo usa "flujo geodésico local" donde la historia causal se codifica en un estado de fase compacto（$x_t$， $v_t$）， y la "atención" se realiza evolucionando a lo largo de curvatura aprendida con integradores preservadores de estructura。

**Flujo de Datos:**

El flujo de datos de Manifold transforma secuencias de tokens discretos en dinámicas manifolds continuas y de vuelta a predicciones discretas：

```
Input Tokens [B, L]
    │
    ▼
Embedding Layer: Tokens → Force Vectors F [B, L, D]
    │
    ▼
M-Layer Stack: (x₀, v₀), F → (x₁, v₁), ..., (xₙ, vₙ) [B, L, 2D]
    │
    ▼
Readout Layer: Final State → Logits [B, L, V]
    │
    ▼
Output Predictions
```

Durante el entrenamiento， todos los L tokens se procesan en paralelo， permitiendo computación eficiente en lotes。 Durante la inferencia， los tokens se procesan autoregresivamente con persistencia de estado， manteniendo memoria O（1） independientemente de la longitud de la secuencia。

**Lo que NO Debe Hacer:**
- ❌ No mezclar el flujo de datos de entrenamiento e inferencia
- ❌ No ignorar la estructura de fase（$x$， $v$） en el diseño
- ❌ No usar arquitecturas no-preservadoras de volumen para longitudes largas

**Lo que Sí Debe Hacer:**
- ✅ Mantener la distinción entre entrenamiento（paralelo） e inferencia（secuencial）
- ✅ Preservar la estructura de fase a través de todas las capas
- ✅ Usar topología compacta（toro） para tareas de horizonte infinito

### 3.2 Representación del Estado

El estado Manifold consiste en componentes de posición y velocidad que juntos codifican el contexto completo。

**Componente de Posición ($x \in \mathbb{R}^{d}$):**

El componente de posición representa la ubicación semántica en la variedad。 Los puntos que están cerca en la geometría manifold corresponden a estados semánticamente similares。 La posición evoluciona según la ecuación geodésica， curvándose en respuesta a la curvatura manifold codificada en los símbolos de Christoffel。

**Componente de Velocidad ($v \in \mathbb{R}^{d}$):**

El componente de velocidad representa el momento del sistema， que sirve como el mecanismo de memoria。 A diferencia de las tiendas de memoria explícitas en otras arquitecturas， el momento codifica el contexto implícitamente a través del historial de interacciones。 Alto momento indica fuerte preservación del estado anterior； bajo momento indica que el contexto reciente domina。

**Componente de Fuerza ($F \in \mathbb{R}^{d}$):**

El componente de fuerza se deriva del embedding del token de entrada y actúa como una entrada externa que empuja el sistema a través de la variedad。 Diferentes tokens producen diferentes vectores de fuerza， causando que el estado se mueva en diferentes direcciones a través del espacio semántico。

**Lo que NO Debe Hacer:**
- ❌ No normalizar la velocidad a cero（destruye la memoria）
- ❌ No permitir que la velocidad crezca sin límites（causa inestabilidad）
- ❌ No ignorar la correlación entre $x$ y $v$

**Lo que Sí Debe Hacer:**
- ✅ Usar normalización de velocidad suave（no hard clipping）
- ✅ Implementar límites suaves en la velocidad（tanh o similar）
- ✅ Considerar la correlación posición-velocidad en la interpretación

### 3.3 Jerarquía de Configuración

Manifold usa un sistema de configuración jerárquica donde cada nivel corresponde a un subsistema mayor：

```python
config = {
    'embedding': {...},       # Configuración de embedding
    'readout': {...},         # Configuración de lectura
    'active_inference': {...}, # Dinámicas adaptativas
    'fractal': {...},         # Estructura jerárquica
    'topology': {...},        # Estructura global
    'stability': {...}        # Parámetros numéricos
}
```

Esta estructura asegura consistencia entre componentes mientras permite control granular。 Cada subsistema valida su configuración durante la inicialización y proporciona mensajes de error significativos para configuraciones inválidas。

**Parámetros Críticos de Configuración:**

| Parámetro | Rango Recomendado | Efecto |
|-----------|-------------------|--------|
| `dim` | 128 - 512 | Dimensión del modelo |
| `depth` | 4 - 12 | Número de capas M |
| `heads` | 4 - 8 | Número de cabezas geométricas |
| `base_dt` | 0.05 - 0.4 | Paso de tiempo base |
| `rank` | 16 - 64 | Rango de Christoffel |

**Lo que NO Debe Hacer:**
- ❌ No usar `base_dt` mayor a 0.5（causa inestabilidad）
- ❌ No usar `dim` menor a 64（limita capacidad）
- ❌ No ignorar la validación de configuración

**Lo que Sí Debe Hacer:**
- ✅ Usar validación de configuración en producción
- ✅ Ajustar `base_dt` según la tarea（tareas más suaves permiten dt más grande）
- ✅ Documentar configuraciones no estándar

---

## 4. Componentes del Modelo

### 4.1 Capa de Embedding

La capa de embedding transforma tokens discretos en vectores de fuerza continuos que impulsan el sistema a través de la variedad。 Manifold implementa varios tipos de embeddings：

**Embedding Funcional (SIREN):**

El embedding funcional usa redes sinusoidales（Redes de Representación Sinusoidal） para generar embeddings：

```python
class FunctionalEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, coord_dim=16, mode='linear'):
        """
        Args:
            vocab_size: Número de tokens en el vocabulario
            emb_dim: Dimensión de salida del embedding
            coord_dim: Dimensión de coordenada para el campo neural
            mode: 'linear' o 'binary' para codificación de coordenadas
        """
```

El modo lineal produce interpolación suave entre representaciones de token：

```python
coords = token_ids.float() / (vocab_size - 1)  # Normalizar a [0, 1]
coords = coords * 2 - 1  # Escalar a [-1, 1]
```

El modo binario produce coordenadas binarias discretas：

```python
binary = token_ids.unsqueeze(-1).bitwise_and(
    2 ** torch.arange(coord_dim, device=token_ids.device)
) > 0
```

**Embedding Implícito:**

El embedding implícito usa una tabla de coordenadas aprendibles：

```python
class ImplicitEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, coord_dim=16, learn_coords=True):
        # Tabla de coordenadas aprendibles
        self.coordinates = nn.Parameter(torch.randn(vocab_size, coord_dim) * 0.1)
        # Proyección de coordenada a embedding
        self.proj = nn.Linear(coord_dim, emb_dim)
```

**Lo que NO Debe Hacer:**
- ❌ No usar embeddings de lookup para vocabularios grandes（escala O(V)）
- ❌ No usar modo binario para tareas que requieren interpolación suave
- ❌ No inicializar pesos SIREN incorrectamente（causa entrenamiento lento）

**Lo que Sí Debe Hacer:**
- ✅ Usar embeddings funcionales para vocabularios grandes
- ✅ Usar modo lineal para generalización out-of-distribution
- ✅ Seguir la inicialización SIREN específica（ω₀ = 30 para primera capa）

### 4.2 Capa M (Manifold Layer)

La capa M implementa las dinámicas geodésicas centrales， reemplazando el mecanismo de atención en Transformers：

```python
class ManifoldLayer(nn.Module):
    def __init__(self, dim, num_heads, integrator_type='leapfrog', 
                 physics_config=None, dropout=0.0):
        """
        Args:
            dim: Dimensión del modelo
            num_heads: Número de cabezas de atención
            integrator_type: Tipo de integrador simpléctico
            physics_config: Configuración de parámetros físicos
            dropout: Tasa de dropout
        """
```

**Arquitectura Multi-Cabeza:**

La arquitectura multi-cabeza procesa el estado a través de flujos geodésicos independientes：

```python
class MultiHeadManifold(nn.Module):
    def forward(self, x, v, F):
        # Dividir en cabezas
        x_heads = x.view(B, L, self.num_heads, self.head_dim)
        v_heads = v.view(B, L, self.num_heads, self.head_dim)
        F_heads = F.view(B, L, self.num_heads, self.head_dim)
        
        # Procesar cada cabeza independientemente
        for h in range(self.num_heads):
            # Computar símbolos de Christoffel
            christoffel = self.christoffel_heads[h](v_h, x_h)
            
            # Computar compuerta de fricción
            gate = torch.sigmoid(self.gate_heads[h](x_h))
            friction = gate * 5.0 * v_h
            
            # Fuerza neta
            net_force = F_h - christoffel - friction
            
            # Actualizar velocidad y posición
            v_h = v_h + 0.5 * net_force
            x_h = x_h + 0.4 * v_h
            
            # Normalizar velocidad（crítico para estabilidad）
            v_h = v_h / (v_h.norm(dim=-1, keepdim=True) + 1e-6)
```

**Lo que NO Debe Hacer:**
- ❌ No omitir la normalización de velocidad（causa explosión）
- ❌ No usar más de 8 cabezas sin justificación（overhead）
- ❌ No ignorar la correlación entre cabezas

**Lo que Sí Debe Hacer:**
- ✅ Incluir normalización de velocidad después de cada actualización
- ✅ Usar 4-8 cabezas para la mayoría de tareas
- ✅ Considerar mezcla de cabezas para compartir información

### 4.3 Módulo Christoffel

El módulo Christoffel implementa la parameterización de bajo rango：

```python
class ChristoffelModule(nn.Module):
    def __init__(self, dim, rank=16):
        super().__init__()
        self.dim = dim
        self.rank = rank
        
        # Parameterización de bajo rango
        self.U = nn.Linear(dim, rank, bias=False)
        self.W = nn.Linear(rank, dim, bias=False)
        
        # Red de compuerta para gating adaptativo
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1)
        )
        
        # Red de modulación de posición
        self.pos_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1)
        )
    
    def forward(self, v, x):
        # Proyección de velocidad de bajo rango
        v_proj = self.U(v)  # [B, L, rank]
        v_squared = v_proj ** 2  # Interacción cuadrática
        
        # Saturación para estabilidad numérica
        v_norm = v.norm(dim=-1, keepdim=True)
        saturation = torch.sigmoid(v_norm / 10.0)
        
        # Símbolos de Christoffel base
        Gamma_base = self.W(v_squared * saturation)
        
        # Gating adaptativo
        gate = torch.sigmoid(self.gate_net(x))
        Gamma_gated = gate * Gamma_base
        
        # Modulación de posición
        pos_scale = 1.0 + torch.tanh(self.pos_net(x))
        Gamma = Gamma_gated * pos_scale
        
        # Clamping para estabilidad
        Gamma = torch.clamp(Gamma, -5.0, 5.0)
        
        return Gamma
```

**Fórmula de Bajo Rango:**

La computación implementa：

$$\Gamma(v， x) = W \cdot \left[ (U^T v)^2 \odot \sigma(\|v\|) \right] \odot \text{gate}(x) \odot \text{pos\_mod}(x)$$

**Lo que NO Debe Hacer:**
- ❌ No usar bias en capas U o W（rompe la simetría）
- ❌ No omitir la saturación（causa inestabilidad）
- ❌ No usar clamping duro（crea discontinuidades en gradientes）

**Lo que Sí Debe Hacer:**
- ✅ Usar `bias=False` en U y W
- ✅ Incluir saturación suave（sigmoid o tanh）
- ✅ Usar clamping suave（tanh） en lugar de hard clipping

### 4.4 Capa de Readout

La capa de readout decodifica el estado manifold final a predicciones：

**Readout Implícito:**

```python
class ImplicitReadout(nn.Module):
    def __init__(self, dim, vocab_size, coord_dim=16):
        super().__init__()
        self.coord_proj = nn.Linear(dim, coord_dim)
        # SIREN inverso para mapeo de coordenada a logit
        self.inverse_siren = SIREN(coord_dim, vocab_size // 2, vocab_size)
    
    def forward(self, x_final):
        # Proyectar a espacio de coordenadas
        coords = self.coord_proj(x_final)
        coords = torch.tanh(coords)  # Normalizar a rango válido
        # Evaluar campo inverso
        logits = self.inverse_siren(coords)
        return logits
```

**Readout Explícito:**

```python
class ExplicitReadout(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size)
    
    def forward(self, x_final):
        return self.proj(x_final)
```

**Lo que NO Debe Hacer:**
- ❌ No usar readout explícito para vocabularios muy grandes（memoria）
- ❌ No usar readout implícito sin suficiente capacidad（underfitting）
- ❌ No olvidar la normalización en readout implícito

**Lo que Sí Debe Hacer:**
- ✅ Usar readout explícito para vocabularios pequeños（menos overhead）
- ✅ Usar readout implícito para vocabularios grandes o desconocidos
- ✅ Incluir normalización（tanh） en coordenadas de readout implícito

---

## 5. Física y Dinámica del Sistema

### 5.1 Dinámica Completa

Las dinámicas completas del sistema， incluyendo fricción termodinámica， son：

$$\frac{dx}{dt} = v$$
$$\frac{dv}{dt} = F_{\text{token}} - \Gamma(v， x) - F_{\text{fricción}}(v， x)$$

donde $F_{\text{token}}$ es la fuerza del embedding del token， $\Gamma(v， x)$ es la aceleración geométrica de los símbolos de Christoffel， y $F_{\text{fricción}}$ es la fuerza de fricción。

**Interpretación Física:**

El primer término（$F_{\text{token}}$） representa la输入 del mundo exterior al sistema。 Cada token de entrada "empuja" el estado en una dirección específica， codificando información semántica。

El segundo término（$-\Gamma(v， x)$） representa la "resistencia" del espacio curvado。 A diferencia de un espacio plano donde la partícula continuaría en línea recta， en una variedad curvada la trayectoria se dobla naturalmente， codificando interacciones entre componentes del estado。

El tercer término（$-F_{\text{fricción}}$） representa la disipación de energía que permite "olvidar" información antigua y escribir nueva información。

**Lo que NO Debe Hacer:**
- ❌ No ignorar ninguno de los tres términos（el sistema no funcionaría correctamente）
- ❌ No permitir fricción negativa（causa ganancia de energía no física）
- ❌ No usar $\Gamma$ sin considerar la simetría（$\Gamma^k_{ij} = \Gamma^k_{ji}$）

**Lo que Sí Debe Hacer:**
- ✅ Implementar los tres términos de manera balanceada
- ✅ Usar fricción positiva（sigmoid） para garantizar disipación
- ✅ Verificar la simetría de Christoffel en la implementación

### 5.2 Fricción Termodinámica (El Embrague)

El término de fricción permite la conmutación entre modos de memoria y computación：

$$F_{\text{fricción}} = -\mu(x， u) \cdot v$$

donde el coeficiente de fricción se aprende como：

$$\mu(x， u) = \text{sigmoid}(W_{\text{gate}} \cdot [\sin x， \cos x]) \cdot \mu_{\text{max}}$$

**Régimen de Fricción Alta（Embrague Engagement）:**

Cuando $\mu \gg 1$， el sistema actúa como un acumulador de primer orden， absorbiendo fuerzas de entrada en la coordenada espacial $x$。 Este es el régimen de **Computación Activa**， donde el estado se actualiza rápidamente para reflejar nueva información。

**Régimen de Cero Fricción（Embrague Desengagement）:**

Cuando $\mu \approx 0$， el sistema actúa como un flujo conservador de segundo orden， permitiendo que el estado "coaste" a través de la variedad mediante inercia。 Este es el régimen de **Persistencia de Información**， donde la memoria se preserva sin decaimiento。

**Regímenes de Razonamiento:**

El modelo optimiza sus parámetros para que el sistema opere en dos estados físicos distintos：

1. **Régimen de Alta Disipación:** Cuando $\mu \gg 1$， el sistema actúa como acumulador de primer orden， absorbiendo fuerzas de entrada en la coordenada espacial $x$。 Este es el régimen de Computación Activa。

2. **Régimen de Cero Disipación:** Cuando $\mu \approx 0$， el sistema actúa como flujo conservador， permitiendo que el estado coaste a través de la variedad mediante inercia。 Este es el régimen de Persistencia de Información。

El rendimiento cognitivo es una función de la capacidad del modelo para "enganchar el embrague" solo cuando información simbólica relevante está presente， y "desengancharlo" para preservar esa información a lo largo del tiempo。 En la práctica， esto aparece como picos escasos， dependientes del estado， en $\mu_{\theta}$ más que como amortiguamiento constante。

**Lo que NO Debe Hacer:**
- ❌ No usar fricción constante（pierde la ventaja de conmutación de régimen）
- ❌ No permitir fricción mayor a 5.0（amortiguación excesiva）
- ❌ No usar activación lineal para $\mu$（puede volverse negativa）

**Lo que Sí Debe Hacer:**
- ✅ Usar sigmoid para garantizar $\mu > 0$
- ✅ Limitar $\mu_{\text{max}}$ a un rango razonable（1.0 - 5.0）
- ✅ Incluir características periódicas $[\sin x， \cos x]$ para topología tórica

### 5.3 Curvatura Reactiva

La curvatura reactiva modula la geometría basada en la energía del modelo， medida por la energía cinética del pensamiento：

$$K = \frac{1}{2} \|v\|^2$$

El escalar de plasticidad se define como：

$$\lambda(K) = \alpha \cdot \tanh(K)$$

La conexión efectiva incluye modulación de plasticidad：

$$\Gamma_{\text{efectiva}} = \Gamma_{\text{base}} \cdot (1 + \lambda(K))$$

**Interpretación:**

Cuando el modelo está "confuso"（oscilación/velocidad alta）：

- El espacio se vuelve viscoso
- La curvatura aumenta
- El proceso de "frena" automáticamente
- Se integra más información antes de proceder

**Lo que NO Debe Hacer:**
- ❌ No usar plasticidad sin límite（puede causar curvatura excesiva）
- ❌ No confundir plasticidad con fricción（son mecanismos diferentes）
- ❌ No usar valores de plasticidad mayores a 1.0（demasiada modulación）

**Lo que Sí Debe Hacer:**
- ✅ Usar tanh para limitar plasticidad a [0， $\alpha$]
- ✅ Elegir $\alpha$ basado en la tarea（0.1 - 0.5 típicamente）
- ✅ Considerar plasticidad como mecanismo de "frenado" suave

### 5.4 Singularidades Lógicas

Las singularidades representan decisiones lógicas discretas como atractores topológicos en espacio continuo。 Esto permite que un sistema puramente continuo represente operaciones lógicas discretas。

Cuando la curvatura local excede un umbral crítico， el sistema "abre" una sub-variedad：

$$x_{\text{macro}} \xrightarrow{\mathcal{R} > \tau} x_{\text{micro}}$$

**Configuración de Singularidades:**

```python
singularities = {
    'enabled': True,
    'strength': 20.0,    # Fuerza de atracción
    'threshold': 0.8     # Activación
}
```

**Lo que NO Debe Hacer:**
- ❌ No habilitar singularidades sin entender su efecto（pueden causar inestabilidad）
- ❌ No usar fortalezas mayores a 50.0（pueden dominar la dinámica）
- ❌ No confundir singularidades con discontinuidades（son suaves）

**Lo que Sí Debe Hacer:**
- ✅ Usar umbrales entre 0.5 y 0.9
- ✅ Usar fortalezas entre 2.0 y 20.0
- ✅ Verificar que las singularidades sean suaves（diferenciables）

---

## 6. Integración Numérica

### 6.1 Esquema Leapfrog (Velocity Verlet)

El integrador principal de Manifold implementa el esquema Leapfrog de segundo orden：

$$v_{n+\frac{1}{2}} = v_n + \frac{1}{2} \Delta t \cdot a_n$$
$$x_{n+1} = x_n + \Delta t \cdot v_{n+\frac{1}{2}}$$
$$v_{n+1} = v_{n+\frac{1}{2}} + \frac{1}{2} \Delta t \cdot a_{n+1}$$

donde $a_n = F_n - \Gamma(x_n， v_n) - \mu_n v_n$ es la aceleración。

**Implementación Completa:**

```python
def leapfrog_step(x, v, F, christoffel, dt, gate_activ):
    # Coeficiente de fricción aprendido
    mu = torch.sigmoid(gate_activ) * 5.0
    
    # Aceleración en tiempo t
    gamma = christoffel(v, x)
    friction = mu * v
    a_t = F - gamma - friction
    
    # Paso medio en velocidad
    v_half = v + 0.5 * dt * a_t
    
    # Paso completo en posición
    x_next = x + dt * v_half
    
    # Aceleración en tiempo t+1
    gamma_next = christoffel(v_half, x_next)
    friction_next = mu * v_half
    a_next = F - gamma_next - friction_next
    
    # Paso medio en velocidad final
    v_next = v_half + 0.5 * dt * a_next
    
    # Normalización de velocidad（crítico para estabilidad）
    v_next = v_next / (||v_next|| + \epsilon)
    
    return x_next, v_next
```

**Propiedades del Integrador:**

| Propiedad | Descripción | Implicación |
|-----------|-------------|-------------|
| Tiempo-reversible | Simétrico bajo t → -t | Estabilidad numérica |
| Volumen-preservador | $\det\left(\frac{\partial(x'， v')}{\partial(x， v)}\right) = 1$ | Gradientes estables |
| Error local | $O(\Delta t^3)$ | Precisión por paso |
| Error global | $O(\Delta t^2)$ | Precisión acumulada |

**Lo que NO Debe Hacer:**
- ❌ No usar integradores de primer orden（euler） para sistemas hamiltonianos
- ❌ No omitir la normalización de velocidad（causa inestabilidad）
- ❌ No usar pasos de tiempo mayores a 0.5（pérdida de precisión）

**Lo que Sí Debe Hacer:**
- ✅ Usar leapfrog como integrador predeterminado
- ✅ Incluir normalización de velocidad después de cada paso
- ✅ Mantener $dt \leq 0.4$ para estabilidad

### 6.2 Integradores de Alto Orden

**Forest-Ruth (4to Orden Simpléctico):**

El integrador Forest-Ruth proporciona precisión superior para tareas de razonamiento complejas：

$$\theta = \frac{1}{2 - 2^{1/3}}$$
$$\lambda = 1 - 2 \cdot \theta$$
$$\chi = -\frac{2^{1/3}}{2(2 - 2^{1/3})}$$

**Comparación de Integradores:**

| Integrador | Orden | Simpléctico | Error de Energía | Velocidad |
|------------|-------|-------------|------------------|-----------|
| Euler | 1 | No | Alto | Rápido |
| Heun | 2 | No | Medio | Rápido |
| Leapfrog | 2 | Sí | Bajo | Rápido |
| Forest-Ruth | 4 | Sí | Muy Bajo | Medio |
| Yoshida | 4 | Sí | Muy Bajo | Medio |
| RK4 | 4 | No | Bajo（puede divergir） | Lento |

**La Paradoja Runge-Kutta:**

Un descubrimiento clave en los benchmarks es que Runge-Kutta de alto orden（RK4） diverge instantáneamente en Manifold， mientras que métodos simplécticos de menor orden permanecen estables。

**Explicación (Aliasing de Singularidades):**

1. **Topología Discontinua:** Características como singularidades lógicas crean un campo de fuerza no suave（$C^2$ o menos）。

2. **Error Multi-Etapa:** RK4 evalúa 4 etapas por paso。 Si una etapa intermedia evalúa una posición dentro de una singularidad de alta curvatura， el polinomio de 4to orden sobre-extrapolará la fuerza。

3. **Criterio de Estabilidad:** Métodos de menor orden（Heun， Euler， Leapfrog） son más "locales" y no intentan modelar derivadas de alto orden de un campo que no es suave。

**Lo que NO Debe Hacer:**
- ❌ No usar RK4 para tareas con singularidades o discontinuidades
- ❌ No asumir que mayor orden siempre significa mejor precisión
- ❌ No ignorar las propiedades simplécticas en la selección de integrador

**Lo que Sí Debe Hacer:**
- ✅ Usar leapfrog para la mayoría de tareas（mejor relación precisión/estabilidad）
- ✅ Usar forest-ruth para tareas suaves que requieren alta precisión
- ✅ Considerar la topología del problema al seleccionar integrador

### 6.3 Condiciones de Frontera Tóricas

Para topología tórica， las coordenadas se envuelven periódicamente：

$$x \leftarrow x \bmod 2\pi$$

**Distancia Toroidal:**

Para objetivos que viven en $T^n$， usar la distancia angular más corta：

$$d_{\text{torus}}(x_1， x_2) = \min\!\big(|\Delta|,\ 2\pi - |\Delta|\big)，\quad \Delta = x_1-x_2$$

**Pérdida de Fase:**

La pérdida de fase suave para gradientes cerca del envolvente：

$$L_{\text{phase}} = 1 - \cos(x_{\text{pred}} - x_{\text{target}})$$

**Lo que NO Debehacer:**
- ❌ No usar distancia euclidiana para coordenadas periódicas
- ❌ No ignorar el envolvente en la pérdida（crea gradientes incorrectos）
- ❌ No usar $x$ directamente en compuertas（debe usar $[\sin x， \cos x]$）

**Lo que Sí Debe Hacer:**
- ✅ Usar características periódicas $[\sin x， \cos x]$ en compuertas
- ✅ Usar distancia toroidal para objetivos periódicos
- ✅ Considerar la pérdida de fase para supervisión suave

---

## 7. Optimización Riemanniana

### 7.1 El Problema

Las actualizaciones euclidianas estándar violan las restricciones de la variedad aprendible：

$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \nabla \mathcal{L}$$

pueden producir matrices que violan propiedades necesarias para mantener una estructura de Christoffel válida。

### 7.2 Solución: Retracciones

Las retracciones mapean actualizaciones euclidianas de vuelta a la variedad：

$$W_{\text{new}} = \text{Retract}_M(W_{\text{old}} - \eta \cdot \nabla \mathcal{L})$$

**Tipos de Retracción:**

| Tipo | Fórmula | Caso de Uso |
|------|---------|-------------|
| Normalize | $W \cdot \min(1， \text{max\_norm}/\|W\|)$ | **Recomendado** |
| Cayley | $(I - \frac{1}{2} A)^{-1}(I + \frac{1}{2} A)$ | Matrices ortogonales |
| Exponencial | $\exp(A)$ | Variedades de matrices |
| Tórica | $W \bmod 2\pi$ | Coordenadas periódicas |

**Implementación de Retracción Normalize:**

```python
def retraction_normalize(W, max_norm=10.0):
    norm = torch.norm(W)
    scale = min(1.0, max_norm / (norm + 1e-8))
    return W * scale
```

**RiemannianAdam:**

```python
class RiemannianAdam(Optimizer):
    def __init__(self, params, lr=1e-3, retraction='normalize', max_norm=10.0):
        defaults = dict(lr=lr, retraction=retraction, max_norm=max_norm)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Inicializar estado si es necesario
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                # Actualizar momentos（igual que Adam）
                exp_avg.mul_(beta1).add_(grad， alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad * grad， alpha=1 - beta2)
                
                # Corrección de bias
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Dirección de paso
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                update = exp_avg / (exp_avg_sq.sqrt() + eps)
                
                # Aplicar retracción
                if retraction == 'normalize':
                    p.data.add_(update， alpha=-step_size)
                    norm = p.data.norm()
                    if norm > max_norm:
                        p.data.mul_(max_norm / norm)
```

**Lo que NO Debe Hacer:**
- ❌ No usar optimizadores euclidianos（AdamW） sin retracción
- ❌ No omitir la normalización después de la actualización
- ❌ No usar retracción tórica para parámetros no periódicos

**Lo que Sí Debe Hacer:**
- ✅ Usar RiemannianAdam con retracción 'normalize' para pesos estándar
- ✅ Usar retracción 'torus' para coordenadas de posición
- ✅ Mantener `max_norm` en un rango razonable（5.0 - 20.0）

### 7.3 Clip de Gradientes

Aunque la estructura simpléctica de Manifold previene la explosión de gradientes por geometría， el clip de gradientes se aplica como medida de seguridad：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
```

El umbral de clip de 0.05 es más estricto que los valores típicos（0.1-1.0） usados con arquitecturas estándar。 Este clip más estricto refleja las diferentes dinámicas de gradientes de los sistemas geométricos y asegura estabilidad numérica en casos extremos。

**Lo que NO Debe Hacer:**
- ❌ No usar umbrales de clip muy grandes（pierde la ventaja de seguridad）
- ❌ No omitir el clip de gradientes completamente

**Lo que Sí Debe Hacer:**
- ✅ Usar umbral de 0.05 para Manifold（más estricto que lo estándar）
- ✅ Considerar clip por-nivel para componentes con diferentes escalas

---

## 8. Configuración y Uso

### 8.1 Configuración Completa de Producción

**Configuración Recomendada para Tareas de Razonamiento:**

```python
physics_config = {
    'embedding': {
        'type': 'functional',
        'mode': 'linear',
        'coord_dim': 16
    },
    'readout': {
        'type': 'implicit',
        'coord_dim': 16
    },
    'active_inference': {
        'enabled': True,
        'dynamic_time': {
            'enabled': True
        },
        'reactive_curvature': {
            'enabled': True,
            'plasticity': 0.2
        },
        'singularities': {
            'enabled': False,
            'strength': 2.0,
            'threshold': 0.8
        },
        'hysteresis': {
            'enabled': False
        }
    },
    'hierarchical_curvature': {
        'enabled': True,
        'ranks': [8, 16, 32]
    },
    'topology': {
        'type': 'torus'
    },
    'stability': {
        'base_dt': 0.05,
        'curvature_clamp': 5.0,
        'velocity_friction_scale': 0.1
    }
}

loss_config = {
    'lambda_h': 0.01,
    'lambda_g': 0.001,
    'hamiltonian_mode': 'adaptive',
    'geodesic_mode': 'structural'
}
```

### 8.2 Parámetros Críticos y sus Efectos

**Parámetros de Física:**

| Parámetro | Rango | Efecto |
|-----------|-------|--------|
| `base_dt` | 0.01 - 0.4 | Precisión vs velocidad de integración |
| `plasticity` | 0.0 - 0.5 | Intensidad de curvatura reactiva |
| `singularity_strength` | 1.0 - 50.0 | Fuerza de atracción de singularidades |
| `friction_scale` | 0.1 - 5.0 | Máxima fricción del embrague |

**Parámetros de Arquitectura:**

| Parámetro | Rango | Efecto |
|-----------|-------|--------|
| `dim` | 64 - 512 | Capacidad del modelo |
| `depth` | 2 - 12 | Profundidad de procesamiento |
| `heads` | 1 - 8 | Número de canales geométricos |
| `rank` | 8 - 64 | Expresividad de Christoffel |

### 8.3 Configuración para Diferentes Tareas

**Tareas Algorítmicas (Paridad, Aritmética):**

```python
config = {
    'topology': {'type': 'torus'},
    'active_inference': {
        'dynamic_time': {'enabled': True},
        'singularities': {'enabled': True, 'strength': 10.0, 'threshold': 0.7}
    }
}
```

**Tareas de Lenguaje (General):**

```python
config = {
    'topology': {'type': 'euclidean'},
    'active_inference': {
        'dynamic_time': {'enabled': False},
        'singularities': {'enabled': False}
    },
    'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2}
}
```

**Tareas de Memoria Larga:**

```python
config = {
    'topology': {'type': 'torus'},
    'active_inference': {
        'hysteresis': {'enabled': True, 'strength': 1.0}
    },
    'stability': {
        'base_dt': 0.1  # dt más grande para coasting
    }
}
```

**Lo que NO Debe Hacer:**
- ❌ No usar topología tórica para tareas sin estructura periódica
- ❌ No habilitar singularidades sin-tuning（pueden causar inestabilidad）
- ❌ No usar `base_dt` mayor a 0.4 en producción

**Lo que Sí Debe Hacer:**
- ✅ Elegir topología basada en la estructura de la tarea
- ✅ Hacer tuning gradual de parámetros activos
- ✅ Empezar con configuraciones conservadoras y ajustar progresivamente

---

## 9. Antipatrrones y Errores Comunes

### 9.1 Errores de Física Fundamental

**Antipatrón 1: Ignorar la Estructura Simpléctica**

❌ **No Hacer:**

```python
# ¡ERROR! Euler no preserva volumen
x_next = x + dt * v
v_next = v + dt * F
```

✅ **Hacer:**

```python
# Leapfrog preserva volumen simpléctico
v_half = v + 0.5 * dt * F
x_next = x + dt * v_half
v_next = v_half + 0.5 * dt * F
```

**Antipatrón 2: Fricción Negativa**

❌ **No Hacer:**

```python
# ¡ERROR! Fricción negativa = ganancia de energía
mu = W_gate @ x  # Puede ser negativo
friction = mu * v
```

✅ **Hacer:**

```python
# Fricción siempre positiva
mu = torch.sigmoid(W_gate @ x) * 5.0
friction = mu * v
```

**Antipatrón 3: Normalización de Velocidad Incorrecta**

❌ **No Hacer:**

```python
# ¡ERROR! Hard clipping destruye gradientes
v = torch.clamp(v, -10, 10)
```

✅ **Hacer:**

```python
# Normalización suave preserva gradientes
v_norm = v.norm(dim=-1, keepdim=True)
v = v / (v_norm + 1e-6)
```

### 9.2 Errores de Configuración

**Antipatrón 4: Paso de Tiempo Muy Grande**

❌ **No Hacer:**

```python
config = {'stability': {'base_dt': 1.0}}  # ¡Inestable!
```

✅ **Hacer:**

```python
config = {'stability': {'base_dt': 0.05}}  # Estable
```

**Antipatrón 5: Demasiadas Singularidades**

❌ **No Hacer:**

```python
config = {
    'active_inference': {
        'singularities': {
            'enabled': True,
            'strength': 100.0,  # ¡Demasiado fuerte!
            'threshold': 0.99
        }
    }
}
```

✅ **Hacer:**

```python
config = {
    'active_inference': {
        'singularities': {
            'enabled': True,
            'strength': 5.0,  # Razonable
            'threshold': 0.7  # Activación temprana
        }
    }
}
```

**Antipatrón 6: Topología Incorrecta**

❌ **No Hacer:**

```python
# Usar topología tórica para texto general sin estructura periódica
config = {'topology': {'type': 'torus'}}
```

✅ **Hacer:**

```python
# Usar topología euclidiana para tareas sin estructura periódica
config = {'topology': {'type': 'euclidean'}}
```

### 9.3 Errores de Implementación

**Antipatrón 7: Christoffel Sin Saturación**

❌ **No Hacer:**

```python
# ¡ERROR! Sin límites en curvatura
Gamma = W @ (U.T @ v) ** 2
```

✅ **Hacer:**

```python
# Saturación suave para estabilidad
v_proj = U.T @ v
v_norm = v_proj.norm()
saturation = torch.sigmoid(v_norm / 10.0)
Gamma = W @ (v_proj ** 2 * saturation)
Gamma = torch.clamp(Gamma, -5.0, 5.0)
```

**Antipatrón 8: Gating Sin Características Periódicas**

❌ **No Hacer:**

```python
# ¡ERROR! No usar características periódicas en toro
mu = torch.sigmoid(W_gate @ x)
```

✅ **Hacer:**

```python
# Características periódicas para continuidad en frontera
x_features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
mu = torch.sigmoid(W_gate @ x_features)
```

### 9.4 Errores de Optimización

**Antipatrón 9: AdamW Sin Retracción**

❌ **No Hacer:**

```python
# ¡ERROR! AdamW viola restricciones de variedad
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

✅ **Hacer:**

```python
# RiemannianAdam con retracción apropiada
optimizer = RiemannianAdam(model.parameters(), lr=1e-3, retraction='normalize')
```

**Antipatrón 10: Clip de Gradientes Excesivo**

❌ **No Hacer:**

```python
# ¡ERROR! Clip muy permisivo
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

✅ **Hacer:**

```python
# Clip conservativo para sistemas geométricos
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
```

---

## 10. Verificación y Testing

### 10.1 Test de Conservación de Energía

Verificar que la energía se conserve en sistemas sin fricción：

```python
def test_energy_conservation():
    # Configurar modelo sin fricción
    model = Manifold(vocab_size=2, physics_config={
        'stability': {'friction': 0.0}
    })
    
    # Medir energía inicial
    H_initial = compute_hamiltonian(x, v, model)
    
    # Ejecutar 1000 steps sin gradientes
    with torch.no_grad():
        for _ in range(1000):
            x, v = leapfrog_step(x, v, force, model)
    
    # Medir energía final
    H_final = compute_hamiltonian(x, v, model)
    
    # Verificar conservación
    energy_drift = abs(H_final - H_initial) / H_initial
    assert energy_drift < 0.0001, f"Energy drift: {energy_drift}"
    print(f"✓ Energy drift: {energy_drift:.6f} (< 0.0001)")
```

**Criterio de Aprobación:** Deriva de energía < 0.01%

### 10.2 Test de Volumen Simpléctico

Verificar que el volumen en espacio de fases se preserve：

```python
def test_symplectic_volume():
    # Sample many points
    x_samples = torch.randn(1000, dim)
    v_samples = torch.randn(1000, dim)
    
    # Compute initial volume (using covariance)
    initial_cov = torch.cov(torch.cat([x_samples, v_samples], dim=-1).T)
    initial_det = torch.det(initial_cov)
    
    # Evolve points
    with torch.no_grad():
        for _ in range(100):
            x_samples, v_samples = leapfrog_batch(x_samples, v_samples, F)
    
    # Compute final volume
    final_cov = torch.cov(torch.cat([x_samples, v_samples], dim=-1).T)
    final_det = torch.det(final_cov)
    
    # Check volume preservation
    volume_ratio = final_det / initial_det
    assert 0.9 < volume_ratio < 1.1, f"Volume ratio: {volume_ratio}"
    print(f"✓ Volume preservation ratio: {volume_ratio:.4f} (≈ 1.0)")
```

**Criterio de Aprobación:** Ratio de volumen en [0.9, 1.1]

### 10.3 Test de Convergencia Monotónica

Verificar que la pérdida sea monotónicamente decreciente：

```python
def test_convergence_monotonic():
    losses = []
    for epoch in range(100):
        loss = train_epoch(model, optimizer, dataloader)
        losses.append(loss)
    
    # Check monotonicity over windows
    window_size = 20
    for i in range(window_size, len(losses)):
        window = losses[i-window_size:i]
        trend = np.polyfit(range(window_size), window, 1)[0]
        assert trend < 0, f"Non-decreasing trend at epoch {i}"
    
    print("✓ Loss curve is monotonically decreasing")
```

**Criterio de Aprobación:** Pendiente negativa en todas las ventanas de 20 epochs

### 10.4 Test de Estabilidad Numérica

Verificar que no haya NaN o Inf：

```python
def test_numerical_stability():
    for batch in dataloader:
        logits, state, trajectory = model(batch)
        
        # Check for NaN/Inf
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"
        assert not torch.isnan(state[0]).any(), "NaN in position"
        assert not torch.isnan(state[1]).any(), "NaN in velocity"
    
    print("✓ No numerical instability detected")
```

**Criterio de Aprobación:** Sin NaN o Inf en 1000 batches

### 10.5 Test de Topología Tórica

Verificar el comportamiento correcto de la topología：

```python
def test_toroidal_topology():
    # Test boundary wrapping
    x = torch.tensor([6.2, 6.3, 6.28])
    x_wrapped = wrap_toroidal(x)
    assert torch.allclose(x_wrapped, x - 2 * np.pi, atol=1e-5)
    
    # Test periodicity
    x_periodic = torch.randn(10, dim)
    x_shifted = (x_periodic + 2 * np.pi) % (2 * np.pi)
    dist_original = toroidal_dist(x_periodic, x_periodic)
    dist_shifted = toroidal_dist(x_periodic, x_shifted)
    assert torch.allclose(dist_original, dist_shifted, atol=1e-5)
    
    print("✓ Toroidal topology behaves correctly")
```

**Criterio de Aprobación:** Distancia toroidal periódica con tolerancia 1e-5

---

## 11. Apéndices

### 11.1 Glosario de Términos

| Término | Definición |
|---------|------------|
| Variedad | Espacio matemático localmente plano pero globalmente curvado |
| Métrica | Función que define distancias y ángulos en la variedad |
| Símbolos de Christoffel | Coeficientes que describen la conexión afín de la variedad |
| Conexión | Regla para transportar vectores entre espacios tangentes |
| Hamiltoniano | Función que determina la dinámica del sistema |
| Simpléctico | Estructura que preserva el volumen en el espacio de fases |
| Geodésica | Camino más corto entre dos puntos en una variedad |
| Retracción | Mapeo que proyecta actualizaciones a la variedad |
| Embrague | Mecanismo de fricción que conmuta entre memoria y computación |

### 11.2 Referencias Bibliográficas

**Fundamentos Matemáticos:**

- Arnold, V. I. (1989). Mathematical Methods of Classical Mechanics. Springer.
- Riemann, B. (1854). On the Hypotheses Which Lie at the Bases of Geometry.
- Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration. Springer.

**Aprendizaje Profundo Geométrico:**

- Bronstein, M. M., et al. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv.
- Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations. NeurIPS.

**Arquitecturas de Secuencia:**

- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
- Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv.

### 11.3 Notas sobre Correcciones de Documentos

Los siguientes documentos/papers contienen errores conocidos que deben tenerse en cuenta：

1. **derivaciones.tex:** Algunas ecuaciones de Christoffel tienen errores de índice。 Verificar siempre contra la implementación de código。

2. **ALL_PAPERS.tex:** La convención de índices no es consistente entre papers。 Usar la convención del código fuente como referencia。

3. **GFN_PAPER.md:** El valor de $\omega_0$ para SIREN debería ser 30， no 10 como se indica en algunas secciones。

### 11.4 Registro de Cambios del Documento

| Versión | Fecha | Descripción |
|---------|-------|-------------|
| 1.0.0 | 2026-02-07 | Versión inicial del documento completo |

---

**Fin del Documento**

*Este documento es la referencia canonical para el proyecto Manifold。 Para actualizaciones， consultar el repositorio oficial。*
