# GFN Dynamics: Pure Geodetic Flow vs G-SSM

Este documento formaliza la transición del ISN hacia flujos geodésicos puros y responde a las interrogantes sobre la conservación de energía y la estructura de orden del sistema.

## 1. La Partícula en el Manifold

En el `GFNWorld`, el estado interno no es una memoria estadística, sino una **partícula** evolucionando en un manifold diferenciable $M$.

### Ecuación de Movimiento (Flujo ISN)
$$\dot{x} = \text{Drift}(x) + \text{Diffusion}(u)$$

Donde:
- $x \in M$ es el estado del mundo.
- $u$ es el impulso externo (token procesado por el `Scanner`).
- **Drift ($\text{Drift}(x)$)**: Representa la inercia interna. En un sistema físico, si no hay fuerzas externas ($u=0$), la partícula sigue una geodésica. El drift asegura que el mundo tenga "continuidad" y no colapse ante la ausencia de estímulos.
- **Diffusion ($\text{Diffusion}(u)$)**: Es el acoplamiento entre la señal externa y la geometría interna.

## 2. Conservación de Energía vs Disipación

¿Conserva energía la partícula?
- **En G-SSM**: El sistema está diseñado para preservar normas (estructuras ortogonales/unitarias) para evitar el desvanecimiento del gradiente. Es esencialmente un sistema conservativo en el sentido de la norma.
- **En GFN (ISN)**: Introducimos la noción de **Energía del Sistema** como $E = \|x\|^2$.
  - Si el `Drift` es una matriz antisimétrica ($A = -A^T$), el sistema conserva la energía de forma natural (rotaciones puras).
  - En la práctica de lenguaje, aplicamos una ligera disipación coordinada con la inyección de energía de los nuevos tokens para mantener el sistema en un **estado estacionario dinámico**.

## 3. ¿Es una 2nd Order SSM?

La respuesta corta es: **Sí, funcionalmente.**

- **G-SSM (1-SSM)**: Mapea $u \to x$ como un sistema de primer orden lineal.
- **GFN ISN**: Al operar en un espacio de embedding de alta dimensión, el vector de estado $x$ se comporta como un **vector de fase** $z = [q, p]$, donde $q$ es la posición y $p$ es el momento.
- Aunque la implementación sea $\dot{x} = f(x, u)$ (primer orden en $x$), la complejidad de la red de `Drift` permite que el sistema aprenda dinámicas de **segundo orden** (donde la aceleración depende de la posición anterior). Esto es lo que permite que el GFN tenga "memoria de largo plazo" sin necesidad de mecanismos de atención cuadrática.

## 4. Diferencias Clave

| Característica | G-SSM | GFN (ISN Pure) |
| :--- | :--- | :--- |
| **Geometría** | Riemanniana Proyectiva | Flujo Geodésico General |
| **Naturaleza** | Filtro Lineal (Recurrente) | Sistema Dinámico No Lineal |
| **Orden** | 1er Orden | Mapeo de 2do Orden (Fase) |
| **Contexto** | Escalamiento de Norma | Invariancia de Flujo |

"La partícula no recuerda el pasado; simplemente fluye por la curvatura que el pasado dejó en el manifold."
