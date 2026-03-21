# GFN Matrix Results: XOR_Logic

| dynamics_type | integrator | topology_type | Accuracy | VRAM (MB) | Peak VRAM (MB) | Params | Duration (s) | Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual | yoshida | torus | 75.0% | 16.36 | 17.58 | 11,729 | 5.99 | 0.6955 |
| residual | rk4 | torus | 75.0% | 16.36 | 17.58 | 11,729 | 6.86 | 0.6956 |
| residual | yoshida | euclidean | 50.0% | 16.33 | 17.47 | 7,313 | 4.50 | 0.6945 |
| mix | yoshida | euclidean | 50.0% | 16.33 | 17.49 | 7,317 | 4.59 | 0.7209 |
| residual | rk4 | euclidean | 50.0% | 16.33 | 17.49 | 7,313 | 5.01 | 0.6945 |
| mix | rk4 | euclidean | 50.0% | 16.33 | 17.51 | 7,317 | 5.42 | 0.7209 |
| direct | yoshida | euclidean | 25.0% | 16.32 | 17.46 | 7,309 | 4.36 | 0.7910 |
| direct | rk4 | euclidean | 25.0% | 16.32 | 17.48 | 7,309 | 4.66 | 0.7902 |
| direct | rk4 | torus | 100.0% | 16.36 | 17.55 | 11,725 | 6.08 | 0.0002 |
| mix | yoshida | torus | 100.0% | 16.36 | 17.61 | 11,733 | 6.70 | 0.7228 |
| direct | yoshida | torus | 100.0% | 16.36 | 17.56 | 11,725 | 7.28 | 0.0006 |
| mix | rk4 | torus | 100.0% | 16.36 | 17.61 | 11,733 | 7.74 | 0.7228 |
