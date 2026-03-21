# GFN Matrix Results: Arithmetic_Sum

| dynamics_type | integrator | topology_type | Accuracy | VRAM (MB) | Peak VRAM (MB) | Params | Duration (s) | Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | leapfrog | euclidean | 18.8% | 16.55 | 18.08 | 34,054 | 8.87 | 0.0000 |
| residual | leapfrog | euclidean | 18.8% | 16.56 | 18.13 | 34,060 | 9.46 | 0.0004 |
| direct | yoshida | euclidean | 18.8% | 16.55 | 18.19 | 34,054 | 13.25 | 0.0011 |
| residual | yoshida | euclidean | 18.8% | 16.56 | 18.25 | 34,060 | 14.43 | 0.0006 |
| direct | rk4 | euclidean | 18.8% | 16.55 | 18.32 | 34,054 | 15.79 | 0.0001 |
| residual | rk4 | euclidean | 18.8% | 16.56 | 18.37 | 34,060 | 16.46 | 0.0005 |
| residual | yoshida | torus | 18.8% | 16.75 | 19.11 | 60,044 | 20.02 | 0.0013 |
| residual | rk4 | torus | 18.8% | 16.75 | 19.06 | 60,044 | 23.01 | 0.9776 |
| direct | leapfrog | torus | 12.5% | 16.74 | 18.65 | 60,038 | 12.29 | 1.3219 |
| residual | leapfrog | torus | 12.5% | 16.75 | 18.82 | 60,044 | 13.89 | 0.1795 |
| direct | yoshida | torus | 12.5% | 16.74 | 18.93 | 60,038 | 19.90 | 0.2365 |
| direct | rk4 | torus | 12.5% | 16.74 | 18.89 | 60,038 | 20.16 | 3.0667 |
