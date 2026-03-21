# GFN Matrix Results: Hierarchical_Tree

| dynamics_type | integrator | topology_type | Accuracy | VRAM (MB) | Peak VRAM (MB) | Params | Duration (s) | Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | leapfrog | euclidean | 50.0% | 16.65 | 18.32 | 44,783 | 16.77 | 0.5274 |
| residual | leapfrog | euclidean | 50.0% | 16.65 | 18.39 | 44,791 | 18.59 | 0.3041 |
| direct | leapfrog | hyperbolic | 50.0% | 16.67 | 18.86 | 46,959 | 22.76 | 0.4202 |
| residual | leapfrog | hyperbolic | 50.0% | 16.68 | 18.93 | 46,967 | 23.57 | 0.2144 |
| direct | yoshida | hyperbolic | 50.0% | 16.67 | 19.29 | 46,959 | 32.00 | 0.4189 |
| residual | yoshida | hyperbolic | 50.0% | 16.68 | 19.36 | 46,967 | 35.05 | 0.2119 |
| residual | leapfrog | hierarchical | 50.0% | 16.82 | 19.28 | 65,291 | 42.71 | 0.4010 |
| direct | yoshida | hierarchical | 50.0% | 16.81 | 19.60 | 65,283 | 54.31 | 0.4394 |
| residual | yoshida | hierarchical | 50.0% | 16.82 | 19.68 | 65,291 | 56.46 | 0.3821 |
| residual | yoshida | euclidean | 37.5% | 16.65 | 18.55 | 44,791 | 25.51 | 0.2104 |
| direct | leapfrog | hierarchical | 25.0% | 16.81 | 19.20 | 65,283 | 38.19 | 0.4278 |
| direct | yoshida | euclidean | 0.0% | 16.65 | 18.48 | 44,783 | 25.34 | 6027.5776 |
