# GFN Matrix Results: Mini_NLP

| dynamics_type | mixer_type | topology_type | Accuracy | VRAM (MB) | Peak VRAM (MB) | Params | Duration (s) | Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual | low_rank | euclidean | 31.6% | 16.47 | 19.49 | 24,759 | 55.25 | 2.0167 |
| residual | geodesic_attention | euclidean | 31.6% | 16.47 | 19.49 | 24,759 | 55.38 | 2.0167 |
| residual | flow_mixer | euclidean | 31.6% | 16.47 | 19.49 | 24,759 | 55.94 | 2.0167 |
| direct | flow_mixer | torus | 31.6% | 16.57 | 21.92 | 39,603 | 236.57 | 0.0703 |
| direct | geodesic_attention | torus | 31.6% | 16.57 | 21.92 | 39,603 | 236.66 | 0.0703 |
| direct | low_rank | torus | 28.1% | 16.57 | 21.92 | 39,603 | 232.83 | 0.0636 |
| direct | flow_mixer | euclidean | 24.6% | 16.46 | 19.29 | 24,755 | 50.33 | 1.2256 |
| direct | geodesic_attention | euclidean | 24.6% | 16.46 | 19.29 | 24,755 | 50.46 | 1.2256 |
| direct | low_rank | euclidean | 24.6% | 16.46 | 19.29 | 24,755 | 51.84 | 1.2256 |
| residual | flow_mixer | torus | 24.6% | 16.57 | 22.64 | 39,607 | 247.24 | 0.8962 |
| residual | geodesic_attention | torus | 24.6% | 16.57 | 22.64 | 39,607 | 247.72 | 0.8962 |
| residual | low_rank | torus | 24.6% | 16.57 | 22.64 | 39,607 | 253.44 | 0.8962 |
