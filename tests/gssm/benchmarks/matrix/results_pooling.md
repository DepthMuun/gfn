# GFN Matrix Results: Pooling_ROI

| pooling_type | readout_type | topology_type | Accuracy | VRAM (MB) | Peak VRAM (MB) | Params | Duration (s) | Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hierarchical | categorical | euclidean | 6.2% | 16.49 | 17.98 | 23,829 | 21.33 | 0.0000 |
| hierarchical | categorical | torus | 6.2% | 16.58 | 18.71 | 37,141 | 97.56 | 2.7113 |
| None | categorical | torus | 6.2% | 16.55 | 19.63 | 37,141 | 135.18 | 2.0512 |
| momentum | categorical | euclidean | 12.5% | 16.49 | 17.95 | 23,829 | 20.25 | 0.0001 |
| hamiltonian | categorical | euclidean | 12.5% | 16.49 | 17.95 | 23,829 | 24.30 | 0.0000 |
| None | categorical | euclidean | 12.5% | 16.45 | 18.32 | 23,829 | 30.62 | 0.0000 |
| momentum | categorical | torus | 12.5% | 16.59 | 18.66 | 37,141 | 92.69 | 2.2491 |
| hamiltonian | categorical | torus | 12.5% | 16.58 | 18.68 | 37,141 | 100.02 | 2.5175 |
