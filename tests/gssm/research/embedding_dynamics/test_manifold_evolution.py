"""
Embedding Dynamics Analysis — GSSM Research Tests
===================================================
Deep analysis of how embedding space evolves during forward pass.
Tests actual GSSM model behavior, not mocks.

Research Questions:
- How do token embeddings deform through layers?
- Is there semantic compression or expansion?
- What's the trajectory of embeddings in latent space?
"""

import torch
import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from test_reporter import ResultsReporter
import gfn


class TestEmbeddingDynamics:
    """Analyze embedding space evolution through layers."""

    def test_embedding_trajectory_analysis(self):
        """
        Track embedding trajectories through layers.
        Tests: Do embeddings follow smooth trajectories?
        """
        reporter = ResultsReporter("embedding_trajectory", "embedding_dynamics")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model.eval()
            
            batch_size = 8
            seq_len = 16
            x = torch.randint(0, 128, (batch_size, seq_len))
            
            with torch.no_grad():
                logits, (xf, vf), info = model(x)
                x_seq = info['x_seq']
            
            trajectory_lengths = []
            for b in range(batch_size):
                for s in range(seq_len):
                    positions = x_seq[b, :s+1, 0, :].cpu()
                    
                    if len(positions) > 1:
                        diffs = torch.diff(positions, dim=0)
                        path_length = torch.norm(diffs, dim=1).sum().item()
                        direct_distance = torch.norm(positions[-1] - positions[0]).item()
                        
                        trajectory_lengths.append({
                            'path_length': path_length,
                            'direct_distance': direct_distance,
                            'efficiency': direct_distance / (path_length + 1e-8)
                        })
            
            avg_path_length = np.mean([t['path_length'] for t in trajectory_lengths])
            avg_direct = np.mean([t['direct_distance'] for t in trajectory_lengths])
            avg_efficiency = np.mean([t['efficiency'] for t in trajectory_lengths])
            
            reporter.log_metric("avg_path_length", avg_path_length)
            reporter.log_metric("avg_direct_distance", avg_direct)
            reporter.log_metric("avg_trajectory_efficiency", avg_efficiency)
            
            if avg_efficiency > 0.1:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Very inefficient trajectories: {avg_efficiency}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_embedding_space_curvature(self):
        """
        Measure local curvature of embedding manifold.
        Tests: Is embedding space locally flat or curved?
        """
        reporter = ResultsReporter("embedding_curvature", "embedding_dynamics")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model.eval()
            
            x = torch.randint(0, 128, (32, 8))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_final = info['x_final']
            
            x_flat = x_final.reshape(x_final.shape[0], -1)
            
            distances = torch.cdist(x_flat, x_flat)
            
            neighbor_distances = []
            for i in range(len(x_flat)):
                sorted_dists, _ = torch.sort(distances[i])
                neighbor_dists = sorted_dists[1:6].tolist()
                neighbor_distances.extend(neighbor_dists)
            
            neighbor_distances = np.array(neighbor_distances)
            mean_neighbor_dist = np.mean(neighbor_distances)
            std_neighbor_dist = np.std(neighbor_distances)
            
            curvature_estimate = std_neighbor_dist / (mean_neighbor_dist + 1e-8)
            
            reporter.log_metric("mean_neighbor_distance", mean_neighbor_dist)
            reporter.log_metric("std_neighbor_distance", std_neighbor_dist)
            reporter.log_metric("curvature_estimate", curvature_estimate)
            
            if curvature_estimate < 2.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"High curvature: {curvature_estimate}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_embedding_temporal_evolution(self):
        """
        Analyze how embedding statistics change over sequence positions.
        Tests: Is there temporal accumulation of information?
        """
        reporter = ResultsReporter("embedding_temporal", "embedding_dynamics")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model.eval()
            
            x = torch.randint(0, 128, (16, 16))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_seq = info['x_seq']
            
            position_stats = []
            for pos in range(x_seq.shape[1]):
                x_pos = x_seq[:, pos].flatten()
                position_stats.append({
                    'mean': x_pos.mean().item(),
                    'std': x_pos.std().item(),
                    'norm': torch.norm(x_pos).item()
                })
            
            first_pos_std = position_stats[0]['std']
            last_pos_std = position_stats[-1]['std']
            
            reporter.log_metric("first_position_std", first_pos_std)
            reporter.log_metric("last_position_std", last_pos_std)
            reporter.log_metric("std_change_ratio", last_pos_std / (first_pos_std + 1e-8))
            
            reporter.log_metric("first_position_mean", position_stats[0]['mean'])
            reporter.log_metric("last_position_mean", position_stats[-1]['mean'])
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_embedding_clustering_quality(self):
        """
        Measure within-cluster vs between-cluster distances.
        Tests: Are similar tokens clustered in embedding space?
        """
        reporter = ResultsReporter("embedding_clustering", "embedding_dynamics")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model.eval()
            
            # Create token IDs - each row has different tokens
            token_ids = torch.randint(0, 16, (8, 16))
            
            with torch.no_grad():
                _, (xf, vf), info = model(token_ids)
                x_final = info['x_final']
            
            x_flat = x_final.reshape(x_final.shape[0], -1)
            
            # Compute cluster centers
            cluster_centers = []
            for token_id in range(16):
                mask = (token_ids == token_id).any(dim=1)  # Rows containing this token
                if mask.sum() > 0:
                    cluster_emb = x_flat[mask]
                    cluster_centers.append(cluster_emb.mean(dim=0))
            
            if len(cluster_centers) > 1:
                center_matrix = torch.stack(cluster_centers)
                between_cluster_dist = torch.cdist(center_matrix, center_matrix)
                
                within_cluster_dists = []
                for token_id in range(16):
                    mask = (token_ids == token_id).any(dim=1)
                    if mask.sum() > 1:
                        cluster_emb = x_flat[mask]
                        within = torch.cdist(cluster_emb, cluster_emb)
                        within_cluster_dists.append(within.mean().item())
                
                avg_within = np.mean(within_cluster_dists) if within_cluster_dists else 0
                avg_between = between_cluster_dist.mean().item()
                
                ratio = avg_between / (avg_within + 1e-8)
                
                reporter.log_metric("avg_within_cluster_dist", avg_within)
                reporter.log_metric("avg_between_cluster_dist", avg_between)
                reporter.log_metric("clustering_ratio", ratio)
                
                if ratio > 1.0:
                    reporter.mark_passed(True)
                else:
                    reporter.add_error(f"Poor clustering: {ratio}")
            else:
                reporter.add_error("Not enough clusters")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
