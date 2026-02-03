#ifndef GFN_CUDA_CHRISTOFFEL_IMPL_CUH
#define GFN_CUDA_CHRISTOFFEL_IMPL_CUH

#include "../common/types.cuh"
#include "../common/device_utils.cuh"
#include "../common/math_utils.cuh"

namespace gfn {
namespace cuda {

// ============================================================================
// Core Christoffel Device Function
// ============================================================================

/**
 * Core Christoffel symbol computation using low-rank decomposition.
 * 
 * Computes: Γ(v,v) = Σ_r (h_r^2 * W_r) * S * M
 * where:
 *   h = U^T * v (projection to rank-R space)
 *   S = 1 / (1 + ||h||)  (stabilization factor)
 *   M = modulation from plasticity and singularities
 * 
 * @param v Velocity vector [dim]
 * @param U Low-rank matrix U [dim x rank]
 * @param W Low-rank matrix W [dim x rank]
 * @param x Position vector [dim] (optional, for friction/singularities)
 * @param V_w Potential weights [dim] (optional, for singularities)
 * @param dim Dimension of manifold
 * @param rank Rank of decomposition
 * @param plasticity Plasticity coefficient (energy-dependent curvature)
 * @param sing_thresh Singularity threshold
 * @param sing_strength Singularity strength multiplier
 * @param topology Topology type (EUCLIDEAN or TORUS)
 * @param R Toroidal major radius
 * @param r Toroidal minor radius
 * @param gamma Output Christoffel force [dim]
 */
GFN_DEVICE void christoffel_device(
    const scalar_t* v,
    const scalar_t* U,
    const scalar_t* W,
    const scalar_t* x,
    const scalar_t* V_w,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* gamma
) {
    // Thread-local storage for intermediate computations
    scalar_t h[64];  // Projection to rank space (max rank = 64)
    scalar_t h_sq[64]; // Squared projections
    
    // 1. Project velocity to rank space: h = U^T * v
    for (int i = 0; i < rank; ++i) {
        scalar_t sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            sum += U[j * rank + i] * v[j];
        }
        h[i] = sum;
    }
    
    // 2. Compute energy and stabilization factor
    // Match Python EXACTLY: scale = 1.0 / (1.0 + norm + 1e-4)
    scalar_t energy = 0.0f;
    for (int i = 0; i < rank; ++i) {
        energy += h[i] * h[i];
    }
    
    scalar_t norm = sqrt(energy);
    scalar_t S = 1.0f / (1.0f + norm + 1e-4f);  // Exact Python epsilon
    
    // 3. Compute modulation factor M
    scalar_t M = 1.0f;
    
    // 3a. Plasticity (energy-dependent curvature)
    if (plasticity != 0.0f) {
        scalar_t v_energy = 0.0f;
        for (int i = 0; i < dim; ++i) {
            v_energy += v[i] * v[i];
        }
        v_energy /= static_cast<scalar_t>(dim);
        M *= (1.0 + plasticity * tanh(v_energy));
    }
    
    // 3b. Singularities (position-dependent)
    if (x != nullptr && V_w != nullptr) {
        scalar_t pot = 0.0f;
        
        if (topology == Topology::TORUS) {
            // Use periodic features for torus
            for (int i = 0; i < dim; ++i) {
                pot += sin(x[i]) * V_w[i];
            }
        } else {
            // Linear potential for Euclidean
            for (int i = 0; i < dim; ++i) {
                pot += x[i] * V_w[i];
            }
        }
        
        scalar_t gate = sigmoid(pot);
        scalar_t soft_m = sigmoid(10.0f * (gate - sing_thresh));
        M *= (1.0f + (sing_strength - 1.0f) * soft_m);
    }
    
    // 4. Compute h^2 * S * M
    for (int i = 0; i < rank; ++i) {
        h_sq[i] = h[i] * h[i] * S * M;
    }
    
    // 5. Project back to manifold space: gamma = W * h_sq
    for (int i = 0; i < dim; ++i) {
        scalar_t sum = 0.0f;
        for (int j = 0; j < rank; ++j) {
            sum += W[i * rank + j] * h_sq[j];
        }
        gamma[i] = sum;
    }
    
    // 6. Apply soft clamping
    for (int i = 0; i < dim; ++i) {
        gamma[i] = soft_clamp(gamma[i], CURVATURE_CLAMP);
    }
}

// ============================================================================
// Friction Computation
// ============================================================================

/**
 * Compute friction coefficient from position and force.
 * 
 * μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE
 * 
 * @param x Position vector [dim]
 * @param force External force [dim] (optional)
 * @param W_forget Forget gate weights [dim x feature_dim]
 * @param b_forget Forget gate bias [dim]
 * @param W_input Input gate weights [dim x dim] (optional)
 * @param dim Dimension
 * @param topology Topology type
 * @param friction Output friction coefficients [dim]
 */
GFN_DEVICE void compute_friction(
    const scalar_t* x,
    const scalar_t* force,
    const scalar_t* W_forget,
    const scalar_t* b_forget,
    const scalar_t* W_input,
    int dim,
    Topology topology,
    scalar_t* friction
) {
    scalar_t features[128]; // Max 2*dim for Fourier features
    int feature_dim = dim;
    
    // Compute features based on topology
    if (topology == Topology::TORUS) {
        // Fourier features: [sin(x), cos(x)]
        compute_fourier_features(features, x, dim);
        feature_dim = 2 * dim;
    } else {
        // Direct features
        vector_copy(features, x, dim);
    }
    
    // Compute gate activation: W_f * features + b_f
    for (int i = 0; i < dim; ++i) {
        scalar_t gate_val = b_forget[i];
        for (int j = 0; j < feature_dim; ++j) {
            gate_val += W_forget[i * feature_dim + j] * features[j];
        }
        
        // Add force component if available
        if (force != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) {
                gate_val += W_input[i * dim + j] * force[j];
            }
        }
        
        friction[i] = sigmoid(gate_val) * FRICTION_SCALE;
    }
}

// ============================================================================
// Combined Christoffel + Friction
// ============================================================================

/**
 * Compute combined Christoffel force with friction.
 * Output: Γ(v,v) + μ(x,F) * v
 */
GFN_DEVICE void christoffel_with_friction(
    const scalar_t* v,
    const scalar_t* U,
    const scalar_t* W,
    const scalar_t* x,
    const scalar_t* V_w,
    const scalar_t* force,
    const scalar_t* W_forget,
    const scalar_t* b_forget,
    const scalar_t* W_input,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* output
) {
    scalar_t gamma[64];
    scalar_t friction[64];
    
    // Compute Christoffel force
    christoffel_device(
        v, U, W, x, V_w, dim, rank,
        plasticity, sing_thresh, sing_strength,
        topology, R, r, gamma
    );
    
    // Compute friction if position is available
    if (x != nullptr && W_forget != nullptr && b_forget != nullptr) {
        compute_friction(
            x, force, W_forget, b_forget, W_input,
            dim, topology, friction
        );
        
        // Add friction term: gamma + μ * v
        for (int i = 0; i < dim; ++i) {
            output[i] = gamma[i] + friction[i] * v[i];
        }
    } else {
        // No friction, just copy gamma
        vector_copy(output, gamma, dim);
    }
}

// ============================================================================
// Separate Christoffel and Friction (for implicit integration)
// ============================================================================

/**
 * Compute Christoffel and friction separately.
 * Used for implicit integration schemes (e.g., Leapfrog).
 */
GFN_DEVICE void christoffel_friction_separate(
    const scalar_t* v,
    const scalar_t* U,
    const scalar_t* W,
    const scalar_t* x,
    const scalar_t* V_w,
    const scalar_t* force,
    const scalar_t* W_forget,
    const scalar_t* b_forget,
    const scalar_t* W_input,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* gamma,
    scalar_t* friction
) {
    // Compute Christoffel force
    christoffel_device(
        v, U, W, x, V_w, dim, rank,
        plasticity, sing_thresh, sing_strength,
        topology, R, r, gamma
    );
    
    // Compute friction if position is available
    if (x != nullptr && W_forget != nullptr && b_forget != nullptr) {
        compute_friction(
            x, force, W_forget, b_forget, W_input,
            dim, topology, friction
        );
    } else {
        // No friction
        vector_zero(friction, dim);
    }
}

/**
 * Backward pass for the Christoffel symbol computation.
 * 
 * Computes gradients: dL/dv, dL/dU, dL/dW, dL/dx.
 * 
 * @param grad_out Grad output from Loss [dim]
 * @param gamma Forward output Γ(v,v) [dim] (needed for soft_clamp derivative)
 * @param v Forward velocity vector [dim]
 * @param U Forward matrix U [dim x rank]
 * @param W Forward matrix W [dim x rank]
 * @param x Forward position vector [dim]
 * @param V_w Potential weights [dim]
 * @param dim Dimension
 * @param rank Rank
 * @param plasticity Plasticity coefficient
 * @param topology Topology type
 * @param grad_v Output gradient w.r.t v [dim]
 * @param grad_U Output gradient w.r.t U [dim x rank]
 * @param grad_W Output gradient w.r.t W [dim x rank]
 * @param grad_x Output gradient w.r.t x [dim] (optional)
 */
GFN_DEVICE void christoffel_backward_device(
    const scalar_t* grad_out,
    const scalar_t* gamma,
    const scalar_t* v,
    const scalar_t* U,
    const scalar_t* W,
    const scalar_t* x,
    const scalar_t* V_w,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* grad_v,
    scalar_t* grad_U,
    scalar_t* grad_W,
    scalar_t* grad_x
) {
    // Thread-local storage (max rank 64, max dim 128 supported)
    scalar_t h[64];
    scalar_t grad_h[64];
    scalar_t grad_q[64];  // Gradient w.r.t h^2
    
    // 1. Re-compute forward state h = U^T * v and modulate factor M
    scalar_t h_energy = 0.0;
    for (int i = 0; i < rank; ++i) {
        scalar_t sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += U[j * rank + i] * v[j];
        }
        h[i] = sum;
        h_energy += sum * sum;
    }
    
    scalar_t norm = sqrt(h_energy);
    scalar_t S = 1.0f / (1.0f + norm + 1e-4f);
    
    // 1b. Re-compute Modulation M
    scalar_t M_plas = 1.0;
    scalar_t v_energy = 0.0;
    if (plasticity != 0.0) {
        for (int i = 0; i < dim; ++i) v_energy += v[i] * v[i];
        v_energy /= static_cast<scalar_t>(dim);
        M_plas = (1.0 + plasticity * tanh(v_energy));
    }
    
    scalar_t M_sing = 1.0;
    scalar_t gate = 0.0, soft_m = 0.0;
    if (x != nullptr && V_w != nullptr) {
        scalar_t pot = 0.0;
        if (topology == Topology::TORUS) {
            for (int i = 0; i < dim; ++i) pot += sin(x[i]) * V_w[i];
        } else {
            for (int i = 0; i < dim; ++i) pot += x[i] * V_w[i];
        }
        gate = sigmoid(pot);
        soft_m = sigmoid(10.0f * (gate - sing_thresh));
        M_sing = (1.0f + (sing_strength - 1.0f) * soft_m);
    }
    scalar_t M = M_plas * M_sing;

    // 2. Correct grad_out for soft_clamp derivative
    // d_clamped / d_raw = 1 - (clamped / 20)^2
    scalar_t grad_raw[128];
    for (int i = 0; i < dim; ++i) {
        scalar_t t = gamma[i] / CURVATURE_CLAMP;
        grad_raw[i] = grad_out[i] * (1.0 - t * t);
    }
    
    // 3. Gradient w.r.t W and intermediate grad_q
    for (int j = 0; j < rank; ++j) {
        grad_q[j] = 0.0;
        scalar_t q_base = h[j] * h[j] * S * M;
        for (int i = 0; i < dim; ++i) {
            grad_W[i * rank + j] += grad_raw[i] * q_base;
            grad_q[j] += W[i * rank + j] * grad_raw[i];
        }
    }
    
    // 4. Gradient w.r.t h (accounting for stabilization S)
    scalar_t sum_grad_q_h_sq = 0.0;
    for (int i = 0; i < rank; ++i) {
        sum_grad_q_h_sq += grad_q[i] * h[i] * h[i];
    }
    
    scalar_t S_sq_M_norm = (norm > 1e-8) ? (M * S * S / norm) : 0.0;
    scalar_t two_S_M = 2.0 * S * M;
    
    for (int i = 0; i < rank; ++i) {
        grad_h[i] = grad_q[i] * h[i] * two_S_M - sum_grad_q_h_sq * S_sq_M_norm * h[i];
    }
    
    // 5. Gradient w.r.t U and v (projection part)
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < rank; ++j) {
            grad_U[i * rank + j] += v[i] * grad_h[j];
            grad_v[i] += U[i * rank + j] * grad_h[j];
        }
    }
    
    // 6. Plasticity contribution to grad_v
    if (plasticity != 0.0) {
        // dL/dM_plas = sum_i grad_q[i] * h[i]^2 * S * M_sing
        scalar_t dL_dM_plas = sum_grad_q_h_sq * S * M_sing;
        scalar_t sech_sq = 1.0 - tanh(v_energy) * tanh(v_energy);
        scalar_t factor = dL_dM_plas * plasticity * sech_sq * (2.0 / static_cast<scalar_t>(dim));
        for (int i = 0; i < dim; ++i) {
            grad_v[i] += factor * v[i];
        }
    }
    
    // 7. Singularity contribution to grad_x
    if (grad_x != nullptr && x != nullptr && V_w != nullptr) {
        scalar_t dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
        scalar_t dM_dsoft = (sing_strength - 1.0);
        scalar_t dsoft_dgate = 10.0f * soft_m * (1.0f - soft_m);
        scalar_t dgate_dpot = gate * (1.0f - gate);
        scalar_t factor = dL_dM_sing * dM_dsoft * dsoft_dgate * dgate_dpot;
        
        for (int i = 0; i < dim; ++i) {
            scalar_t dpot_dxi = (topology == Topology::TORUS) ? cos(x[i]) * V_w[i] : V_w[i];
            grad_x[i] += factor * dpot_dxi;
        }
    }
}

/**
 * Backward pass for friction computation.
 * 
 * Computes gradients: dL/dW_forget, dL/db_forget, dL/dx.
 */
GFN_DEVICE void friction_backward_device(
    const scalar_t* grad_out,
    const scalar_t* x,
    const scalar_t* force,
    const scalar_t* W_forget,
    const scalar_t* b_forget,
    const scalar_t* W_input,
    int dim,
    Topology topology,
    scalar_t* grad_W_forget,
    scalar_t* grad_b_forget,
    scalar_t* grad_W_input,
    scalar_t* grad_x,
    scalar_t* grad_force
) {
    scalar_t features[128];
    int feature_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    
    // Re-compute features
    if (topology == Topology::TORUS) {
        compute_fourier_features(features, x, dim);
    } else {
        vector_copy(features, x, dim);
    }
    
    // Gradient w.r.t. gate pre-activation z: mu = sigmoid(z) * scale
    for (int i = 0; i < dim; ++i) {
        scalar_t z = b_forget[i];
        for (int j = 0; j < feature_dim; ++j) {
            z += W_forget[i * feature_dim + j] * features[j];
        }
        
        if (force != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) {
                z += W_input[i * dim + j] * force[j];
            }
        }
        
        scalar_t s = sigmoid(z);
        scalar_t dz = grad_out[i] * FRICTION_SCALE * s * (1.0f - s);
        
        // Accumulate gradients for weights and bias
        grad_b_forget[i] += dz;
        for (int j = 0; j < feature_dim; ++j) {
            grad_W_forget[i * feature_dim + j] += dz * features[j];
        }
        
        // Force and W_input gradients
        if (force != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) {
                grad_W_input[i * dim + j] += dz * force[j];
                if (grad_force != nullptr) {
                    grad_force[j] += dz * W_input[i * dim + j];
                }
            }
        }
        
        // Propagate to features
        if (topology == Topology::TORUS) {
            for (int j = 0; j < dim; ++j) {
                scalar_t d_sin = W_forget[i * feature_dim + j] * dz;
                scalar_t d_cos = W_forget[i * feature_dim + (dim + j)] * dz;
                grad_x[j] += d_sin * cos(x[j]) - d_cos * sin(x[j]);
            }
        } else {
            for (int j = 0; j < dim; ++j) {
                grad_x[j] += W_forget[i * feature_dim + j] * dz;
            }
        }
    }
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_CHRISTOFFEL_IMPL_CUH
