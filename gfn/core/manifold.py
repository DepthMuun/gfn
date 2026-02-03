import torch
import torch.nn as nn
from ..layers import MLayer, ParallelMLayer
from ..readouts import ImplicitReadout
from ..model.state import ManifoldState
from ..model.fusion import CUDAFusionManager


class Manifold(nn.Module):
    """
    Manifold sequence model that evolves (x, v) via geodesic flow.
    
    Pipeline:
        1. Embed tokens into forces
        2. Apply M-Layers to update (x, v)
        3. Project positions to logits
    
    Args:
        vocab_size: Size of vocabulary
        dim: Hidden dimension (default: 256)
        depth: Number of M-Layers (default: 4)
        rank: Low-rank Christoffel approximation (default: 32)
        heads: Number of independent geodesic heads (default: 4)
        integrator_type: 'heun', 'rk4', or 'symplectic' (default: 'heun')
    
    Example:
        >>> model = Manifold(vocab_size=16, dim=512, depth=12, integrator_type='heun')
        >>> logits, state = model(input_ids)
    """
    
    
    def __init__(self, vocab_size, dim=256, depth=4, rank=32, heads=4, integrator_type='heun', base_dt=1.0, use_scan=False, physics_config=None, impulse_scale=None, holographic=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.integrator_type = integrator_type
        self.use_scan = use_scan
        self.physics_config = physics_config or {}
        self.holographic = holographic or self.physics_config.get('holographic', False)
        
        emb_cfg = self.physics_config.get('embedding', {})
        emb_type = emb_cfg.get('type', 'standard')
        
        if emb_type == 'implicit':
            from ..embeddings import ImplicitEmbedding
            coord_dim = emb_cfg.get('coord_dim', 16)
            self.embedding = ImplicitEmbedding(vocab_size, dim, coord_dim=coord_dim)
        elif emb_type == 'functional':
            from ..embeddings import FunctionalEmbedding
            coord_dim = emb_cfg.get('coord_dim', 16)
            mode = emb_cfg.get('mode', 'binary') 
            imp = impulse_scale if impulse_scale is not None else emb_cfg.get('impulse_scale', 1.0)
            omega_0 = emb_cfg.get('omega_0', 30.0)
            self.embedding = FunctionalEmbedding(vocab_size, dim, coord_dim=coord_dim, mode=mode, impulse_scale=imp, omega_0=omega_0)
        else:
            self.embedding = nn.Embedding(vocab_size, dim)
        
        self.layers = nn.ModuleList()
        for idx in range(depth):
            if use_scan:
                self.layers.append(ParallelMLayer(dim, heads=heads, physics_config=self.physics_config))
            else:
                if self.physics_config.get('fractal', {}).get('enabled', False):
                    from ..layers import FractalMLayer
                    self.layers.append(FractalMLayer(dim, heads=heads, rank=rank, integrator_type=integrator_type, 
                                                     physics_config=self.physics_config, layer_idx=idx, total_depth=depth))
                else:
                    self.layers.append(MLayer(dim, heads=heads, rank=rank, base_dt=base_dt, integrator_type=integrator_type, 
                                             physics_config=self.physics_config, layer_idx=idx, total_depth=depth))
        
        readout_cfg = self.physics_config.get('readout', {})
        readout_type = readout_cfg.get('type', 'standard')
        
        self.readout_norm = nn.LayerNorm(dim)
        
        if readout_type == 'implicit' or readout_type == 'binary':
             coord_dim = emb_cfg.get('coord_dim', 16) 
             # Implicit readout uses temperature-annealed sigmoid MLP
             if self.holographic:
                 self.readout = nn.Identity()
             else:
                 self.readout = ImplicitReadout(dim, coord_dim)
        else:
             self.readout = nn.Linear(dim, vocab_size)
        
        
        self._print_manifest(vocab_size, dim, depth, heads, integrator_type, use_scan)
        
        self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)

        self.apply(self._init_weights)
        
        # Initialize CUDA fusion manager AFTER weight initialization
        # to avoid interfering with _init_weights
        self.fusion_manager = CUDAFusionManager(self)


    def _print_manifest(self, vocab_size, dim, depth, heads, integrator, scan):
        from ..cuda.ops import CUDA_AVAILABLE
        from ..embeddings import FunctionalEmbedding, ImplicitEmbedding
        
        emb_name = "Standard"
        if isinstance(self.embedding, FunctionalEmbedding): emb_name = f"Functional ({self.embedding.mode})"
        elif isinstance(self.embedding, ImplicitEmbedding): emb_name = "Implicit (SIREN)"
        
        accel = "HARDWARE (CUDA)" if CUDA_AVAILABLE else "EMULATED (CPU)"
        readout = "Identity" if self.holographic else "Implicit MLP"
        
        print(f"\n[GFN] --- Holographic Engine Manifest ---")
        print(f"[GFN]  - Configuration: {depth} Layers | {heads} Heads | {dim} Dim")
        print(f"[GFN]  - Integrator:    {integrator.upper()}")
        print(f"[GFN]  - Acceleration:  {accel}")
        print(f"[GFN]  - Embedding:     {emb_name}")
        print(f"[GFN]  - Readout:       {readout}")
        
        active_inf = self.physics_config.get('active_inference', {}).get('enabled', False)
        if active_inf:
            features = []
            if self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('enabled', False): features.append("Dynamic Time-Stepping")
            if self.physics_config.get('topology', {}).get('type') == 'torus': features.append("Toroidal Topology")
            if features:
                 print(f"[GFN]  - Features:      {', '.join(features)}")
        print(f"[GFN] -----------------------------------\n")
    
    def _init_weights(self, module):
        from ..embeddings import FunctionalEmbedding
        if isinstance(module, FunctionalEmbedding):
            return
            
        if hasattr(self, 'embedding') and isinstance(self.embedding, FunctionalEmbedding):
            # If the module is owned by the embedding, skip it
            emb_params = set(self.embedding.parameters())
            mod_params = set(module.parameters())
            if mod_params.issubset(emb_params) and len(mod_params) > 0:
                return

        if isinstance(module, nn.Linear):
            std = 0.1 if hasattr(module, 'is_readout') else 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids=None, attention_mask=None, state=None, force_manual=None, collect_christ=False):
        """
        Forward pass through geodesic flow.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len] (1=valid, 0=pad)
            state: Optional tuple (x, v) to continue from previous state
            force_manual: Optional pre-computed force sequence [batch, seq_len, dim]
            collect_christ: Whether to accumulate all Christoffel metadata (slow)
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            state: Final state tuple (x, v) for continuation
            christoffels: List of accumulated curvature tensors (if collect_christ is True)
        """
        if force_manual is not None:
            all_forces = force_manual
            batch_size, seq_len, _ = all_forces.shape
        else:
            batch_size, seq_len = input_ids.shape
            all_forces = self.embedding(input_ids)  # [batch, seq_len, dim]
        
        # Use ManifoldState for better state management
        state_obj = ManifoldState.from_tuple(state, self.x0, self.v0, batch_size)
        x, v = state_obj.x, state_obj.v
        
        if self.use_scan:
            x_scan = self.x0.expand(batch_size, seq_len, -1)
            
            curr_input = all_forces # [B, L, D]
            all_christoffels = []
            
            for layer in self.layers:
                out_x, out_v, out_ctx, layer_christoffels = layer(None, None, force=curr_input)
                all_christoffels.extend(layer_christoffels)
                
                curr_input = out_x # Use position as input to next layer
                
            x_final = curr_input 
            if not self.holographic:
                x_final = self.readout_norm(x_final)
            logits = self.readout(x_final) # [batch, seq_len, vocab_size]
            
            return logits, (x_final[:, -1], None), all_christoffels

        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            else:
                mask = torch.ones(batch_size, seq_len, 1, device=all_forces.device)
            
            logits_list = []
            all_christoffels = []
            
            context = None
            
            # Try CUDA fusion using manager
            if self.fusion_manager.can_fuse(collect_christ):
                params = self.fusion_manager.prepare_parameters()
                if params is not None:
                    result = self.fusion_manager.execute_fused_forward(x, v, all_forces, mask, params)
                    if result is not None:
                        x_final, v_final, x_seq, reg_loss = result
                        out_seq = x_seq
                        if not self.holographic:
                            out_seq = self.readout_norm(x_seq)
                        logits = self.readout(out_seq)
                        
                        # Log fusion success (only once)
                        if not hasattr(self, '_fusion_logged'):
                            print(f"[GFN:PERF] ✓ CUDA Fusion ACTIVE - Integrator: {self.integrator_type}")
                            self._fusion_logged = True
                        
                        return logits, (x_final, v_final), [reg_loss], [], x_seq, all_forces

            v_seq = []
            x_seq = []
            for t in range(seq_len):
                # Force for current timestep
                force = all_forces[:, t] * mask[:, t]
                
                # Evolve state through layers
                for layer in self.layers:
                    x, v, context, layer_christoffels = layer(x, v, force, context, collect_christ=collect_christ)
                    if collect_christ:
                        all_christoffels.extend(layer_christoffels) 
                
                v_seq.append(v)
                x_seq.append(x)
                
                # Project position to logits
                out = x
                if not self.holographic:
                    out = self.readout_norm(x)
                logit = self.readout(out)  # [batch, vocab_size]
                logits_list.append(logit.unsqueeze(1))
            
            # Stack all logits
            logits = torch.cat(logits_list, dim=1)  # [batch, seq_len, vocab_size]
            
            return logits, (x, v), all_christoffels, v_seq, x_seq, all_forces
    
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None):
        """
        Autoregressive generation with sampling.
        
        Args:
            prompt_ids: Prompt token indices [1, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Softmax temperature (1.0 = normal, <1 = sharper)
            top_k: Limit to top K tokens (e.g. 40)
            top_p: Nucleus sampling probability (e.g. 0.9)
            
        Returns:
            generated_ids: Full sequence including prompt
        """
        self.eval()
        device = prompt_ids.device
        
        # Process prompt
        logits, state, _ = self(prompt_ids)
        
        # Start generation
        generated = prompt_ids.tolist()[0]
        
        def sample_next(logits, temp=1.0, k=None, p=None):
            # Last timestep logits
            next_logit = logits[:, -1, :] / temp
            probs = torch.softmax(next_logit, dim=-1)
            
            # Top-K
            if k is not None:
                v, _ = torch.topk(next_logit, k)
                next_logit[next_logit < v[:, [-1]]] = -float('Inf')
            
            # Top-P (nucleus)
            if p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens above cumulative threshold
                sorted_indices_to_remove = cumulative_probs > p
                # Keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logit[indices_to_remove] = -float('Inf')
            
            # Sample
            if k is None and p is None:
                # Greedy
                return torch.argmax(next_logit, dim=-1, keepdim=True)
            else:
                # Multinomial
                probs = torch.softmax(next_logit, dim=-1)
                return torch.multinomial(probs, num_samples=1)

        # Initial sample
        curr_token = sample_next(logits, temperature, top_k, top_p)
        generated.append(curr_token.item())
        
        for _ in range(max_new_tokens - 1):
            logits, state, _ = self(curr_token, state=state)
            curr_token = sample_next(logits, temperature, top_k, top_p)
            generated.append(curr_token.item())
        
        return generated
