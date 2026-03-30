"""
gfn/realizations/gssm/config/normalizer.py
ConfigNormalizer: Pre-procesa y normaliza configuración para ModelFactory.

Elimina el código spaghetti de mapeo de kwargs del factory,
centralizando toda la lógica de normalización en una clase testeable.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class ConfigNormalizationError(Exception):
    """Error durante la normalización de configuración."""
    pass


class ConfigNormalizer:
    """
    Normaliza configuración para ModelFactory.
    
    Responsabilidades:
    1. Mapear kwargs planos a estructura anidada (dotted y prefix)
    2. Sincronizar parámetros entre ManifoldConfig y PhysicsConfig
    3. Validar configuración resultante
    """
    
    # Sub-configs válidos en PhysicsConfig
    PHYSICS_SUBCONFIGS = [
        'topology', 'stability', 'dynamics', 'active_inference',
        'embedding', 'readout', 'mixture', 'fractal', 
        'hysteresis', 'singularities'
    ]
    
    # Parámetros que requieren sincronización bidireccional
    SYNC_PARAMETERS = [
        ('integrator', 'stability', 'integrator_type'),
        ('impulse_scale', 'embedding', 'impulse_scale'),
        ('rank', 'topology', 'riemannian_rank'),
        ('dynamics_type', 'dynamics', 'type'),
        ('trajectory_mode', None, 'trajectory_mode'),  # Directo en physics
        ('coupler_mode', 'mixture', 'coupler_mode'),
        ('holographic', 'active_inference', 'holographic_geometry'),
    ]
    
    def __init__(self, config: Any, explicit_keys: Set[str]):
        self.config = config
        self.explicit_keys = explicit_keys
        self.normalized_kwargs: Dict[str, Any] = {}
    
    def normalize(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza kwargs planos en config anidada.
        
        Args:
            kwargs: Diccionario de argumentos planos
            
        Returns:
            kwargs restantes que no pudieron ser mapeados
        """
        remaining = dict(kwargs)
        
        for k, v in list(remaining.items()):
            mapped = self._try_map_kwarg(k, v)
            if mapped:
                remaining.pop(k)
        
        return remaining
    
    def _try_map_kwarg(self, key: str, value: Any) -> bool:
        """
        Intenta mapear un kwarg a config. Retorna True si tuvo éxito.
        
        Estrategias en orden de prioridad:
        1. Dotted path (e.g., 'physics.topology.type')
        2. Directo en ManifoldConfig
        3. Directo en sub-config de física
        4. Prefijo (e.g., 'topology_type')
        """
        # 1. Dotted path
        if '.' in key:
            if self._try_dotted_path(key, value):
                return True
        
        # 2. Directo en ManifoldConfig
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            return True
        
        # 3. Directo en sub-config de física
        for sub_name in self.PHYSICS_SUBCONFIGS:
            target = getattr(self.config.physics, sub_name, None)
            if target and hasattr(target, key):
                setattr(target, key, value)
                return True
        
        # 4. Prefijo
        if '_' in key:
            if self._try_prefix_mapping(key, value):
                return True
        
        return False
    
    def _try_dotted_path(self, key: str, value: Any) -> bool:
        """Mapea dotted path como 'physics.topology.type'."""
        try:
            parts = key.split('.')
            obj = self.config
            
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            setattr(obj, parts[-1], value)
            return True
        except (AttributeError, KeyError):
            return False
    
    def _try_prefix_mapping(self, key: str, value: Any) -> bool:
        """
        Mapea prefijo como 'topology_type' -> physics.topology.type.
        
        Caso especial: active_inference tiene '_' en el nombre.
        """
        # Caso especial para active_inference
        if key.startswith('active_inference_'):
            real_k = key[len('active_inference_')+1:]
            self._apply_physics_override('active_inference', {real_k: value})
            return True
        
        # Prefijos estándar
        for prefix in self.PHYSICS_SUBCONFIGS:
            if key.startswith(prefix + '_'):
                real_k = key[len(prefix)+1:]
                self._apply_physics_override(prefix, {real_k: value})
                return True
        
        return False
    
    def _apply_physics_override(self, prefix: str, override_dict: Dict[str, Any]) -> None:
        """Aplica override a sub-config de física."""
        from ..config.loader import apply_physics_overrides
        apply_physics_overrides(self.config.physics, {prefix: override_dict})
    
    def synchronize_parameters(self) -> None:
        """
        Sincroniza parámetros entre ManifoldConfig y PhysicsConfig.
        
        Prioriza ManifoldConfig si el valor fue provisto explícitamente.
        """
        for param_config, sub_config, param_physics in self.SYNC_PARAMETERS:
            self._sync_parameter(param_config, sub_config, param_physics)
    
    def _sync_parameter(self, param_config: str, sub_config: Optional[str], param_physics: str) -> None:
        """Sincroniza un parámetro individual."""
        # Si fue provisto explícitamente, priorizar ManifoldConfig -> PhysicsConfig
        if param_config in self.explicit_keys:
            val = getattr(self.config, param_config)
            
            if sub_config:
                target = getattr(self.config.physics, sub_config)
                setattr(target, param_physics, val)
            else:
                # Parámetro directo en physics (como trajectory_mode)
                setattr(self.config.physics, param_physics, val)
        else:
            # Sincronizar PhysicsConfig -> ManifoldConfig
            if sub_config:
                source = getattr(self.config.physics, sub_config)
                val = getattr(source, param_physics)
            else:
                val = getattr(self.config.physics, param_physics)
            
            setattr(self.config, param_config, val)
        
        # Caso especial: holographic es bidireccional con OR lógico
        if param_config == 'holographic':
            config_val = getattr(self.config, 'holographic', False)
            physics_val = getattr(
                self.config.physics.active_inference, 'holographic_geometry', False
            )
            final_val = config_val or physics_val
            
            self.config.holographic = final_val
            self.config.physics.active_inference.holographic_geometry = final_val
    
    def validate(self) -> List[str]:
        """
        Valida la configuración normalizada.
        
        Returns:
            Lista de errores encontrados (vacía si todo OK)
        """
        errors = []
        
        # Validar dimensiones coherentes
        if hasattr(self.config, 'heads') and hasattr(self.config, 'dim'):
            if self.config.dim % self.config.heads != 0:
                errors.append(
                    f"dim ({self.config.dim}) debe ser divisible por heads ({self.config.heads})"
                )
        
        # Validar topología
        if hasattr(self.config.physics, 'topology'):
            valid_topologies = ['euclidean', 'torus', 'spherical', 'hyperbolic', 'hierarchical', 'holographic']
            topo_type = getattr(self.config.physics.topology, 'type', None)
            if topo_type and topo_type not in valid_topologies:
                errors.append(f"Topología inválida: {topo_type}")
        
        return errors


def normalize_config(config: Any, kwargs: Dict[str, Any], explicit_keys: Set[str]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Función helper para normalizar configuración.
    
    Returns:
        (kwargs_restantes, errores_de_validación)
    """
    normalizer = ConfigNormalizer(config, explicit_keys)
    remaining = normalizer.normalize(kwargs)
    normalizer.synchronize_parameters()
    errors = normalizer.validate()
    
    if errors:
        logger.warning(f"Config validation errors: {errors}")
    
    return remaining, errors
