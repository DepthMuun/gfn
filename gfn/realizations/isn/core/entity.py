"""
Entity representation for the Reality Model.

An entity is something that EXISTS in the internal world with properties,
behavior, and relations.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np

from ..utils.properties import extract_numeric_properties, validate_number_properties, is_prime


class EntityType(Enum):
    """Types of entities that can exist in the world."""
    NUMBER = "number"
    CONCEPT = "concept"
    OBJECT = "object"
    OPERATION = "operation"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """
    Represents an entity that exists in the internal world.
    
    Mathematical definition:
    E = (id, τ, p, e, R, s)
    
    Where:
    - id: unique identifier
    - τ: entity type
    - p: intrinsic properties vector
    - e: semantic embedding
    - R: relations with other entities
    - s: dynamic state
    """
    
    id: int
    entity_type: EntityType
    properties: torch.Tensor  # p ∈ R^{d_p}
    embedding: torch.Tensor   # e ∈ R^{d_e}
    state: Dict[str, any] = field(default_factory=dict)
    relations: List[Tuple[int, str, float]] = field(default_factory=list)  # (entity_id, relation_type, weight)
    
    # Metadata
    source_symbol: Optional[str] = None
    creation_time: int = 0
    
    def __post_init__(self):
        """Validate entity after initialization."""
        assert self.properties.dim() == 1, "Properties must be 1D tensor"
        assert self.embedding.dim() == 1, "Embedding must be 1D tensor"
    
    def coherence(
        self,
        alpha: float = 0.33,
        beta: float = 0.33,
        gamma: float = 0.34
    ) -> float:
        """
        Compute coherence of this entity.
        
        C(E) = α·C_prop(p) + β·C_embed(e) + γ·C_rel(R)
        
        Args:
            alpha: Weight for property coherence
            beta: Weight for embedding coherence
            gamma: Weight for relation coherence
            
        Returns:
            Coherence score in [0, 1]
        """
        c_prop = self._property_coherence()
        c_embed = self._embedding_coherence()
        c_rel = self._relation_coherence()
        
        return alpha * c_prop + beta * c_embed + gamma * c_rel
    
    def _property_coherence(self) -> float:
        """
        Compute coherence of properties.
        
        C_prop(p) = 1 - (1/|C|) Σ 1_violate(c, p)
        
        Checks if properties satisfy type-specific constraints.
        """
        violations = 0
        total_constraints = 0
        
        if self.entity_type == EntityType.NUMBER:
            # For numbers, use centralized validation
            total_constraints = 3
            violations = validate_number_properties(self.properties)
        
        elif self.entity_type == EntityType.CONCEPT:
            # For concepts, properties should be normalized
            total_constraints = 1
            norm = torch.norm(self.properties)
            if norm > 10.0:  # Reasonable upper bound
                violations += 1
        
        if total_constraints == 0:
            return 1.0
        
        return 1.0 - (violations / total_constraints)
    
    def _embedding_coherence(self) -> float:
        """
        Compute coherence of embedding.
        
        C_embed(e) = σ(||e||_2 / √d_e)
        
        Normalizes embedding norm to avoid degenerate embeddings.
        """
        d_e = self.embedding.shape[0]
        norm = torch.norm(self.embedding, p=2)
        normalized_norm = norm / np.sqrt(d_e)
        
        # Sigmoid to map to [0, 1], centered around norm=1
        coherence = torch.sigmoid(1.0 - torch.abs(normalized_norm - 1.0))
        
        return coherence.item()
    
    def _relation_coherence(self) -> float:
        """
        Compute coherence of relations.
        
        C_rel(R) = (1/|R|) Σ 1_valid(E_i, E_j, w)
        
        For now, checks if relation weights are in valid range.
        """
        if len(self.relations) == 0:
            return 1.0
        
        valid_relations = sum(
            1 for (_, _, weight) in self.relations
            if 0.0 <= weight <= 1.0
        )
        
        return valid_relations / len(self.relations)
    
    def can_interact_with(
        self,
        other: 'Entity',
        type_compatibility: Dict[Tuple[EntityType, EntityType], bool],
        distance_threshold: float = 2.0
    ) -> bool:
        """
        Check if this entity can interact with another.
        
        I(E_i, E_j) = Φ(τ_i, τ_j) ∧ D(e_i, e_j) < θ
        
        Args:
            other: Other entity
            type_compatibility: Function Φ mapping type pairs to compatibility
            distance_threshold: Maximum semantic distance θ
            
        Returns:
            True if entities can interact
        """
        # Check type compatibility
        type_pair = (self.entity_type, other.entity_type)
        if type_pair not in type_compatibility:
            return False
        if not type_compatibility[type_pair]:
            return False
        
        # Check semantic distance
        distance = self._semantic_distance(other)
        
        return distance < distance_threshold
    
    def _semantic_distance(self, other: 'Entity') -> float:
        """
        Compute semantic distance between entities.
        
        D(e_i, e_j) = 1 - cos(e_i, e_j)
        
        Args:
            other: Other entity
            
        Returns:
            Distance in [0, 2]
        """
        cos_sim = torch.nn.functional.cosine_similarity(
            self.embedding.unsqueeze(0),
            other.embedding.unsqueeze(0)
        )
        
        distance = 1.0 - cos_sim.item()
        
        return distance
    
    def add_relation(self, entity_id: int, relation_type: str, weight: float = 1.0):
        """Add a relation to another entity."""
        self.relations.append((entity_id, relation_type, weight))
    
    def get_relations_of_type(self, relation_type: str) -> List[Tuple[int, float]]:
        """Get all relations of a specific type."""
        return [
            (entity_id, weight)
            for (entity_id, rel_type, weight) in self.relations
            if rel_type == relation_type
        ]
    
    def to_dict(self) -> Dict:
        """Convert entity to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.entity_type.value,
            'properties': self.properties.tolist(),
            'embedding': self.embedding.tolist(),
            'state': self.state,
            'relations': self.relations,
            'source_symbol': self.source_symbol,
            'creation_time': self.creation_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        """Create entity from dictionary."""
        return cls(
            id=data['id'],
            entity_type=EntityType(data['type']),
            properties=torch.tensor(data['properties']),
            embedding=torch.tensor(data['embedding']),
            state=data.get('state', {}),
            relations=data.get('relations', []),
            source_symbol=data.get('source_symbol'),
            creation_time=data.get('creation_time', 0),
        )
    
    def __repr__(self) -> str:
        return (
            f"Entity(id={self.id}, type={self.entity_type.value}, "
            f"symbol='{self.source_symbol}', coherence={self.coherence():.3f})"
        )


class EntityFactory:
    """Factory for creating entities with proper initialization."""
    
    def __init__(self, d_properties: int = 64, d_embedding: int = 256):
        self.d_properties = d_properties
        self.d_embedding = d_embedding
        self._next_id = 0
    
    def create_number_entity(
        self,
        magnitude: float,
        embedding: torch.Tensor,
        source_symbol: Optional[str] = None
    ) -> Entity:
        """
        Create a number entity.
        
        Properties for numbers:
        p_num = [m, sgn(m), m mod 2, 1_prime(m), ⌊log₁₀|m|⌋, ...]
        """
        # Use centralized extraction
        properties = extract_numeric_properties(magnitude, self.d_properties)
        
        entity = Entity(
            id=self._next_id,
            entity_type=EntityType.NUMBER,
            properties=properties,
            embedding=embedding,
            source_symbol=source_symbol,
        )
        
        self._next_id += 1
        return entity
    
    def create_operation_entity(
        self,
        operation_name: str,
        embedding: torch.Tensor,
        source_symbol: Optional[str] = None
    ) -> Entity:
        """Create an operation entity."""
        properties = torch.zeros(self.d_properties)
        
        # Encode operation type in properties
        operation_encoding = {
            'add': 0,
            'subtract': 1,
            'multiply': 2,
            'divide': 3,
        }
        
        if operation_name in operation_encoding:
            properties[0] = operation_encoding[operation_name]
        
        entity = Entity(
            id=self._next_id,
            entity_type=EntityType.OPERATION,
            properties=properties,
            embedding=embedding,
            source_symbol=source_symbol,
            state={'operation': operation_name}
        )
        
        self._next_id += 1
        return entity
    
    def create_concept_entity(
        self,
        concept_name: str,
        embedding: torch.Tensor,
        attributes: Optional[torch.Tensor] = None,
        source_symbol: Optional[str] = None
    ) -> Entity:
        """Create a concept entity."""
        if attributes is not None:
            properties = attributes
        else:
            properties = torch.randn(self.d_properties) * 0.1
        
        entity = Entity(
            id=self._next_id,
            entity_type=EntityType.CONCEPT,
            properties=properties,
            embedding=embedding,
            source_symbol=source_symbol,
            state={'concept': concept_name}
        )
        
        self._next_id += 1
        return entity
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if number is prime (uses centralized utility)."""
        return is_prime(n)
    
    def reset_id_counter(self):
        """Reset the ID counter (useful for testing)."""
        self._next_id = 0
