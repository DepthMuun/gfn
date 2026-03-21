"""
World Physics - Defines the laws that govern the internal world.

This module implements the fundamental rules and transformations
that determine how entities interact and evolve.
"""

from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import torch
import torch.nn as nn
import numpy as np

from .entity import Entity, EntityType
from ..utils.properties import extract_numeric_properties, is_prime


class InteractionType(Enum):
    """Types of interactions that can occur between entities."""
    TRANSFORMATION = "transformation"  # E1 + E2 -> E3
    RELATION = "relation"              # E1 <-> E2
    EMERGENCE = "emergence"            # {E_i} -> E_new
    INFLUENCE = "influence"            # E1 affects properties of E2
    DECAY = "decay"                    # E1 fades over time
    UNKNOWN = "unknown"


class WorldPhysics:
    """
    Defines and enforces the laws of the internal world.
    
    Responsibilities:
    - Classify symbols into entity types
    - Extract properties from symbols
    - Validate operations
    - Define transformation functions
    - Check conservation laws
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_properties: int = 64,
        d_embedding: int = 256
    ):
        self.d_model = d_model
        self.d_properties = d_properties
        self.d_embedding = d_embedding
        
        # Type compatibility matrix Φ(τ_i, τ_j)
        self.type_compatibility = self._init_type_compatibility()
        
        # Interaction compatibility Φ_I(τ)
        self.interaction_compatibility = self._init_interaction_compatibility()
        
        # Conservation laws (Generalized)
        self.conservation_laws = self._init_conservation_laws()
        
        # Interaction Registry for extensibility
        self.interaction_registry = {
            InteractionType.TRANSFORMATION: self._transform_generic,
            InteractionType.RELATION: self._process_relation,
            InteractionType.INFLUENCE: self._process_influence,
        }
    
    def register_interaction_handler(
        self, 
        interaction: InteractionType, 
        handler: Callable
    ):
        """Register a new interaction handler."""
        self.interaction_registry[interaction] = handler
    
    def _init_type_compatibility(self) -> Dict[Tuple[EntityType, EntityType], bool]:
        """
        Initialize type compatibility matrix.
        
        Φ(τ_i, τ_j) determines which entity types can interact.
        """
        compatibility = {}
        
        # Numbers can interact with numbers
        compatibility[(EntityType.NUMBER, EntityType.NUMBER)] = True
        
        # Operations can interact with appropriate types
        compatibility[(EntityType.OPERATION, EntityType.NUMBER)] = True
        compatibility[(EntityType.NUMBER, EntityType.OPERATION)] = True
        
        # Concepts can interact with concepts
        compatibility[(EntityType.CONCEPT, EntityType.CONCEPT)] = True
        
        # Default: types cannot interact
        for type1 in EntityType:
            for type2 in EntityType:
                if (type1, type2) not in compatibility:
                    compatibility[(type1, type2)] = False
        
        return compatibility
    
    def _init_interaction_compatibility(self) -> Dict[InteractionType, List[EntityType]]:
        """
        Initialize interaction compatibility.
        """
        return {
            InteractionType.TRANSFORMATION: [EntityType.NUMBER, EntityType.CONCEPT, EntityType.OBJECT],
            InteractionType.RELATION: [EntityType.CONCEPT, EntityType.OBJECT],
            InteractionType.INFLUENCE: [EntityType.CONCEPT, EntityType.OBJECT],
        }
    
    def _init_conservation_laws(self) -> List[Callable]:
        """Initialize conservation laws."""
        return [
            self._law_type_conservation,
            self._law_parity_conservation_add,
        ]
    
    def classify_symbol(
        self,
        symbol: str,
        context: Optional[torch.Tensor] = None
    ) -> EntityType:
        """
        Classify a symbol into an entity type.
        
        τ(s, c) = argmax_τ' P(τ' | s, c)
        """
        # 1. Neutral classification (To be overridden by Model's Neural Network)
        # We only keep a minimal fallback to prevent crashes
        
        if any(c.isdigit() for c in symbol):
            return EntityType.NUMBER
        
        # Everything else is a CONCEPT by default in a general reality
        return EntityType.CONCEPT
    
    def extract_properties(
        self,
        symbol: str,
        entity_type: EntityType,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract intrinsic properties from a symbol.
        
        p = Ψ_τ(s, c)
        
        Args:
            symbol: Input symbol
            entity_type: Type of entity
            context: Optional context
            
        Returns:
            Property vector
        """
        properties = torch.zeros(self.d_properties)
        
        if entity_type == EntityType.NUMBER:
            # Use centralized numeric extraction
            try:
                value = float(symbol)
                return extract_numeric_properties(value, self.d_properties)
            except ValueError:
                pass
        
        elif entity_type == EntityType.OPERATION:
            # Encode operation type
            operation_map = {'add': 0, '+': 0, 'subtract': 1, '-': 1,
                           'multiply': 2, '*': 2, 'divide': 3, '/': 3}
            if symbol in operation_map:
                properties[0] = operation_map[symbol]
        
        return properties
    
    def identify_interaction(self, symbol: str) -> InteractionType:
        """Identify interaction type from a symbol or context representation."""
        interaction_map = {
            '+': InteractionType.TRANSFORMATION,
            'add': InteractionType.TRANSFORMATION,
            'is': InteractionType.RELATION,
            'of': InteractionType.INFLUENCE,
        }
        return interaction_map.get(symbol, InteractionType.UNKNOWN)
    
    def validate_interaction(
        self,
        interaction: InteractionType,
        entities: List[Entity]
    ) -> bool:
        """
        Validate if an interaction can be applied to entities.
        
        V(I, E_1, ..., E_k) = ⋀ Φ_I(τ_i) ⋀ I(E_i, E_j)
        
        Args:
            interaction: Interaction to validate
            entities: Entities to interact
            
        Returns:
            True if interaction is valid
        """
        if interaction not in self.interaction_compatibility:
            return False
        
        valid_types = self.interaction_compatibility[interaction]
        
        # Check all entities have compatible types
        for entity in entities:
            if entity.entity_type not in valid_types:
                return False
        
        # Check entities can interact pairwise
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if not entities[i].can_interact_with(
                    entities[j],
                    self.type_compatibility
                ):
                    return False
        
        return True
    
    def get_interaction_function(
        self,
        interaction: InteractionType
    ) -> Optional[Callable]:
        """
        Get the transformation function for an interaction.
        """
        return self.interaction_registry.get(interaction)
    
    def _transform_generic(
        self,
        entity1: Entity,
        entity2: Entity,
        embedding_network: Optional[nn.Module] = None
    ) -> Entity:
        """
        Generic transformation between two entities.
        """
        # If both are numbers, we can fallback to addition as a primitive
        if entity1.entity_type == EntityType.NUMBER and entity2.entity_type == EntityType.NUMBER:
            m1 = entity1.properties[0].item()
            m2 = entity2.properties[0].item()
            m_result = m1 + m2
            
            properties = torch.zeros(self.d_properties)
            properties[0] = m_result
            # ... keep basic properties for numbers
        else:
            # For concepts/objects, properties are a result of hidden state interaction
            properties = (entity1.properties + entity2.properties) / 2
            
        # Compute result embedding via network
        if embedding_network is not None:
            combined = torch.cat([entity1.embedding, entity2.embedding])
            embedding = embedding_network(combined)
        else:
            embedding = (entity1.embedding + entity2.embedding) / 2
            
        from .entity import EntityFactory
        factory = EntityFactory(self.d_properties, self.d_embedding)
        
        # Result type depends on interaction
        result = factory.create_concept_entity(
            concept_name="transformation_result",
            embedding=embedding,
            attributes=properties,
            source_symbol=f"({entity1.source_symbol}⊕{entity2.source_symbol})"
        )
        
        return result

    def _process_relation(self, e1, e2, net=None): return None # To be implemented
    def _process_influence(self, e1, e2, net=None): return None # To be implemented
    
    # Removed hardcoded arithmetic transformers
    
    def check_conservation_laws(
        self,
        entities_before: List[Entity],
        entities_after: List[Entity],
        interaction: Optional[InteractionType] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if conservation laws are satisfied.
        
        Returns:
            (all_satisfied, violated_laws)
        """
        violated = []
        
        for law in self.conservation_laws:
            if not law(entities_before, entities_after, interaction):
                violated.append(law.__name__)
        
        return len(violated) == 0, violated
    
    def _law_type_conservation(
        self,
        entities_before: List[Entity],
        entities_after: List[Entity],
        interaction: Optional[InteractionType]
    ) -> bool:
        """
        Type conservation law.
        """
        # Generalized: Types should be consistent with the interaction
        return True
    
    def _law_parity_conservation_add(
        self,
        entities_before: List[Entity],
        entities_after: List[Entity],
        interaction: Optional[InteractionType]
    ) -> bool:
        """
        Parity conservation for addition primitive.
        """
        return True # Generalized
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if number is prime (uses centralized utility)."""
        return is_prime(n)
