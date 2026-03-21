"""
Materialization - Conversion between symbols and entities.

Handles the bidirectional transformation:
- Symbol → Entity (materialization)
- Entity → Symbol (collapse)
"""

from typing import Optional
import torch
import torch.nn as nn

from .entity import Entity, EntityType, EntityFactory
from .world_physics import WorldPhysics, InteractionType


class Materializer:
    """
    Handles materialization of symbols into entities and vice versa.
    
    Materialize(s, c) = E
    Collapse(E, W) = s
    """
    
    def __init__(
        self,
        world_physics: WorldPhysics,
        entity_factory: EntityFactory,
        embedding_network: Optional[nn.Module] = None
    ):
        self.physics = world_physics
        self.factory = entity_factory
        self.embedding_network = embedding_network
    
    def materialize(
        self,
        symbol: str,
        context: Optional[torch.Tensor] = None
    ) -> Entity:
        """
        Materialize a symbol into an entity.
        
        Process:
        1. Classify type: τ = τ(s, c)
        2. Extract properties: p = Ψ_τ(s, c)
        3. Generate embedding: e = Encoder(s, c)
        4. Initialize relations: R = ∅
        5. Initial state: s_0 = default(τ)
        
        Args:
            symbol: Symbol to materialize
            context: Optional context vector
            
        Returns:
            Materialized entity
        """
        # Step 1: Classify type
        entity_type = self.physics.classify_symbol(symbol, context)
        
        # Step 2: Extract properties
        properties = self.physics.extract_properties(symbol, entity_type, context)
        
        # Step 3: Generate embedding
        if self.embedding_network is not None and context is not None:
            embedding = self.embedding_network(context)
        else:
            # Default: random embedding
            embedding = torch.randn(self.factory.d_embedding) * 0.1
        
        # Step 4 & 5: Create entity based on type
        if entity_type == EntityType.NUMBER:
            try:
                magnitude = float(symbol)
                entity = self.factory.create_number_entity(
                    magnitude=magnitude,
                    embedding=embedding,
                    source_symbol=symbol
                )
            except ValueError:
                # Fallback if parsing fails
                entity = Entity(
                    id=self.factory._next_id,
                    entity_type=EntityType.UNKNOWN,
                    properties=properties,
                    embedding=embedding,
                    source_symbol=symbol
                )
                self.factory._next_id += 1
        
        elif entity_type == EntityType.OPERATION:
            interaction_type = self.physics.identify_interaction(symbol)
            entity = self.factory.create_operation_entity(
                operation_name=interaction_type.value if interaction_type else symbol,
                embedding=embedding,
                source_symbol=symbol
            )
        
        elif entity_type == EntityType.CONCEPT:
            entity = self.factory.create_concept_entity(
                concept_name=symbol,
                embedding=embedding,
                attributes=properties,
                source_symbol=symbol
            )
        
        else:
            # Unknown type
            entity = Entity(
                id=self.factory._next_id,
                entity_type=EntityType.UNKNOWN,
                properties=properties,
                embedding=embedding,
                source_symbol=symbol
            )
            self.factory._next_id += 1
        
        return entity
    
    def collapse(
        self,
        entity: Entity,
        world_state: Optional[torch.Tensor] = None
    ) -> str:
        """
        Collapse an entity back to a symbol.
        
        Args:
            entity: Entity to collapse
            world_state: Optional world state for context
            
        Returns:
            Symbol representation
        """
        if entity.entity_type == EntityType.NUMBER:
            # Extract magnitude from properties
            magnitude = entity.properties[0].item()
            
            # Format as integer if whole number, otherwise float
            if abs(magnitude - round(magnitude)) < 1e-6:
                return str(int(round(magnitude)))
            else:
                return f"{magnitude:.6f}".rstrip('0').rstrip('.')
        
        elif entity.entity_type == EntityType.OPERATION:
            # Return operation symbol
            if 'operation' in entity.state:
                op_name = entity.state['operation']
                op_symbols = {
                    'transformation': '⊕',
                    'relation': '⇔',
                    'influence': '»',
                    'emergence': '⇶'
                }
                return op_symbols.get(op_name, op_name)
            return entity.source_symbol or "?"
        
        elif entity.entity_type == EntityType.CONCEPT:
            # Return concept name
            if 'concept' in entity.state:
                return entity.state['concept']
            return entity.source_symbol or "concept"
        
        else:
            # Unknown - return source symbol or placeholder
            return entity.source_symbol or f"entity_{entity.id}"
