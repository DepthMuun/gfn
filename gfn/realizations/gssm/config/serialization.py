import dataclasses
from typing import Any, Dict, Type, TypeVar, get_type_hints, get_args, get_origin, Union

T = TypeVar('T')

def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    Reconstructs a nested dataclass from a dictionary.
    Handles nested dataclasses and basic types.
    """
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = get_type_hints(cls)
    kwargs = {}
    
    for field in dataclasses.fields(cls):
        if field.name in data:
            value = data[field.name]
            field_type = field_types[field.name]
            
            # Handle Optional[T]
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                if type(None) in args:
                    # It's an Optional. Find the non-None type
                    field_type = [arg for arg in args if arg is not type(None)][0]
            
            # Handle nested dataclasses
            if dataclasses.is_dataclass(field_type):
                if value is not None:
                    kwargs[field.name] = from_dict(field_type, value)
                else:
                    kwargs[field.name] = None
            else:
                kwargs[field.name] = value
                
    return cls(**kwargs)
