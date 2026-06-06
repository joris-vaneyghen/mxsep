import inspect
from typing import Any, Dict, Optional, Type

import audiomentations
from audiomentations.core.transforms_interface import BaseWaveformTransform

import musep.data.augmentations


class TransformRegistry:
    """Registry for custom transform implementations"""

    _transforms_dict: Dict[str, Type[BaseWaveformTransform]] = {}

    @classmethod
    def finalize_registration(cls):
        # Call this after all imports are done
        for name, transform_class in cls._pending.items():
            cls._registry[name] = transform_class


    @classmethod
    def register(cls, name: Optional[str] = None):
        """Decorator to register a custom transform"""

        def decorator(transform_class: Type[BaseWaveformTransform]):
            print(f"Registered custom transform '{transform_class.__name__}'")
            transform_name = name or transform_class.__name__
            cls._transforms_dict[transform_name] = transform_class
            return transform_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseWaveformTransform]]:
        """Get a transform class by name (custom or audiomentations)"""
        # Check custom transforms first
        if name in cls._transforms_dict:
            return cls._transforms_dict[name]

        cls_name = getattr(musep.data.augmentations, name, None)
        if cls_name is None:
            # Fall back to audiomentations
            return getattr(audiomentations, name, None)

        if cls_name is not None:
            cls._transforms_dict[name] = cls_name
        return cls_name


class DynamicTransformFactory:
    """Factory for creating transform instances with flexible configuration"""

    @staticmethod
    def create(class_name: str, init_params: Dict[str, Any], transform_params: Dict[str, Any]=None) -> BaseWaveformTransform:
        """Create transform instance from class name and parameters"""

        # Get the transform class from registry
        transform_class = TransformRegistry.get(class_name)

        if transform_class is None:
            raise ValueError(
                f"Unknown transform class '{class_name}'. "
                f"Available custom transforms: {list(TransformRegistry._transforms_dict.keys())}"
            )

        # Create instance with parameters
        return DynamicTransformFactory._create_instance(transform_class, init_params, transform_params)

    @staticmethod
    def _create_instance(transform_class: Type[BaseWaveformTransform],
                         init_params: Dict[str, Any], transform_params: Dict[str, Any]=None) -> BaseWaveformTransform:
        """Create instance handling different initialization patterns"""

        transform = transform_class(**init_params)
        if transform_params is not None :
            transform.parameters = transform_params
            transform.parameters['should_apply'] = True
            transform.freeze_parameters()

        return transform









