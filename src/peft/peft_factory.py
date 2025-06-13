#!/usr/bin/env python3
"""
PEFT factory for creating PEFT method instances.

This module provides a factory class for creating different PEFT methods
based on configuration.

Author: STRATO-PEFT Research Team
Date: 2024
"""

import logging
from typing import Dict, Type, Optional

from omegaconf import DictConfig
from transformers import PreTrainedModel

from .base_peft import BasePEFT
from .lora import LoRAPEFT
from .adalora import AdaLoRAPEFT
from .dora import DoRAPEFT
from .strato_peft import StratoPEFT


class PEFTFactory:
    """
    Factory class for creating PEFT method instances.
    """
    
    # Registry of available PEFT methods
    _PEFT_REGISTRY: Dict[str, Type[BasePEFT]] = {
        'lora': LoRAPEFT,
        'adalora': AdaLoRAPEFT,
        'dora': DoRAPEFT,
        'strato_peft': StratoPEFT,
        'strato-peft': StratoPEFT,  # Alternative naming
    }
    
    @classmethod
    def create_peft_method(
        cls,
        config: DictConfig,
        model: PreTrainedModel,
        logger: Optional[logging.Logger] = None
    ) -> BasePEFT:
        """
        Create a PEFT method instance based on configuration.
        
        Args:
            config: Complete experiment configuration
            model: Pre-trained model to adapt
            logger: Logger instance
            
        Returns:
            PEFT method instance
            
        Raises:
            ValueError: If PEFT method is not supported
            KeyError: If required configuration is missing
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        # Validate configuration
        cls._validate_config(config)
        
        # Get PEFT method name
        peft_method = config.peft.method.lower()
        
        if peft_method not in cls._PEFT_REGISTRY:
            available_methods = list(cls._PEFT_REGISTRY.keys())
            raise ValueError(
                f"Unsupported PEFT method: {peft_method}. "
                f"Available methods: {available_methods}"
            )
        
        # Get PEFT-specific configuration
        peft_config = getattr(config.peft, peft_method, None)
        if peft_config is None:
            raise KeyError(
                f"Configuration for PEFT method '{peft_method}' not found. "
                f"Expected config.peft.{peft_method}"
            )
        
        # Create PEFT method instance
        peft_class = cls._PEFT_REGISTRY[peft_method]
        
        logger.info(f"Creating PEFT method: {peft_class.__name__}")
        logger.info(f"PEFT configuration: {peft_config}")
        
        try:
            peft_instance = peft_class(
                config=peft_config,
                model=model,
                logger=logger
            )
            
            # Validate the created instance
            peft_instance.validate_config()
            
            logger.info(f"Successfully created {peft_class.__name__} instance")
            return peft_instance
            
        except Exception as e:
            logger.error(f"Failed to create {peft_class.__name__}: {str(e)}")
            raise
    
    @classmethod
    def _validate_config(cls, config: DictConfig) -> None:
        """
        Validate the PEFT configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            KeyError: If required configuration is missing
            ValueError: If configuration values are invalid
        """
        # Check if PEFT configuration exists
        if not hasattr(config, 'peft'):
            raise KeyError("Configuration must contain 'peft' section")
        
        # Check if method is specified
        if not hasattr(config.peft, 'method'):
            raise KeyError("PEFT configuration must specify 'method'")
        
        # Validate method name
        method = config.peft.method.lower()
        if not method:
            raise ValueError("PEFT method cannot be empty")
        
        # Check if method-specific config exists
        if not hasattr(config.peft, method):
            raise KeyError(
                f"Configuration for PEFT method '{method}' not found. "
                f"Expected config.peft.{method}"
            )
    
    @classmethod
    def register_peft_method(
        cls,
        name: str,
        peft_class: Type[BasePEFT]
    ) -> None:
        """
        Register a new PEFT method.
        
        Args:
            name: Name of the PEFT method
            peft_class: PEFT class to register
            
        Raises:
            ValueError: If the class is not a subclass of BasePEFT
        """
        if not issubclass(peft_class, BasePEFT):
            raise ValueError(
                f"PEFT class must be a subclass of BasePEFT, "
                f"got {peft_class.__name__}"
            )
        
        cls._PEFT_REGISTRY[name.lower()] = peft_class
        logging.getLogger(__name__).info(
            f"Registered PEFT method: {name} -> {peft_class.__name__}"
        )
    
    @classmethod
    def get_available_methods(cls) -> list:
        """
        Get list of available PEFT methods.
        
        Returns:
            List of available PEFT method names
        """
        return list(cls._PEFT_REGISTRY.keys())
    
    @classmethod
    def get_method_info(cls, method_name: str) -> Dict[str, str]:
        """
        Get information about a specific PEFT method.
        
        Args:
            method_name: Name of the PEFT method
            
        Returns:
            Dictionary containing method information
            
        Raises:
            ValueError: If method is not found
        """
        method_name = method_name.lower()
        if method_name not in cls._PEFT_REGISTRY:
            raise ValueError(f"PEFT method '{method_name}' not found")
        
        peft_class = cls._PEFT_REGISTRY[method_name]
        
        return {
            'name': method_name,
            'class_name': peft_class.__name__,
            'module': peft_class.__module__,
            'docstring': peft_class.__doc__ or "No description available"
        }
    
    @classmethod
    def log_available_methods(cls, logger: logging.Logger) -> None:
        """
        Log information about all available PEFT methods.
        
        Args:
            logger: Logger instance
        """
        logger.info("Available PEFT methods:")
        for method_name in sorted(cls._PEFT_REGISTRY.keys()):
            try:
                info = cls.get_method_info(method_name)
                logger.info(f"  - {method_name}: {info['class_name']}")
            except Exception as e:
                logger.warning(f"  - {method_name}: Error getting info - {e}")