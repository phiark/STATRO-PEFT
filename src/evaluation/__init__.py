#!/usr/bin/env python3
"""
Evaluation module for STRATO-PEFT experimental framework.

This module provides comprehensive evaluation tools for PEFT methods,
including performance metrics, efficiency analysis, and comparative studies.

Author: STRATO-PEFT Research Team
Date: 2024
"""

from .evaluator import Evaluator
from .metrics import (
    MetricCalculator,
    PerformanceMetrics,
    EfficiencyMetrics,
    StratoMetrics
)
from .comparative_analysis import (
    ComparativeAnalyzer,
    ComparisonResult,
    MethodComparison
)
from .visualization import (
    ResultVisualizer,
    PlotConfig,
    VisualizationUtils
)

__all__ = [
    # Core evaluation
    'Evaluator',
    
    # Metrics
    'MetricCalculator',
    'PerformanceMetrics',
    'EfficiencyMetrics',
    'StratoMetrics',
    
    # Comparative analysis
    'ComparativeAnalyzer',
    'ComparisonResult',
    'MethodComparison',
    
    # Visualization
    'ResultVisualizer',
    'PlotConfig',
    'VisualizationUtils'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "STRATO-PEFT Research Team"
__description__ = "Comprehensive evaluation tools for PEFT methods"