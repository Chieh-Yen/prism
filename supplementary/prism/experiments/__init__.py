from .base import BaseExperiment
from .quantization import QuantizationExperiment
from .forgetting import ForgettingExperiment

EXPERIMENT_REGISTRY = {
    "quantization": QuantizationExperiment,
    "forgetting": ForgettingExperiment,
}
