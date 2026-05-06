from .base import BaseExperiment
from .quantization import QuantizationExperiment

EXPERIMENT_REGISTRY = {
    "quantization": QuantizationExperiment,
}
