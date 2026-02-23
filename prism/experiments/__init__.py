from .base import BaseExperiment
from .quantization import QuantizationExperiment
from .forgetting import ForgettingExperiment
from .ood import OODExperiment
from .merging import MergingExperiment

EXPERIMENT_REGISTRY = {
    "quantization": QuantizationExperiment,
    "forgetting": ForgettingExperiment,
    "ood": OODExperiment,
    "merging": MergingExperiment,
}
