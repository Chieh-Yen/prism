from .base import BaseExperiment
from .quantization import QuantizationExperiment
from .forgetting import ForgettingExperiment
from .finetuning import FinetuningExperiment
from .ood import OODExperiment
from .merging import MergingExperiment
from .cross_scale import CrossScaleExperiment

EXPERIMENT_REGISTRY = {
    "quantization": QuantizationExperiment,
    "forgetting": ForgettingExperiment,
    "finetuning": FinetuningExperiment,
    "ood": OODExperiment,
    "merging": MergingExperiment,
    "cross_scale": CrossScaleExperiment,
}
