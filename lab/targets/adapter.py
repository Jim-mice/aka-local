"""Small target-adapter interface; not a second campaign framework."""
from abc import ABC, abstractmethod


class TargetAdapter(ABC):
    @abstractmethod
    def load_target_spec(self): ...

    @abstractmethod
    def prepare_inputs(self, seed=0, dtype=None): ...

    @abstractmethod
    def run_megatron_reference(self, inputs): ...

    @abstractmethod
    def run_replay(self, inputs): ...

    @abstractmethod
    def compare_outputs(self, reference, replay): ...

    @abstractmethod
    def benchmark_reference(self, inputs, warmup=5, iterations=30): ...
