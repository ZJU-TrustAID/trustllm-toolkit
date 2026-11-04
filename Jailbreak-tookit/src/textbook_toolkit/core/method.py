"""A chapter chooses its own input and output types."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


class Method(ABC, Generic[Input, Output]):
    """Small shared contract; no model, dataset, or evaluation assumptions."""

    @abstractmethod
    def build(self, value: Input) -> Output:
        """Construct the result for one input."""

    def __call__(self, value: Input) -> Output:
        return self.build(value)
