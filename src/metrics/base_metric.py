from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Abstract interface for detection or tracking metric accumulators."""

    @abstractmethod
    def reset(self):
        """Clear cached state before a new evaluation run."""
        pass

    @abstractmethod
    def add_batch(self, image_ids, batch_preds, batch_gts):
        """Accumulate a single batch of predictions and ground truth."""
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        """Run the metric computation."""
        pass

    @abstractmethod
    def get_full_metrics(self) -> dict:
        """Return the final metric dictionary used by reporting and callbacks."""
        pass
