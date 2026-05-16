"""Scorer framework for the EduVBench evaluation engine.

Defines the abstract base class for all scoring methods and a central
registry that maps scoring-method names to concrete scorer instances.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from ..models import Prompt, VideoRef, ItemEvalResult


class BaseScorer(ABC):
    """Abstract base class for all EduVBench scoring methods.

    Every concrete scorer must declare a ``method_name`` property that
    matches one of the :class:`~eduvbench.eval.models.ScoringMethod` values
    and implement the asynchronous :meth:`score` method.
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the canonical scoring-method name (e.g. ``"exact_match"``)."""
        ...

    @abstractmethod
    async def score(
        self,
        prompt: Prompt,
        video_ref: VideoRef,
        vlm_client: Any,
        config: Dict[str, Any],
    ) -> ItemEvalResult:
        """Evaluate a single prompt--video pair and return a normalised result.

        Args:
            prompt:     The evaluation prompt containing ground truth and criteria.
            video_ref:  Reference to the generated video file.
            vlm_client: An instance of
                :class:`~eduvbench.eval.vlm_client.VLMClient` (typed as
                ``Any`` to avoid circular imports).
            config:     Run-level configuration dictionary (thresholds, rubrics,
                        feature flags, etc.).

        Returns:
            An :class:`ItemEvalResult` with a normalised score in [0, 1].
        """
        ...


class ScorerRegistry:
    """Central registry mapping scoring-method names to scorer instances.

    Usage::

        registry = ScorerRegistry()
        registry.register("exact_match", ExactMatchScorer())
        scorer = registry.get("exact_match")
    """

    def __init__(self) -> None:
        self._scorers: Dict[str, BaseScorer] = {}

    def register(self, method: str, scorer: BaseScorer) -> None:
        """Register a scorer instance under the given method name.

        Args:
            method: Canonical scoring-method name (must match
                :attr:`BaseScorer.method_name`).
            scorer: Concrete scorer instance.

        Raises:
            ValueError: If *method* does not match ``scorer.method_name``.
            KeyError: If *method* is already registered.
        """
        if scorer.method_name != method:
            raise ValueError(
                f"Scorer method_name '{scorer.method_name}' does not match "
                f"registration key '{method}'"
            )
        if method in self._scorers:
            raise KeyError(f"Scorer already registered for method '{method}'")
        self._scorers[method] = scorer

    def get(self, method: str) -> BaseScorer:
        """Retrieve the scorer registered for *method*.

        Args:
            method: Scoring-method name.

        Returns:
            The registered :class:`BaseScorer` instance.

        Raises:
            KeyError: If no scorer is registered for *method*.
        """
        if method not in self._scorers:
            raise KeyError(
                f"No scorer registered for method '{method}'. "
                f"Available: {self.available_methods()}"
            )
        return self._scorers[method]

    def available_methods(self) -> List[str]:
        """Return a sorted list of all registered scoring-method names."""
        return sorted(self._scorers.keys())
