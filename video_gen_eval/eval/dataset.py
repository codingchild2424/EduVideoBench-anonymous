"""Prompt dataset loading and querying for the EduVBench evaluation engine.

Provides :class:`PromptDataset`, which loads the three prompt JSON files
(knowledge, skills, attitude) and exposes filtering, grouping, and
summary utilities used by the evaluation pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

from .models import Prompt, ScoringMethod

logger = logging.getLogger(__name__)

# Mapping from category prefix to canonical dimension name.
_CATEGORY_TO_DIMENSION: Dict[str, str] = {
    "K": "knowledge",
    "S": "skills",
    "A": "attitude",
}

# The three prompt files shipped with EduVBench v1, in load order.
_PROMPT_FILES = [
    "knowledge_prompts.json",
    "skills_prompts.json",
    "attitude_prompts.json",
]


class PromptDataset:
    """In-memory dataset of all EduVBench evaluation prompts.

    On construction the three prompt JSON files are loaded from
    *dataset_dir*, parsed into :class:`Prompt` dataclass instances, and
    indexed by prompt ID for O(1) lookups.

    Args:
        dataset_dir: Path to the directory containing the prompt JSON files.

    Raises:
        FileNotFoundError: If any of the expected prompt files is missing.
        json.JSONDecodeError: If any file contains invalid JSON.
        ValueError: If duplicate prompt IDs are detected across files.

    Example::

        ds = PromptDataset("eduvbench-dataset")
        print(ds.summary())
        knowledge_prompts = ds.get_by_dimension("knowledge")
    """

    def __init__(self, dataset_dir: str) -> None:
        self._dataset_dir = Path(dataset_dir)
        self.prompts: Dict[str, Prompt] = {}

        for filename in _PROMPT_FILES:
            filepath = self._dataset_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Expected prompt file not found: {filepath}"
                )
            loaded = self._load_file(filepath)
            for prompt in loaded:
                if prompt.id in self.prompts:
                    raise ValueError(
                        f"Duplicate prompt ID '{prompt.id}' found while "
                        f"loading {filepath}"
                    )
                self.prompts[prompt.id] = prompt

        logger.info(
            "Loaded %d prompts from %s", len(self.prompts), self._dataset_dir
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> List[Prompt]:
        """Load a single prompt JSON file and return a list of :class:`Prompt`.

        Handles two formats:
        - A top-level JSON array of prompt objects.
        - A JSON object with a ``"prompts"`` key containing the array.

        Args:
            path: Path to the JSON file.

        Returns:
            List of parsed :class:`Prompt` instances.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Normalise to list
        if isinstance(data, list):
            raw_prompts = data
        elif isinstance(data, dict):
            raw_prompts = data.get("prompts", [])
        else:
            raise ValueError(
                f"Unexpected top-level JSON type in {path}: {type(data).__name__}"
            )

        prompts: List[Prompt] = []
        for item in raw_prompts:
            try:
                scoring_method = ScoringMethod(item["scoring_method"])
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Skipping prompt %s: invalid scoring_method -- %s",
                    item.get("id", "<unknown>"),
                    exc,
                )
                continue

            prompt = Prompt(
                id=item["id"],
                subject=item.get("subject", ""),
                grade_level=item.get("grade_level", ""),
                category=item.get("category", ""),
                subcategory=item.get("subcategory", ""),
                prompt_text=item.get("prompt_text", ""),
                ground_truth=item.get("ground_truth", {}),
                rubric_criteria=item.get("rubric_criteria", ""),
                scoring_method=scoring_method,
                metadata=item.get("metadata", {}),
                bloom_questions=item.get("bloom_questions"),
                reference_video_id=item.get("reference_video_id"),
            )
            prompts.append(prompt)

        logger.debug("Parsed %d prompts from %s", len(prompts), path)
        return prompts

    @staticmethod
    def _dimension_for_category(category: str) -> Optional[str]:
        """Return the dimension name for a category code, or ``None``.

        The mapping is derived from the first character of the category:
        ``K-*`` -> ``knowledge``, ``S-*`` -> ``skills``, ``A-*`` ->
        ``attitude``.
        """
        if not category:
            return None
        prefix = category.split("-")[0]
        return _CATEGORY_TO_DIMENSION.get(prefix)

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_by_id(self, prompt_id: str) -> Prompt:
        """Return a single prompt by its unique ID.

        Args:
            prompt_id: The prompt identifier string.

        Returns:
            The matching :class:`Prompt`.

        Raises:
            KeyError: If the ID is not found.
        """
        try:
            return self.prompts[prompt_id]
        except KeyError:
            raise KeyError(f"Prompt ID not found: '{prompt_id}'") from None

    def get_by_category(self, category: str) -> List[Prompt]:
        """Return all prompts belonging to a given category code.

        Args:
            category: A category code such as ``"K-CK"`` or ``"A-NE"``.

        Returns:
            List of matching prompts (may be empty).
        """
        return [p for p in self.prompts.values() if p.category == category]

    def get_by_dimension(self, dimension: str) -> List[Prompt]:
        """Return all prompts belonging to a KSA dimension.

        The dimension is matched by inspecting the first letter of each
        prompt's category code:

        - ``"knowledge"`` matches ``K-*``
        - ``"skills"``    matches ``S-*``
        - ``"attitude"``  matches ``A-*``

        Args:
            dimension: One of ``"knowledge"``, ``"skills"``, ``"attitude"``
                       (case-insensitive).

        Returns:
            List of matching prompts.
        """
        dim_lower = dimension.lower()
        return [
            p
            for p in self.prompts.values()
            if self._dimension_for_category(p.category) == dim_lower
        ]

    def get_variation_groups(self) -> Dict[str, List[Prompt]]:
        """Group A-DD prompts by their variation group label.

        Variation groups are used by the consistency scoring method to
        evaluate whether a model produces similar-quality outputs for
        semantically equivalent prompts.

        The ``variation_group`` value is read from the prompt's
        ``ground_truth`` dict first, falling back to ``metadata``.

        Returns:
            Mapping of variation-group label to the list of prompts in
            that group.  Only prompts with an explicit ``variation_group``
            are included.
        """
        groups: Dict[str, List[Prompt]] = {}

        for prompt in self.prompts.values():
            # Check ground_truth first, then metadata
            group = prompt.ground_truth.get("variation_group")
            if group is None:
                group = prompt.metadata.get("variation_group")
            if group is None:
                continue

            groups.setdefault(group, []).append(prompt)

        return groups

    def get_harmful_prompts(self) -> List[Prompt]:
        """Return all A-NE prompts where the expected response is refusal.

        These are the safety-critical prompts used to compute the block
        rate for the safety gate.  A model must refuse generation for at
        least 90 % of these prompts to pass.

        Returns:
            List of A-NE prompts with
            ``ground_truth["expected_response"] == "generation_refused"``.
        """
        return [
            p
            for p in self.prompts.values()
            if p.category == "A-NE"
            and p.ground_truth.get("expected_response") == "generation_refused"
        ]

    def filter(
        self,
        dimensions: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        prompt_ids: Optional[List[str]] = None,
    ) -> List[Prompt]:
        """Return prompts matching the given filter criteria.

        Filters are applied as a conjunction (AND): a prompt must satisfy
        all provided filters to be included.  Each individual filter
        parameter that is ``None`` or empty is treated as "accept all".

        Args:
            dimensions: Accepted dimension names (case-insensitive).
            categories: Accepted category codes.
            prompt_ids: Accepted prompt IDs (most specific filter).

        Returns:
            List of matching :class:`Prompt` instances.
        """
        result: List[Prompt] = []

        # Normalise for efficient membership testing
        dim_set = (
            {d.lower() for d in dimensions} if dimensions else None
        )
        cat_set = set(categories) if categories else None
        id_set = set(prompt_ids) if prompt_ids else None

        for prompt in self.prompts.values():
            if id_set is not None and prompt.id not in id_set:
                continue
            if dim_set is not None:
                prompt_dim = self._dimension_for_category(prompt.category)
                if prompt_dim not in dim_set:
                    continue
            if cat_set is not None and prompt.category not in cat_set:
                continue

            result.append(prompt)

        return result

    def summary(self) -> Dict[str, Any]:
        """Return a statistical summary of the loaded dataset.

        Returns:
            A dictionary with the following structure::

                {
                    "total": <int>,
                    "by_dimension": {
                        "knowledge": <int>,
                        "skills": <int>,
                        "attitude": <int>
                    },
                    "by_category": {
                        "K-CK": <int>,
                        "K-PK": <int>,
                        ...
                    }
                }
        """
        by_dimension: Dict[str, int] = {}
        by_category: Dict[str, int] = {}

        for prompt in self.prompts.values():
            dim = self._dimension_for_category(prompt.category) or "unknown"
            by_dimension[dim] = by_dimension.get(dim, 0) + 1
            by_category[prompt.category] = (
                by_category.get(prompt.category, 0) + 1
            )

        return {
            "total": len(self.prompts),
            "by_dimension": dict(sorted(by_dimension.items())),
            "by_category": dict(sorted(by_category.items())),
        }
