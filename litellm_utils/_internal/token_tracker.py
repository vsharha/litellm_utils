"""
Token usage tracking for priority-based model selection.

Tracks token consumption across multiple models to enable automatic
model switching based on budget constraints.
"""

from typing import Optional
import logging

from litellm_utils.models import ModelPriority

logger = logging.getLogger("litellm_utils")


class TokenTracker:
    """
    Tracks token usage across priority-based models.

    Manages token budgets for multiple models and determines which
    model should be used based on remaining budget.

    Attributes:
        priorities: List of ModelPriority configurations (sorted by priority)
        _usage: Dictionary mapping model names to token counts

    Example:
        >>> tracker = TokenTracker([
        ...     ModelPriority(model="gpt-4o-mini", budget=10000, priority=1),
        ...     ModelPriority(model="gpt-4o", budget=50000, priority=2)
        ... ])
        >>> model = tracker.get_available_model()  # Returns "gpt-4o-mini"
        >>> tracker.add_usage("gpt-4o-mini", 5000)
        >>> tracker.get_usage_summary()
    """

    def __init__(self, priorities: list[ModelPriority]):
        """
        Initialize token tracker with priority configurations.

        Args:
            priorities: List of ModelPriority objects defining budgets
        """
        # Sort by priority (lower number = higher priority)
        self.priorities = sorted(priorities, key=lambda p: p.priority)

        # Initialize usage counters
        self._usage = {p.model: 0 for p in self.priorities}

        logger.debug(
            f"Initialized TokenTracker with {len(self.priorities)} models: "
            f"{[p.model for p in self.priorities]}"
        )

    def get_available_model(self) -> Optional[str]:
        """
        Get the next available model within budget.

        Returns the highest-priority model that hasn't exceeded its budget.

        Returns:
            Model identifier, or None if all budgets exhausted

        Example:
            >>> model = tracker.get_available_model()
            >>> if model is None:
            ...     raise RuntimeError("All budgets exhausted")
        """
        for priority in self.priorities:
            used = self._usage[priority.model]
            if used < priority.budget:
                logger.debug(
                    f"Selected model '{priority.model}' "
                    f"(priority {priority.priority}, "
                    f"{used}/{priority.budget} tokens used)"
                )
                return priority.model

        logger.warning("All model budgets exhausted")
        return None

    def add_usage(self, model: str, tokens: int):
        """
        Record token usage for a model.

        Args:
            model: Model identifier
            tokens: Number of tokens consumed

        Example:
            >>> tracker.add_usage("gpt-4o-mini", 1250)
        """
        if model in self._usage:
            self._usage[model] += tokens
            logger.debug(
                f"Added {tokens} tokens to '{model}'. "
                f"Total: {self._usage[model]}"
            )
        else:
            logger.warning(
                f"Attempted to add usage for unknown model '{model}'. "
                f"Known models: {list(self._usage.keys())}"
            )

    def get_usage_summary(self) -> dict:
        """
        Get detailed usage statistics for all models.

        Returns:
            Dictionary with mode and per-model statistics including:
            - model: Model identifier
            - priority: Priority level
            - used_tokens: Tokens consumed
            - budget: Total budget
            - remaining: Remaining tokens
            - percentage: Percentage of budget used

        Example:
            >>> summary = tracker.get_usage_summary()
            >>> print(summary)
            {
                "mode": "priority",
                "models": [
                    {
                        "model": "gpt-4o-mini",
                        "priority": 1,
                        "used_tokens": 8500,
                        "budget": 10000,
                        "remaining": 1500,
                        "percentage": 85.0
                    },
                    ...
                ]
            }
        """
        summary = {
            "mode": "priority",
            "models": []
        }

        for priority in self.priorities:
            model = priority.model
            used = self._usage[model]
            remaining = priority.budget - used
            percentage = (used / priority.budget) * 100 if priority.budget > 0 else 0

            summary["models"].append({
                "model": model,
                "priority": priority.priority,
                "used_tokens": used,
                "budget": priority.budget,
                "remaining": remaining,
                "percentage": round(percentage, 2)
            })

        return summary

    def reset(self):
        """
        Reset all usage counters to zero.

        Useful for starting fresh tracking without recreating the tracker.

        Example:
            >>> tracker.reset()
            >>> tracker.get_usage_summary()  # All usage counts are 0
        """
        self._usage = {model: 0 for model in self._usage}
        logger.info("Reset all token usage counters")

    def get_total_usage(self) -> int:
        """
        Get total tokens used across all models.

        Returns:
            Sum of all token usage

        Example:
            >>> total = tracker.get_total_usage()
            >>> print(f"Total tokens: {total}")
        """
        return sum(self._usage.values())

    def __repr__(self) -> str:
        """Developer-friendly representation"""
        total_used = self.get_total_usage()
        total_budget = sum(p.budget for p in self.priorities)
        return (
            f"TokenTracker(models={len(self.priorities)}, "
            f"used={total_used}/{total_budget})"
        )
