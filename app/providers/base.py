"""
Base provider interface for model backends (Gemini, ollama/Gemma, etc.).

Allows swapping between cloud and local model providers without changing
application logic.
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model_override: str | None = None,
    ) -> str:
        """Generate text from a prompt (no images)."""
        pass

    @abstractmethod
    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        model_override: str | None = None,
    ) -> str:
        """Generate text from a prompt with a single image."""
        pass

    @abstractmethod
    def generate_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        model_override: str | None = None,
    ) -> str:
        """Generate text from a prompt with multiple images."""
        pass

    @abstractmethod
    def evaluate_quality(
        self,
        enhanced_prompt: str,
        prompt_type: str | None = None,
        model: str | None = None,
    ) -> dict | None:
        """Evaluate prompt quality and return scores + improvements."""
        pass
