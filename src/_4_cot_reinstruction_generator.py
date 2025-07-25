import base64
import json
from typing import List, Dict
import openai
from typing import Any


class MultiModalAnalysisGenerator:
    """Generate chain-of-thought analysis and revised editing instructions for images."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o"
    ) -> None:
        """
        Args:
            api_key: OpenAI API key.
            model:   Model identifier, e.g., "gpt-4o".
        """
        openai.api_key = api_key
        self.model = model

    def _encode_image_bytes(self, img_data: Any) -> str:
        """Extract bytes from Parquet cell (dict with 'bytes') and base64 encode."""
        if isinstance(img_data, dict) and "bytes" in img_data:
            return base64.b64encode(img_data["bytes"]).decode("utf-8")
        elif isinstance(img_data, (bytes, bytearray)):
            return base64.b64encode(img_data).decode("utf-8")
        else:
            raise TypeError("Unsupported image data format in Parquet")

    def generate(
        self,
        step_image: Any,
        source_image: Any, 
        edit_text: Any
    ) -> str:
        """
        Call the API to get chain-of-thought analysis and re-editing instructions as JSON.

        Args:
            record: A dict with keys:
              - source: str path to source image
              - step_edited: str path to first-step edited image
              - difference: json or str describing differences
              - edits: json/dict of editing instructions

        Returns:
            A JSON-formatted string from the API containing analysis and re-edit instructions.
        """
        # Prepare Base64 images
        step_image_b64 = self._encode_image_bytes(step_image)
        source_image_b64 = self._encode_image_bytes(source_image)


        # Construct messages
        system_prompt = (
            "You are a helpful assistant for visual thinking, design, and editing. "
            "When given source image (the first image), desired editing instruction in JSON, and the first-step edited image (the second image), "
            "you will perform two tasks:"
            "1) Provide a step-by-step chain of thought assessing (a) instruction compliance & subject integrity, "
            "(b) visual realism (geometry, lighting consistency, physical logic), (c)contextual consistency(e.g., scene-element matching, cross-attribute logic, (d)and ethics/safety. "
            "!!!Only note unusual issues if not explicitly instructed, concise without preamble."
            "2) Generate re-editing instructions to refine the first-step result, concise without preamble. "
            "You should answer the question without preamble or additional explanation."
            "The Output should follow the structure that"
            "each keys are a string like 'CoT_1','CoT_2','CoT_3', ... & 'Re_Edit_1', 'Re_Edit_2', ...."
            "and each value is a string of CoT and re-editing description corresponding to the each keys separately. "
        )
        user_text = (
            f"<desired_editing_instruction>{edit_text}</desired_editing_instruction>"
            "Please output a step-by-step chain of thought and re-editing instructions result as required."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{source_image_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{step_image_b64}"}}
                ],
            },
        ]

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

