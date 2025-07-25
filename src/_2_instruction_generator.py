import base64
import json
from typing import List, Dict
from typing import Any
import openai


class EditInstructionGenerator:
    """Generate editing instructions for an image based on desired differences."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        openai.api_key = api_key
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        """Base64 encode an image from disk."""
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
        
    def _encode_image_bytes(self, img_data: Any) -> str:
        """Extract bytes from Parquet cell (dict with 'bytes') and base64 encode."""
        if isinstance(img_data, dict) and "bytes" in img_data:
            return base64.b64encode(img_data["bytes"]).decode("utf-8")
        elif isinstance(img_data, (bytes, bytearray)):
            return base64.b64encode(img_data).decode("utf-8")
        else:
            raise TypeError("Unsupported image data format in Parquet")

    def generate_instructions(self, source_img: Any, difference: str) -> str:
        """Call the API to get editing instructions."""
        source_b64 = self._encode_image_bytes(source_img)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful visual assistant and image editor. "
                    "When provided with a source image context and an ideal difference description, "
                    "you will organize and output the specific editing actions needed to achieve the target image. "
                    "Each action should be concise, use an editing verb, "
                    "and omit any preamble or extra explanation. "
                    "Output must be valid JSON."
                    "The Output should follow the structure that"
                    "each key is a stringified number ('1','2','3',...) and each value is a string describing specific editing action. "
                    "You should answer the question without preamble or additional explanation."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"<difference_description>{difference}</difference_description>"
                            "Please list the editing actions one by one in JSON format, numbered sequentially. " ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{source_b64}"},
                    },
                ],
            },
        ]
        # # Old version openai API call
        # response = openai.ChatCompletion.create(model=self.model, messages=messages)

        # New version openai API call
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def process_batch(self, records: List[Dict[str, object]]) -> List[str]:
        """Generate instructions for a batch of records."""
        results = []
        for rec in records:
            diff_text = (
                rec["difference"]
                if isinstance(rec["difference"], str)
                else json.dumps(rec["difference"], ensure_ascii=False)
            )
            # result is a string of editing instructions with json format (but not json yet)
            results.append(self.generate_instructions(rec["source"], diff_text))
        return results
