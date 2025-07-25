import base64
from typing import List
from typing import Any
import openai


class DifferenceDescriptionGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        """Initialize the generator with the OpenAI API key and model."""
        openai.api_key = api_key
        self.model = model

    def _encode_image_path(self, image_path: str) -> str:
        """Read and base64 encode an image from disk."""
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

    def describe_difference(self, source_img: Any, target_img: Any) -> str:
        """Call the API to describe differences between two images."""
        source_b64 = self._encode_image_bytes(source_img)
        target_b64 = self._encode_image_bytes(target_img)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful visual assistant. "
                    "When given two images, you will analyze and describe their differences "
                    "according to categories like object, style, color, motion, 2D & 3D spatial, "
                    "texture, and shape differences. "
                    "If a category has no obvious difference, omit it."   
                    "You should answer the question without preamble or additional explanation."
                    "You must output your answer as a single valid JSON object: "
                    "each key is a stringified number ('1','2','3',...) and each value is a string describing one difference. "
                    "Do not include Markdown formatting, code blocks, or any explanation outside the JSON. "
                    "Ensure the JSON is valid and parsable."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I have two similar images: the first is the source image, the second is the target image. "
                            "Please describe the differences between them as defiend format requirments. "
                            "providing one paragraph per category, numbered sequentially. "
                            "Each paragraph should be a single sentence starting with “<sequence number>: There is a difference at <category>, …”."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{source_b64}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{target_b64}"
                        },
                    },
                ],
            },
        ]

        # # Old version openai API call
        # response = openai.ChatCompletion.create(
        #     model=self.model,
        #     messages=messages,
        # )
        # return response.choices[0].message.content
    
        # New version of openai API call
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def process_batch(self, source_images: List[str], target_images: List[str]) -> List[str]:
        """Process lists of images and return a list of JSON difference descriptions."""
        if len(source_images) != len(target_images):
            raise ValueError("source_paths and target_paths must have the same length")

        results = []
        for src, tgt in zip(source_images, target_images):
            results.append(self.describe_difference(src, tgt))
        return results
