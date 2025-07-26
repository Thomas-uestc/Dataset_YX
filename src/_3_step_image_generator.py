# step_image_editor.py

import os
import base64
from pathlib import Path
from typing import List, Dict
from typing import Any
import openai
import io
import requests
from PIL import Image


class StepImageEditor:
    """Apply only the 'step1' edit instruction to a batch of images using DALL·E Edit."""
    
    def __init__(
        self,
        api_key: str,
        n: int = 1,
    ) -> None:
        
        openai.api_key = api_key
        self.n = n

    def ensure_editable_format(self, image_bytes: bytes, img_format: Any) -> io.BytesIO:
        img = Image.open(io.BytesIO(image_bytes))
        buffer = io.BytesIO()
        buffer.name = "image.png"
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer
    
    # if using DALL-E Edit API, we need to create a mask image
    def create_mask(self, image_bytes: bytes):
         img = Image.open(io.BytesIO(image_bytes))
         w, h = img.size
         mask = Image.new("L", (w, h), 0)
         mask_buffer = io.BytesIO() 
         mask_buffer.name = "mask.png"
         mask.save(mask_buffer, format="PNG")
         mask_buffer.seek(0)
         return mask_buffer  

    def apply_step(self, source_image: Any, edit_text: Any, width: int, height: int, img_format: Any) -> str:
        
        image_file = self.ensure_editable_format(source_image, img_format)

        resp = openai.images.edit(
            model = "gpt-image-1",
            image=image_file,
            prompt = edit_text,
            n=self.n,
        ) 

        image_base64 = resp.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)


        return image_bytes
