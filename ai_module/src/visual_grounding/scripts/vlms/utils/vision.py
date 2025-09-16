# Copyright (c) 2025, Urban Robotics Lab. @ KAIST, I Made Aswin Nahrendra
# Copyright (c) 2025 @anahrendra
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
# See LICENSE file in the project root for more information.

import base64
import mimetypes
import numpy as np
from io import BytesIO
from PIL import Image
import os

def encode_image(image_input: str, resize=None) -> str:
    """Encode image to base64

    Args:
        image_path (str): Path to image

    Returns:
        str: Base64 encoded image
    """
    if isinstance(image_input, (str, os.PathLike)):
        mime_type, _ = mimetypes.guess_type(image_input)
        if mime_type is None:
            raise ValueError(f"Cannot determine MIME type from {image_input}")
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise TypeError("image_input must be a file path or PIL.Image.Image")
    
    format = image.format.upper() if image.format else "PNG"
    mime_type = f"image/{'jpeg' if format == 'JPEG' else format.lower()}"
    
    if format == "JPEG" and image.mode == "RGBA":
        image = image.convert("RGB")
        
    if resize is not None:
        image.thumbnail(resize, Image.Resampling.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format=format)
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"