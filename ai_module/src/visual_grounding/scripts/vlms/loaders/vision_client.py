# Copyright (c) 2025, Urban Robotics Lab. @ KAIST, I Made Aswin Nahrendra
# Copyright (c) 2025 @anahrendra
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
# See LICENSE file in the project root for more information.

import os
from typing import Dict, List, Union, Tuple

from ai_module.src.visual_grounding.scripts.vlms import URL_LLM_ROOT_DIR
from ai_module.src.visual_grounding.scripts.vlms.loaders.client import LlmClient
from ai_module.src.visual_grounding.scripts.vlms.utils.vision import encode_image


def _is_retryable_llm_error(err: Exception) -> bool:
    s = str(err).lower()
    # 흔한 재시도 사유들
    retry_keys = [
        "503", "service unavailable", "unavailable",
        "overloaded", "rate limit", "429",
        "gateway timeout", "504", "502", "500",
        "timeout", "timed out", "temporarily"
    ]
    return any(k in s for k in retry_keys)


class VisionLlmClient(LlmClient):
    """Vision LLM client

    Args:
        LlmClient (_type_): LLM client
    """

    def __init__(self, model_name: str = 'gpt-4o', api_key: str = None):
        """Initialize the LLM client

        Args:
            model_name (str, optional): Model name. Defaults to 'gpt-4o'.
        """
        super().__init__(model_name=model_name, api_key=api_key)

    def construct_message(
        self,
        text_prompt: str,
        image_src: Union[str, List[str]],
        system_instruction: str,
        resize: Union[None, Tuple[int, int]] = None,
        detail: str = 'auto',
        use_url: bool = False,
        **kwargs
    ) -> List[dict]:
        """Construct the prompt for the Vision model

        Args:
            text_prompt (str): Text prompt
            image_src (Union[str, List[str]]): Image source (may be a list of image paths or URLs)
            system_instruction (str): System instruction
            resize (Union[None, Tuple[int, int]]): Resize the image to this size (width, height)
            detail (str): Detail level (low, high, auto)
            use_url (bool): Use URL

        Returns:
            List[dict]: Constructed content

        """
        image_urls = [image_src] if isinstance(image_src, str) else image_src
        image_urls = image_urls if use_url else [encode_image(image_url, resize) for image_url in image_urls]
        
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt,
                }
            ]
        }

        for image_url in image_urls:
            user_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": detail,
                    },
                },
            )
            
        messages = [user_message]
            
        if system_instruction:
            system_message = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_instruction,
                    }
                ]
            }
            messages.insert(0, system_message)
        
        return messages 


if __name__ == "__main__":
    client = VisionLlmClient()

    # Image path input
    image_path = os.path.join(URL_LLM_ROOT_DIR, "assets/dreamstep.png")
    content = client.construct_content(
        text_prompt="Describe the image",
        image_src=image_path,
    )

    message = client.construct_message(role="user", content=content)
    response, chat_completion, usage_log = client.get_response(message, temperature=0.6)
    print(f"Response (str): {response} \n")
    print(f"Usage log (dict): {usage_log} \n")

    # Image URL input
    content = client.construct_content(
        text_prompt="Describe the image",
        image_src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        use_url=True,
    )
    message = client.construct_message(role="user", content=content)
    response, chat_completion, usage_log = client.get_response(message, temperature=0.6)
    print(f"Response (str): {response} \n")
    print(f"Usage log (dict): {usage_log} \n")