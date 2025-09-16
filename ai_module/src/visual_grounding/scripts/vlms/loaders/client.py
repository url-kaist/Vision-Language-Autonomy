# Copyright (c) 2025, Urban Robotics Lab. @ KAIST, I Made Aswin Nahrendra
# Copyright (c) 2025 @anahrendra
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
# See LICENSE file in the project root for more information.

import json
import os
import pkgutil
from typing import List, Tuple

from openai import OpenAI

from ai_module.src.visual_grounding.scripts.vlms import CONFIG

class LlmClient:
    def __init__(
        self,
        model_name: str = 'gpt-4o',
        api_key: str = None, 
        
    ):
        """Initialize the LLM client

        Args:
            model_name (str, optional): Model name. Defaults to 'gpt-4o'.
        """
        self.model_name = model_name
        if 'gemini' in model_name:
            service = 'GEMINI'
        else:
            service = 'OPENAI'

        base_url = CONFIG[service]['BASE_URL']
        self.pricing = CONFIG[service]['PRICING']
        if api_key is None:
            api_key = CONFIG[service]['API_KEY']

        self.gpt = OpenAI(api_key=api_key, base_url=base_url)

        self.use_reasoning_model = True if 'o1' in self.model_name else False

    def handshake(self):
        """Handshake with the GPT model to check the connection"""
        messages = [
            {
                "role": "user",
                "content": "Tell me what model are you using now",
            }
        ]

        response, chat_completion, usage_log = self.get_response(messages, temperature=0.6)

        print(f"Response (str): {response} \n")
        print(f"Chat completion (dict): {chat_completion} \n")
        print(f"Usage log (dict): {usage_log} \n")

    def get_response(
        self,
        messages: list,
        temperature: float,
        reasoning_effort: str = 'medium',
        **kwargs,
    ) -> Tuple[str, dict, dict]:
        """Get response from the GPT model

        Args:
            messages (list): List of messages
            temperature (float): Temperature for sampling
            reasoning_effort (str): Reasoning effort for the reasoning model, 'low', 'medium', or 'high'

        Returns:
            Tuple[str, dict, dict]: Tuple of response, chat completion, and usage dictionary
        """

        messages, temperature = self.verify_input(
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        chat_completion = self.gpt.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=temperature,
        )
        response = chat_completion.choices[0].message.content
        usage_log = self.log_usage(chat_completion, temperature, reasoning_effort)
        return response, chat_completion, usage_log

    def verify_input(
        self,
        messages: list,
        temperature: float,
        reasoning_effort: str,
    ) -> Tuple[List[dict], float]:
        """Verify and refactor inputs to:
        1. use 'user' role and assign 'reasoning_effort' key when using reasoning model
        2. set temperature to 1.0 when using reasoning model

        Args:
            messages (list): List of messages
            temperature (float): Temperature for sampling
            reasoning_effort (str): Reasoning effort for the reasoning model, 'low', 'medium', or 'high'

        Returns:
            List[dict]: Refactored messages
        """
        if self.use_reasoning_model:
            temperature = 1.0
            for message in messages:
                if message['role'] == 'system':
                    message['role'] = 'user'
                message['reasoning_effort'] = reasoning_effort

        return messages, temperature

    def load_prompts(self, package_name: str, prompt_name: str) -> str:
        """Load prompt from a package

        Args:
            package_name (str): Package name
            prompt_name (str): Prompt name in a text file

        Returns:
            str: Prompt content
        """
        data = pkgutil.get_data(package_name, prompt_name)
        return data.decode("utf-8")

    def load_multiline_prompts(self, package_name: str, prompt_name: str) -> List[str]:
        """Load multiline prompts from a package

        Args:
            package_name (str): Package name
            prompt_name (str): Prompt name in a text file

        Returns:
            List[str]: List of prompts
        """
        multiline_prompts = self.load_prompts(package_name, prompt_name)
        return multiline_prompts.split("\n")

    def json_to_dict(self, json_data: str) -> dict:
        """Convert JSON string to dictionary

        Args:
            json_data (str): JSON string

        Returns:
            dict: Dictionary
        """
        return json.loads(json_data)

    def log_usage(self, chat_completion: dict, temperature: float, reasoning_effort: str) -> dict:
        """Log the usage of the GPT model

        Args:
            chat_completion (dict): Chat completion
            temperature (float): Temperature for sampling
            reasoning_effort (str): Reasoning effort for the reasoning model

        Returns:
            dict: Usage dictionary
        """
        input_cost = chat_completion.usage.prompt_tokens * self.pricing[chat_completion.model]['input']
        output_cost = chat_completion.usage.completion_tokens * self.pricing[chat_completion.model]['output']
        total_cost = input_cost + output_cost

        usage_dict = {
            'model': chat_completion.model,
            'temperature': temperature,
            'reasoning_effort': reasoning_effort if self.use_reasoning_model else None,
            'completion_tokens': chat_completion.usage.completion_tokens,
            'prompt_tokens': chat_completion.usage.prompt_tokens,
            'total_tokens': chat_completion.usage.total_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
        }
        return usage_dict

    def response_to_txt(self, response, path: str = None) -> None:
        """Convert response to text file

        Args:
            response (str): Response string
            path (str, optional): Path to save the text file. Defaults to None.

        """
        response_path = "response.txt" if path is None else os.path.join(path, "response.txt")
        with open(response_path, "w") as f:
            f.write(response)
        print(f"Response saved to {response_path}")