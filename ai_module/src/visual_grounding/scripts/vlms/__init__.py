import os
import json
from dotenv import load_dotenv

load_dotenv()

URL_LLM_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

tokens_per_dollar = 1e6
OPENAI_PRICING = {
    'gpt-4o': {'input': 2.5 / tokens_per_dollar, 'output': 10.0 / tokens_per_dollar},
    'gpt-4o-2024-11-20': {'input': 2.5 / tokens_per_dollar, 'output': 10.0 / tokens_per_dollar},
    'gpt-4o-2024-08-06': {'input': 2.5 / tokens_per_dollar, 'output': 10.0 / tokens_per_dollar},
    'gpt-4o-mini': {'input': 0.15 / tokens_per_dollar, 'output': 0.075 / tokens_per_dollar},
    'gpt-4o-mini-2024-07-18': {'input': 0.15 / tokens_per_dollar, 'output': 0.075 / tokens_per_dollar},
    'o1-mini': {'input': 1.1 / tokens_per_dollar, 'output': 0.55 / tokens_per_dollar},
    'o1-mini-2024-09-12': {'input': 1.1 / tokens_per_dollar, 'output': 0.55 / tokens_per_dollar},
}

GEMINI_PRICING = {
    'gemini-2.0-flash': {'input': 0.1 / tokens_per_dollar, 'output': 0.4 / tokens_per_dollar},
    'gemini-2.0-flash-lite': {'input': 0.1 / tokens_per_dollar, 'output': 0.4 / tokens_per_dollar},
    'gemini-2.5-flash': {'input': 0.1 / tokens_per_dollar, 'output': 0.4 / tokens_per_dollar},
}


def model_name_to_type(model_name):
    if 'gemini' in model_name:
        return 'GEMINI'
    else:
        return 'OPENAI'


CONFIG = {
    'OPENAI': {
        'API_KEY': os.getenv("OPENAI_API_KEY"),
        'API_KEY_NAME': "OPENAI_API_KEY",
        'BASE_URL': None,
        'PRICING': OPENAI_PRICING,
    },
    'GEMINI': {
        'API_KEY': os.getenv("GOOGLE_API_KEY"),
        'API_KEY_NAME': "GOOGLE_API_KEY",
        'BASE_URL': "https://generativelanguage.googleapis.com/v1beta/openai/",
        'PRICING': GEMINI_PRICING,
    },
}