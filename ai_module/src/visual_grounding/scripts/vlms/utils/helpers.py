# Copyright (c) 2025, Urban Robotics Lab. @ KAIST, I Made Aswin Nahrendra
# Copyright (c) 2025 @anahrendra
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
# See LICENSE file in the project root for more information.

import json

def parse_json(json_output: str) -> str:
    """Parse the JSON output from the LLM response.

    Args:
        json_output (str): The JSON output string from the LLM response.

    Returns:
        str: The parsed JSON object, or False if parsing fails.
    """

    try:
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i + 1 :])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        data = json.loads(json_output)
        return data
    except ValueError:
        return False

def removeprefix(s: str, prefix: str) -> str:
    """Remove the specified prefix from a string if it exists.

    Args:
        s (str): The string from which to remove the prefix.
        prefix (str): The prefix to remove from the string.

    Returns:
        str: The string with the prefix removed if it was present, otherwise the original string.
    """
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s