# =======================================================
# src/util/data.py
# Description: Data processing, formatting, and IO utilities
# =======================================================

import os
import json
import yaml
import datetime
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional
from transformers import AutoTokenizer

# --- IO Utilities ---

def load_config_file(path: str) -> dict:
    """
    Load a configuration file from the specified path.
    Supports both JSON (.json) and YAML (.yaml, .yml) formats.
    Returns the configuration as a dictionary, or None if an error occurs.
    """
    try:
        if path.lower().endswith(('.yaml', '.yml')):
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            print(f"Error: Unsupported file extension for '{path}'. "
                  "Supported formats: .json, .yaml, .yml")
            return None

        if not isinstance(config_dict, dict):
            print(f"Error: Configuration file '{path}' does not contain a top-level dictionary.")
            return None

        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        print(f"Error decoding config file '{path}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{path}': {e}")
        return None


def load_json_as_list(input_file: str) -> list:
    """Load a JSON file (or JSON Lines) as a list of dictionaries."""
    res = []
    with open(input_file, 'r', encoding='utf-8') as f:
        # Check if it's a JSON list or JSONL
        try:
            # Try reading as whole JSON
            content = f.read().strip()
            if not content:
                return []
            if content.startswith('['):
                return json.loads(content)
            else:
                # Assume JSONL
                f.seek(0)
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        res.append(json.loads(line))
        except json.JSONDecodeError:
            # Fallback to line-by-line
            f.seek(0)
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    try:
                        res.append(json.loads(line))
                    except:
                        pass
    return res


def create_directory_for_file(file_path: str) -> None:
    """Create the directory for the specified file path if it does not already exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def json_converter(o: Any) -> Any:
    """
    Helper function to serialize non-JSON-compliant types (NumPy, DateTime).
    Usage: json.dump(data, f, default=json_converter)
    """
    if isinstance(o, (np.floating, np.integer)):
        val = float(o)
        return round(val, 4)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, datetime.datetime) or isinstance(o, datetime.date):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def round_floats(o: Any, decimal_places: int = 4) -> Any:
    """
    Recursively rounds all float values in a dictionary or list to the specified number of decimal places.
    """
    if isinstance(o, float):
        return round(o, decimal_places)
    if isinstance(o, dict):
        return {k: round_floats(v, decimal_places) for k, v in o.items()}
    if isinstance(o, list):
        return [round_floats(v, decimal_places) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return round(float(o), decimal_places)
    return o


# --- Text Processing Utilities ---

def normalize_quotes(text: str) -> str:
    """
    Normalizes smart quotes and other typographic characters to their
    ASCII equivalents.
    """
    # Mapping of smart quotes to ASCII quotes
    translation_table = str.maketrans({
        '\u2019': "'",  # Right Single Quotation Mark
        '\u2018': "'",  # Left Single Quotation Mark
        '\u201d': '"',  # Right Double Quotation Mark
        '\u201c': '"',  # Left Double Quotation Mark
        '\u2013': '-',  # En Dash
        '\u2014': '--', # Em Dash
    })
    
    return text.translate(translation_table)


# --- Data Structure Utilities ---

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value
    
def convert_to_dict_pythonic(data):
    dict_data = Vividict()
    for row in data:
        d = dict_data
        for key in row[:-2]:
            d = d[key]
        d[row[-2]] = row[-1]
    return dict_data


# --- Chat Formatter ---

class ChatFormatter:
    def __init__(self, model_name_or_path: str, tokenizer=None):
        """
        Initialize the ChatFormatter.
        
        Args:
            model_name_or_path: Path or name of the model (used if tokenizer is not provided).
            tokenizer: Pre-loaded tokenizer instance (optional, for optimization).
        """
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            
    def format_c4_to_instruct(self, raw_text: str) -> str:
        """
        Wraps C4 raw text into a "continue generation" instruction.
        
        Args:
            raw_text: The truncated raw text from C4.
            
        Returns:
            The formatted prompt string including chat template.
        """
        # Ensure we don't have leading/trailing whitespace issues
        clean_text = raw_text.strip()
        
        # Construct messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Continue the following text naturally:\n{clean_text}..."}
        ]
        
        # Apply template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt

    def format_instruction_dataset(self, instruction: str, input_text: str = "") -> str:
        """
        Formats standard instruction datasets (e.g., Dolly).
        
        Args:
            instruction: The instruction text.
            input_text: Optional input context.
            
        Returns:
            The formatted prompt string.
        """
        content = instruction
        if input_text:
            content += f"\n{input_text}"
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
