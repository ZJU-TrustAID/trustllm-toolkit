import torch

def slice_completion(outputs, input_len, tokenizer, skip_special_tokens=True):
    """
    Slices the completion from the full generation ids and decodes it.
    
    Args:
        outputs (torch.Tensor): Full generation ids (batch or single sequence).
                                If batch, only the first sequence is processed.
        input_len (int): Length of the input prompt (number of tokens).
        tokenizer: Tokenizer for decoding.
        skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.
        
    Returns:
        str: The completion-only text.
    """
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 2:
            sequence = outputs[0]
        else:
            sequence = outputs
    else:
        # Fallback if it's a list or other iterable (e.g. list of lists)
        # Check if first element is iterable to guess if it's 2D
        try:
            if hasattr(outputs[0], '__iter__') or (isinstance(outputs[0], torch.Tensor) and outputs[0].dim() > 0):
                 sequence = outputs[0]
            else:
                 sequence = outputs
        except:
            sequence = outputs

    completion_ids = sequence[input_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)
    return text.strip()

def strip_prompt(text: str, prompt: str) -> str:
    """
    Robustly strips the prompt from the generated text.
    Handles cases where tokenizer decoding might introduce/remove leading spaces.
    """
    if not text:
        return text

    # 1. Exact match
    if text.startswith(prompt):
        return text[len(prompt) :]

    # 2. Match with leading space in text (common artifact)
    if text.startswith(' ' + prompt):
        return text[len(prompt) + 1 :]

    # 3. Match with stripped prompt (if prompt has trailing space but text doesn't)
    if text.startswith(prompt.strip()):
        return text[len(prompt.strip()) :]
        
    markers = [
        'assistant\n\n',
        'Assistant:\n\n',
        'assistant:\n',
        'Assistant:\n',
        '<|start_header_id|>assistant<|end_header_id|>\n\n',  # Llama 3 raw
        '<|im_start|>assistant\n',  # Qwen / ChatML
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            last_idx = text.rfind(marker)
            if last_idx != -1:
                candidate = text[last_idx + len(marker) :]
                return candidate
                
    if len(prompt) > 50:
        suffix = prompt[-50:]
        idx = text.find(suffix)
        if idx != -1 and idx < len(prompt) + 10:  # Allow some tolerance
            return text[idx + len(suffix) :]
            
    return text
