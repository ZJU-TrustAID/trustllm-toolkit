"""Small random Transformers model for offline plumbing checks, not ASR tests."""

import torch

from .backend import CausalLM


def tiny_backend(*, chat: bool = False) -> CausalLM:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    special = ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(special + alphabet)}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=special[4:],
    )
    if chat:
        tokenizer.chat_template = (
            "{{ bos_token }}{% for message in messages %}"
            "<{{ message['role'] }}>{{ message['content'] }}{{ eos_token }}"
            "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=1024,
                n_ctx=1024,
                n_embd=32,
                n_layer=1,
                n_head=2,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                attn_pdrop=0,
                resid_pdrop=0,
                embd_pdrop=0,
            )
        )
    return CausalLM(model, tokenizer, device="cpu", chat=chat)
