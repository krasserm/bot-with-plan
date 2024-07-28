import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


# Same chat template as mistralai/Mistral-7B-Instruct-v0.2 tokenizer but with an extra space before start of assistant message
TOKENIZER_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"""


def create_tokenizer(repo_id, model_max_length=1024, pad_token_id=0, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.chat_template = TOKENIZER_CHAT_TEMPLATE
    tokenizer.model_max_length = model_max_length
    tokenizer.pad_token_id = pad_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def create_completion_only_collator(tokenizer):
    return DataCollatorForCompletionOnlyLM("[/INST]", tokenizer=tokenizer)


def create_attn_kwargs(flash_attn: bool = False):
    if flash_attn:
        return dict(attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    else:
        return {}
