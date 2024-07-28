import os

import jsonargparse
import torch
from datasets import DatasetDict
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, get_kbit_device_map

from train.planner.utils import create_attn_kwargs, create_tokenizer, create_completion_only_collator


def create_model(repo_id, bnb_config=None, lora_config=None, flash_attn=False):
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=bnb_config,
        device_map=get_kbit_device_map(),
        **create_attn_kwargs(flash_attn=flash_attn),
    )
    model = prepare_model_for_kbit_training(model)
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
    return model


def main(args):
    if args.completion_only and args.packing:
        raise ValueError("Cannot use packing with completion_only")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = create_model(args.base_model, bnb_config=bnb_config, lora_config=lora_config, flash_attn=args.flash_attn)
    tokenizer = create_tokenizer(args.base_model)
    collator = create_completion_only_collator(tokenizer) if args.completion_only else None
    dataset = DatasetDict.load_from_disk(args.dataset_dir)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=tokenizer.model_max_length,  # workaround for bug in SFTTrainer when packing = True
        packing=args.packing,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=args.neftune_noise_alpha,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.03,
        save_total_limit=2,
        save_strategy="epoch",
        eval_strategy="epoch",
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
        dataset_text_field="text",
        dataloader_num_workers=2,
        load_best_model_at_end=True,
    )

    sft_trainer = SFTTrainer(
        model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    sft_trainer.train(resume_from_checkpoint=False)
    sft_trainer.save_model()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ---------------------------------------------------------------------------
    #  TODO: use Hugging Face argparser and make all training args configurable
    # ---------------------------------------------------------------------------

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="gba-planner-7B-v0.2")
    parser.add_argument("--dataset_dir", type=str, default="output/dataset")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--neftune_noise_alpha", type=float, default=None)
    parser.add_argument("--completion_only", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--flash_attn", type=bool, default=False)

    main(parser.parse_args())
