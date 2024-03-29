from pathlib import Path

import jsonargparse
import torch
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig


def main(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        quantization_config=bnb_config,
        device_map=args.device,
    )

    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )

    for i, example in enumerate(DatasetDict.load_from_disk(args.dataset_dir)["test"]):
        # Currently requires an open curly brace at the end of the prompt otherwise
        # the model will generate an EOS token immediately. TODO: investigate ...

        prompt = example["prompt"]
        target = example["target"]

        ext = "{"
        ext_prompt = f"[INST] {prompt} [/INST]" + ext

        input_ids = tokenizer(ext_prompt, return_tensors="pt", max_length=1024, truncation=True)["input_ids"]
        input_ids = input_ids.to(args.device)

        with torch.no_grad():
            result = model.generate(input_ids, generation_config=generation_config)
            result = result[:, input_ids.shape[1]:]

        decoded = tokenizer.batch_decode(result, skip_special_tokens=True)
        decoded = ext + decoded[0]

        # -------------------------------------------------------------
        #  TODO: use an LLM to compare generated plan with target plan
        # -------------------------------------------------------------

        print(f"Example {i} prompt:")
        print(prompt)
        print()

        print(f"Example {i} target:")
        print(target)
        print()

        print(f"Example {i} decoded:")
        print(decoded)
        print()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("project-1", "checkpoint-999"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("output", "dataset"))
    parser.add_argument("--device", type=str, default="cuda:0")
    main(parser.parse_args())
