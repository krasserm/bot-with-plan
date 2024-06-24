from pathlib import Path

import jsonargparse
import torch
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=bnb_config,
        device_map=args.device,
    )

    for i, example in enumerate(DatasetDict.load_from_disk(str(args.dataset_dir))["test"]):
        prompt = example["prompt"]
        target = example["target"]

        ext_prompt = f"[INST] {prompt} [/INST]"

        input_ids = tokenizer(ext_prompt, return_tensors="pt", max_length=1024, truncation=True)["input_ids"]
        input_ids = input_ids.to(args.device)

        with torch.no_grad():
            result = model.generate(input_ids, generation_config=generation_config)
            result = result[:, input_ids.shape[1] :]

        decoded = tokenizer.batch_decode(result, skip_special_tokens=True)
        decoded = decoded[0]

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
    parser.add_argument("--model_dir", type=Path, default=Path("gba-planner-7B-v0.2"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("output", "dataset"))
    parser.add_argument("--device", type=str, default="cuda:0")
    main(parser.parse_args())
