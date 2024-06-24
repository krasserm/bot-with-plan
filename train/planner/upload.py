from pathlib import Path

import jsonargparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    repo_id = f"{args.username}/{args.model_dir.name}"

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=Path("gba-planner-7B-completion-only-v0.2"))
    parser.add_argument("--username", default="krasserm")
    main(parser.parse_args())
