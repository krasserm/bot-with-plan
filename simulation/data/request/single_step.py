from pathlib import Path

import jsonargparse
from dotenv import load_dotenv

from gba.client import OpenAIClient


SYSTEM_PROMPT = """You are a creative assistant that can generate questions or instructions on a variety of topics. These questions or instructions are called "requests".
A request is a short sentence or phrase that can be answered with basic common knowledge in 1 or 2 sentences."""


USER_PROMPT_TEMPLATE = """Generate {num} very specific and simple requests. Be creative when choosing a topic for each request. 
Half of the requests should be questions and the other half should be instructions. 
The assistant that will answer these requests can only generate text, no drawings, sounds, ... etc.

Use the following JSON format:

```json
{{
    "requests": [
      {{ "request": "<request>", "topic": "<the topic you have chosen>" }},
      ...
    ]
}}
```"""


def generate_request_batch(num_requests: int):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(num=num_requests)},
    ]
    response = OpenAIClient().complete(messages)
    return response["content"]


def main(args):
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    output_file = args.output_dir / "0.txt"

    if not output_file.exists():
        with output_file.open("w") as f:
            f.write(generate_request_batch(num_requests=args.num_requests))


if __name__ == "__main__":
    load_dotenv()

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("output", "requests"))
    parser.add_argument("--num_requests", type=int, default=20)

    main(parser.parse_args())
