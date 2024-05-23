import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Set, Tuple

import jsonargparse
from dotenv import load_dotenv
from tqdm import tqdm

from gba.client import OpenAIClient
from gba.tools import ToolsSpec
from simulation.tools import tools as create_tools

SYSTEM_PROMPT_TEMPLATE = """You are a creative assistant that can generate questions or instructions related to a topic. These questions or instructions are called "requests".
A request is a short sentence or phrase to be answered by an agent in one or more steps using a combination of the following tools, one at each step:

```
{tools}
```

A tool can be used more than once per request. Here are some examples of requests:

```
{examples}
```

It is important that requests are very specific. They must be formulated as if the user has written them."""


USER_PROMPT_TEMPLATE = """Start by generating a topic of your choice from category "{category}" in the following format:

```
Topic: <chosen topic>
```

Then, for each tool in the tool list, think about how it could help the agent answering requests related to this topic. Use the following format:

```
Tool 1: <tool>, your explanation how <tool> could help the agent
...
Tool N: final_answer, your explanation how final_answer could help the agent
```

Finally, generate 6 very specific requests that require an increasing number of steps, from 1 to 6, to be answered.
Avoid generating requests that explicitly instruct the agent to ask the user like "Ask me ...".
Use the following JSON format, including triple backticks:

```json
{{
    "topic": "<chosen topic>",
    "requests": [
      {{
        "request": "<generated request>",
        "explanation": "<explain tool usage steps required for answering this request as compact as possible>"
      }},
      {{
        "request": "<generated request>",
        "explanation": "<explain tool usage steps required for answering this request as compact as possible>"
      }},
      ...
    ]
}}
```

A tool can be used more than once per request. The last step of each request must always use the final_answer tool.
Request 1 should require 1 step, request 2 should require 2 steps, and so on, up to request 6."""


CATEGORIES = [
    "Advertising",
    "Machine Learning",
    "Business Software",
    "Communication",
    "Cryptography",
    "Cybersecurity",
    "Devices",
    "eCommerce",
    "Education",
    "Energy",
    "Entertainment",
    "Events",
    "Finance",
    "Financial",
    "Food",
    "Gaming",
    "Health and Fitness",
    "Jobs",
    "Location",
    "Logistics",
    "Mapping",
    "Media",
    "Medical",
    "Monitoring",
    "Movies",
    "Music",
    "Payments",
    "Science",
    "Search",
    "Social",
    "Sports",
    "Storage",
    "Translation",
    "Transportation",
    "Travel",
    "Weather",
]


EXAMPLES = [
    "what is Leo DiCaprio's current girlfriend's age raised to the 0.24 power?",
    "find the the name of the town Albert Einstein's father was born in",
    "what's the average price of a pepperoni pizza at the 3 closest restaurants?",
    "invite michael@example.com to Alexa's birthday party on 13.12.2024 8pm and tell him to bring a gift",
    "are there any images with the same filename on my computer in the '/data/photos' directory tree?",
    "how did the population of austria's capital grew in the last 20 years?",
    "invite my wife to join me for a walk tomorrow in vienna's largest park",
]


def create_messages(category: str, tools_spec: ToolsSpec):
    examples = EXAMPLES.copy()
    random.shuffle(examples)
    examples.append("...")
    examples_str = "\n".join([f"- {example}" for example in examples])

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(examples=examples_str, tools=tools_spec.tools_repr()),
        },
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(category=category)},
    ]


def choose_category():
    category_idx = random.randint(0, len(CATEGORIES) - 1)
    return CATEGORIES[category_idx]


def generate_request_batch(category: str, idx: int) -> Tuple[str, int]:
    client = OpenAIClient()
    tools = create_tools(client)
    message = client.complete(create_messages(category=category, tools_spec=ToolsSpec(tools)))
    return message["content"], idx


def generated_indices(output_dir: Path) -> Set[int]:
    generated = set()
    for output_file in output_dir.glob("*.txt"):
        idx = int(output_file.stem)
        if idx > 0:
            generated.add(idx)
    return generated


def main(args):
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    # indices start from 1
    _generated_indices = generated_indices(args.output_dir) - {0}
    _remaining_indices = set(range(1, args.num_batches + 1)) - _generated_indices

    futures = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for idx in _remaining_indices:
            futures.append(pool.submit(generate_request_batch, category=choose_category(), idx=idx))

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                content, idx = future.result()
            except Exception:
                print("Failed to generate batch")
                traceback.print_exc()
            else:
                output_file = args.output_dir / f"{idx}.txt"
                with output_file.open("w") as f:
                    f.write(content)


if __name__ == "__main__":
    load_dotenv()

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("output", "requests"))
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=20)

    main(parser.parse_args())
