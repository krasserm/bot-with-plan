import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jsonargparse
from dotenv import load_dotenv
from tqdm import tqdm

from gba.client import OpenAIClient
from simulation.data.trajectory import Trajectory, load_requests

TRAJECTORY_TEMPLATE = """Request:

```
{request}
```

Steps:

```
{steps}
```

Answer:

```
{answer}
```"""


SYSTEM_PROMPT = """Your are an expert in evaluating the steps that an agent has taken to answer a request.
A request can be either a question or instruction. Steps are given in the following format:

```
Task: <task>
Tool: <tool>
Result: <result>
```

Your goal is to rate each step and the final answer with a score from 1 to 5. Provide your answer in the following JSON format:

{
  "steps": [
    {
      "task_rating": your rating if the task description is a useful next step given the previous steps,
      "tool_rating": your rating if the selected tools is appropriate for the task,
      "result_rating": your rating how well the result answers the task,
      "explanation": your brief explanation for these ratings
    },
    ...
  ],
  "final_answer_rating": your rating how well the final answer answers the initial request,
  "explanation": your brief explanation for the final answer rating
}"""


USER_PROMPT_TEMPLATE = """{trajectory}

Begin!"""


def format_trajectory(trajectory: Trajectory, request: str) -> str:
    answer = trajectory[-1]["result"]
    steps = []

    for step in trajectory[:-1]:
        plan = step["plan"]
        tool = plan["selected_tool"]
        task = plan["task"]
        result = step["result"]

        if "user_request" in plan and plan["user_request"] != request:
            print("Repeated request doesn't match the original request:")
            print(f"- Original request: {request}")
            print(f"- Repeated request: {plan['user_request']}")

        steps.append(f"Task: {task}\nTool: {tool}\nResult: {result}")

    return TRAJECTORY_TEMPLATE.format(
        request=request,
        steps="\n\n".join(steps),
        answer=answer,
    )


def evaluate(trajectory: Trajectory, request: str, request_id: str):
    formatted_trajectory = format_trajectory(trajectory, request)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(trajectory=formatted_trajectory)},
    ]

    message = OpenAIClient().complete(messages, enforce_json_output=True, temperature=0.0)
    return json.loads(message["content"]), formatted_trajectory, request_id


def main(args):
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = []

        for request, request_id in load_requests(args.requests_dir):
            output_file = args.output_dir / f"{request_id}.json"
            if output_file.exists():
                continue

            trajectory_file = args.trajectories_dir / f"{request_id}.json"
            if not trajectory_file.exists():
                continue

            with trajectory_file.open("r") as f:
                trajectory = json.load(f)

            if len(trajectory) == 0:
                continue

            future = pool.submit(evaluate, trajectory=trajectory, request=request, request_id=request_id)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                evaluation, formatted_trajectory, request_id = future.result()
            except Exception:
                print("Failed to evaluate trajectory")
                traceback.print_exc()
            else:
                with open(args.output_dir / f"{request_id}.json", "w") as f:
                    json.dump(evaluation, f, indent=2)

                if args.output_formatted_trajectories:
                    with open(args.output_dir / f"{request_id}.txt", "w") as f:
                        f.write(formatted_trajectory)


if __name__ == "__main__":
    load_dotenv()

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("output", "evaluations"))
    parser.add_argument("--output_formatted_trajectories", type=bool, default=False)
    parser.add_argument("--requests_dir", type=Path, default=Path("output", "requests"))
    parser.add_argument("--trajectories_dir", type=Path, default=Path("output", "trajectories"))
    parser.add_argument("--num_workers", type=int, default=20)

    main(parser.parse_args())
