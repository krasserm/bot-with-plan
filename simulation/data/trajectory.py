import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import jsonargparse
from dotenv import load_dotenv
from tqdm import tqdm

from gba.client import ChatClient, LlamaCppClient, MistralInstruct, OpenAIClient
from gba.planner import FineTunedPlanner, ZeroShotPlanner
from gba.tools import ToolsSpec
from gba.utils import extract_json
from simulation.agent import Agent
from simulation.planner import OpenAIPlanner
from simulation.tools import tools as create_tools

Trajectory = List[Dict]


def run_agent(agent: Agent, request: str, request_id: str, direct_answer: bool) -> Tuple[Trajectory, str]:
    plans, scratchpad_entries = agent.run(request, direct_answer=direct_answer, verbose=False)

    trajectory = []
    for plan, entry in zip(plans, scratchpad_entries):
        trajectory.append(
            {
                "plan": plan.to_dict(),
                "result": entry.result,
            }
        )
    return trajectory, request_id


def load_requests(requests_dir: Path) -> Iterator[Tuple[str, str]]:
    for requests_file in requests_dir.glob("*.txt"):
        try:
            with requests_file.open("r") as f:
                output_json = extract_json(f.read())
                requests_json = output_json["requests"]
        except Exception:
            print(f"Failed to parse requests file {requests_file}")
            traceback.print_exc()
            continue

        for i, entry in enumerate(requests_json):
            request = entry["request"]
            request_id = f"{requests_file.stem}_{i}"
            yield request, request_id


def create_client(planner_host: str, planner_port: int):
    url = f"http://{planner_host}:{planner_port}/completion"
    model = MistralInstruct(llm=LlamaCppClient(url=url, temperature=-1))
    return ChatClient(model=model)


def main(args):
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = []

        for request, request_id in load_requests(args.requests_dir):
            output_file = args.output_dir / f"{request_id}.json"
            if output_file.exists():
                continue

            if args.direct_answer == "simple":
                # requests in 0.txt must be answered directly
                direct_answer = request_id.startswith("0_")
            else:
                direct_answer = args.direct_answer == "on"

            client = OpenAIClient()
            tools = create_tools(client=client)
            tools_spec = ToolsSpec(tools=tools)

            if args.planner == "openai":
                planner = OpenAIPlanner(client=client, tools_spec=tools_spec)
            elif args.planner == "finetuned":
                planner = FineTunedPlanner(client=create_client(args.planner_host, args.planner_port))
            elif args.planner == "zeroshot":
                planner = ZeroShotPlanner(
                    client=create_client(args.planner_host, args.planner_port),
                    tools_spec=tools_spec,
                )
            else:
                raise ValueError(f"Unknown planner type'{args.planner}'")

            future = pool.submit(
                run_agent,
                agent=Agent(planner=planner, tools=tools),
                request=request,
                request_id=request_id,
                direct_answer=direct_answer,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                trajectory, request_id = future.result()
            except Exception:
                print("Failed to generate trajectory")
                traceback.print_exc()
            else:
                output_file = args.output_dir / f"{request_id}.json"
                with output_file.open("w") as f:
                    json.dump(trajectory, f, indent=2)


if __name__ == "__main__":
    load_dotenv()

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--planner", type=str, choices=["openai", "finetuned", "zeroshot"], default="openai")
    parser.add_argument("--planner_host", type=str, default="localhost")
    parser.add_argument("--planner_port", type=int, default=8082)
    parser.add_argument("--output_dir", type=Path, default=Path("output", "trajectories"))
    parser.add_argument("--requests_dir", type=Path, default=Path("output", "requests"))
    parser.add_argument("--direct_answer", type=str, default="off", choices=["off", "simple", "on"])
    parser.add_argument("--num_workers", type=int, default=20)
    main(parser.parse_args())
