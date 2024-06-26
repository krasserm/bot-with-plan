import json
from functools import partial
from pathlib import Path

import jsonargparse
from datasets import Dataset

from gba.planner.fine_tuned import FineTunedPlanner
from gba.utils import Scratchpad
from simulation.data.trajectory import load_requests


def load_data(
    requests_dir: Path,
    trajectories_dir: Path,
    evaluations_dir: Path,
    rating_threshold: int = 4,
):
    for request, request_id in load_requests(requests_dir):
        trajectory_file = trajectories_dir / f"{request_id}.json"
        evaluation_file = evaluations_dir / f"{request_id}.json"

        if not trajectory_file.exists():
            print(f"Skipping trajectory {request_id} because it does not exist.")
            continue

        if not evaluation_file.exists():
            print(f"Skipping evaluation {request_id} because it does not exist.")
            continue

        with open(trajectory_file, "r") as f:
            trajectory = json.load(f)

        with open(evaluation_file, "r") as f:
            evaluation = json.load(f)

        if evaluation["final_answer_rating"] < rating_threshold:
            print(f"Skipping trajectory {trajectory_file} because of low rating.")
            continue

        scratchpad = Scratchpad()

        for step in trajectory:
            plan = step["plan"]
            result = step["result"]

            del plan["user_request"]
            del plan["missing_information_or_action"]

            task = plan["task"]
            plan_str = json.dumps(plan)

            prompt = FineTunedPlanner.create_messages(request, scratchpad)[0]["content"]
            scratchpad.add(task=task, result=result)

            yield {
                "prompt": prompt,
                "target": plan_str,
            }


def main(args):
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    def format_example(example):
        prompt = example["prompt"]
        target = example["target"]
        return {"text": f"<s>[INST] {prompt} [/INST] {target}</s>"}

    ds = Dataset.from_generator(
        partial(
            load_data,
            requests_dir=args.requests_dir,
            trajectories_dir=args.trajectories_dir,
            evaluations_dir=args.evaluations_dir,
        ),
    )
    ds = ds.map(format_example).train_test_split(test_size=args.validation_size, seed=0)
    ds.save_to_disk(str(args.output_dir))

    ds = ds.remove_columns(["prompt", "target"])
    ds["train"].to_csv(args.output_dir / "train.csv")
    ds["test"].to_csv(args.output_dir / "validation.csv")


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("output", "dataset"))
    parser.add_argument("--requests_dir", type=Path, default=Path("output", "requests"))
    parser.add_argument("--trajectories_dir", type=Path, default=Path("output", "trajectories"))
    parser.add_argument("--evaluations_dir", type=Path, default=Path("output", "evaluations"))
    parser.add_argument("--rating_threshold", type=int, default=4)
    parser.add_argument("--validation_size", type=int, default=30)

    main(parser.parse_args())
