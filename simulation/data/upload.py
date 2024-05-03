from pathlib import Path

import jsonargparse
from datasets import DatasetDict


def main(args):
    ds = DatasetDict.load_from_disk(args.dataset_dir)
    ds.push_to_hub(f"{args.username}/{args.dataset_name}")
    

if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("output", "dataset"))
    parser.add_argument("--dataset_name", type=Path, default=Path("gba-trajectories"))    
    parser.add_argument("--username", default="krasserm")
    main(parser.parse_args())
