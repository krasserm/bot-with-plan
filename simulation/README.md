## Agent simulation

GPT-4 based agent simulation is used for creating a fine-tuning dataset for the planner module.  

### Setup

Add your OpenAI API key to `.env`, then `export PYTHONPATH=.`

### Generate requests 

Start generating requests (random questions and instructions). First 20 requests are intended to be answered directly 
by the agent:

```shell
python simulation/data/request/single_step.py \
  --output_dir=output/requests \
  --num_requests=20
```

The next 460 batches of 6 requests each are intended to be answered by the agent in multiple steps:

```shell
python simulation/data/request/multi_step.py \
  --output_dir=output/requests \
  --num_batches=460 \
  --num_workers=20
```

### Generate trajectories

Run the simulation agent on each generated request and save the trajectories:  

```shell
python simulation/data/trajectory.py \
  --requests_dir=output/requests \
  --output_dir=output/trajectories \
  --num_workers=20
```

### Evaluate trajectories

Evaluate the quality of generated trajectories:  

```shell
python simulation/data/evaluation.py \
  --requests_dir=output/requests \
  --trajectories_dir=output/trajectories \
  --output_dir=output/evaluations \
  --num_workers=20
```

### Package dataset

Package a fine-tuning dataset for the planner module from generated trajectories that have a rating of 4 or higher:  

```shell
python simulation/data/package.py \
  --requests_dir=output/requests \
  --trajectories_dir=output/trajectories \
  --evaluations_dir=output/evaluations \
  --output_dir=output/dataset \
  --validation_size=5 \
  --rating_threshold=4
```
