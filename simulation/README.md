## Agent simulation

Uses a GPT-4 based planner to generate agent trajectories in a simulation environment. The environment consists of a predefined set of simluated tools backed by GPT-4. For example, the [search_internet](tools/search_internet.py) tool pretends to be an internet search engine that generates answers by making a best guess. See package [tools](tools) for all tools available in the simulation environment.

The [GPT-4 based planner](planner.py), used for driving [dataset generation](#dataset-generation), is a ReAct-style planner that interacts with these tools. The generated trajectories of tool interactions are then converted to a fine-tuning dataset for training smaller, open-source LLMs to mimic the behavior of the GPT-4 based planner. See [planner fine-tuning](../train/README.md) for further details on fine-tuning a planner module with the generated dataset.

Fine-tuned planners can also be [evaluated](#planner-evaluation) in the simulation environment. By replacing the GPT-4 based planner with a fine-tuned planner, the agent simulation can be used to evaluate the performance of the fine-tuned planner. It can also be used for evaluating an open-source zero-shot planner that has not been fine-tuned on any simulation data.

### Setup

Add your OpenAI API key to [.env](../.env), then `export PYTHONPATH=.`

### Dataset generation

This section describes generation of a planner fine-tuning dataset in a simulation environment. You can also download the complete output of the following commands from [here](https://martin-krasser.com/gba/gba-output.zip).

#### Generate requests

Start generating requests (random questions and instructions). First 20 requests are intended to be answered directly by the agent:

```shell
python simulation/data/request/single_step.py \
  --output_dir=output/requests \
  --num_requests=20
```

The next 460 batches of 6 requests each are intended to be answered by the agent in multiple steps. Each batch is on a random topic:

```shell
python simulation/data/request/multi_step.py \
  --output_dir=output/requests \
  --num_batches=460 \
  --num_workers=20
```

#### Generate trajectories

Run the agent simulation on each generated request and save the trajectories:

```shell
python simulation/data/trajectory.py \
  --requests_dir=output/requests \
  --output_dir=output/trajectories \
  --direct_answer=simple \
  --num_workers=20
```

The `--direct_answer=simple` option forces the agent answer simple requests directly. These are the 20 single-step requests generated previously. All other requests are answered in multiple steps.

#### Evaluate trajectories

Evaluate the quality of generated trajectories with GPT-4. Generates ratings from 1 (poor) to 5 (excellent):

```shell
python simulation/data/evaluation.py \
  --requests_dir=output/requests \
  --trajectories_dir=output/trajectories \
  --output_dir=output/evaluations \
  --num_workers=20
```

#### Package dataset

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

### Planner evaluation

Instead of running planner evaluatiion yourself, you can also download the complete output of the following commands from [here](https://martin-krasser.com/gba/gba-output-eval.zip). Evaluation is done on a separate, smaller dataset consisting of 20 simple and 30 more complex requests. They can be created with:

```shell
python simulation/data/request/single_step.py \
  --output_dir=output-eval/requests \
  --num_requests=20

python simulation/data/request/multi_step.py \
  --output_dir=output-eval/requests \
  --num_batches=5 \
  --num_workers=5
```

Evaluated are the following planners:

- the GPT-4 based teacher planner used in the previous section but without constraining it to generate direct answers for single-step requests
- a Mistral-7B based fine-tuned student planner
- a Mistral-7B-Instruct based zero-shot planner

Since the simulated `search_internet` and `search_wikipedia` tools are configured to return no answer with a probability of `0.1` or only a partial answer with a probability of `0.1`, four evaluation runs are repeated for each planner.

#### GPT-4 based planner

```shell
for i in {1..4}
do
  python simulation/data/trajectory.py \
    --planner=openai \
    --requests_dir=output-eval/requests \
    --output_dir=output-eval/openai/trajectories_$i \
    --num_workers=20

  python simulation/data/evaluation.py \
    --requests_dir=output-eval/requests \
    --trajectories_dir=output-eval/openai/trajectories_$i \
    --output_dir=output-eval/openai/evaluations_$i \
    --num_workers=20
done
```

#### Fine-tuned planner

Evaluated are [8-bit and 4-bit quantized versions](https://huggingface.co/krasserm/gba-planner-7B-v0.1-GGUF) of the [fine-tuned planner](../train/README.md). The following script shows evaluation of the 8-bit quantized model.

```shell
for i in {1..4}
do
  python simulation/data/trajectory.py \
    --planner=finetuned \
    --planner_host=localhost \
    --planner_port=8082 \
    --requests_dir=output-eval/requests \
    --output_dir=output-eval/finetnued-8bit/trajectories_$i \
    --num_workers=5

  python simulation/data/evaluation.py \
    --requests_dir=output-eval/requests \
    --trajectories_dir=output-eval/finetnued-8bit/trajectories_$i \
    --output_dir=output-eval/finetnued-8bit/evaluations_$i \
    --num_workers=5
done
```

#### Zero-shot planner

Evaluated is an [8-bit quantized version](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) of the zero-shot planner.

```shell
for i in {1..4}
do
  python simulation/data/trajectory.py \
    --planner=zeroshot \
    --planner_host=localhost \
    --planner_port=8081 \
    --requests_dir=output-eval/requests \
    --output_dir=output-eval/zeroshot-8bit/trajectories_$i \
    --num_workers=5

  python simulation/data/evaluation.py \
    --requests_dir=output-eval/requests \
    --trajectories_dir=output-eval/zeroshot-8bit/trajectories_$i \
    --output_dir=output-eval/zeroshot-8bit/evaluations_$i \
    --num_workers=5
done
```

The statistics over these runs are calculated and summarized in [planner_evaluation.ipynb](../planner_evaluation.ipynb).
