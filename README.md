## Modular open LLM agents via prompt chaining and schema-guided generation

- [From monolithic to modular open LLM agents - A system of experts via prompt chaining](example_agent_zeroshot.ipynb) ([blog post](https://krasserm.github.io/2024/03/06/modular-agent/)).
Demonstrates how prompt chaining can lead to useful agentic behavior even when used with smaller open LLMs. 
To reliably extract specific pieces of information generated by one module and use it as input for another module, 
this article makes heavy use of schema-guided generation ([JSON mode](example_json.ipynb)).

- [Reliable JSON mode for open LLMs](example_json.ipynb) ([blog post](https://krasserm.github.io/2023/12/18/llm-json-mode/)).
Implements JSON mode for open LLMs by enforcing structured output via schema-guided generation. Users specify the 
output JSON schema via [pydantic](https://docs.pydantic.dev/) models passed as arguments to completion calls. The 
schema is converted into a grammar, used by llama.cpp for [constrained decoding](https://github.com/ggerganov/llama.cpp/pull/1773).

### Work in progress

- [Agent simulation](simulation/README.md)
- [Planner fine-tuning](train/README.md)
- Tool improvements 

### Previous work

- [Open LLM agents with schema-guided generation of function calls](https://github.com/krasserm/grammar-based-agents/blob/wip-article-1/example_agent.ipynb) ([blog post](https://krasserm.github.io/2023/12/10/grammar-based-agents/)).
Initial experiments with a Llama-2-70B-Chat model to generate function calls conforming to a JSON schema via [constrained decoding](https://github.com/ggerganov/llama.cpp/pull/1773).
Implements a monolithic, zero-shot prompted LLM agent compatible with LangChain's [agent framework](https://python.langchain.com/docs/modules/agents/). 

## Getting started

### [Zero-shot modular agent](example_agent_zeroshot.ipynb) example

```shell
mkdir -p models

wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf?download=true \
  -O models/mistral-7b-instruct-v0.2.Q8_0.gguf

wget https://huggingface.co/TheBloke/NexusRaven-V2-13B-GGUF/resolve/main/nexusraven-v2-13b.Q8_0.gguf?download=true \
  -O models/nexusraven-v2-13b.Q8_0.gguf

wget https://huggingface.co/krasserm/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf?download=true \
  -O models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

```shell
docker run --gpus all --rm -p 8081:8080 -v $(realpath models):/models ghcr.io/ggerganov/llama.cpp:server-cuda--b1-3fe847b \
  -m /models/mistral-7b-instruct-v0.2.Q8_0.gguf -c 1024 --n-gpu-layers 33 --host 0.0.0.0 --port 8080

docker run --gpus all --rm -p 8084:8080 -v $(realpath models):/models ghcr.io/ggerganov/llama.cpp:server-cuda--b1-3fe847b \
  -m /models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -c 2048 --n-gpu-layers 33 --host 0.0.0.0 --port 8080

docker run --gpus all --rm -p 8089:8080 -v $(realpath models):/models ghcr.io/ggerganov/llama.cpp:server-cuda--b1-3fe847b \
  -m /models/nexusraven-v2-13b.Q8_0.gguf --n-gpu-layers 41 --host 0.0.0.0 --port 8080

```

### [Fine-tuned modular agent](example_agent_finetuned.ipynb) example


```shell
mkdir -p models

wget https://huggingface.co/krasserm/gba-planner-7B-v0.1-GGUF/resolve/main/gba-planner-7B-v0.1-Q8_0.gguf?download=true \
  -O models/gba-planner-7B-v0.1-Q8_0.gguf

wget https://huggingface.co/TheBloke/NexusRaven-V2-13B-GGUF/resolve/main/nexusraven-v2-13b.Q8_0.gguf?download=true \
  -O models/nexusraven-v2-13b.Q8_0.gguf

wget https://huggingface.co/krasserm/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf?download=true \
  -O models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

```shell
docker run --gpus all --rm -p 8082:8080 -v $(realpath models):/models ghcr.io/ggerganov/llama.cpp:server-cuda--b1-3fe847b \
  -m /models/gba-planner-7B-v0.1-Q8_0.gguf -c 1024 --n-gpu-layers 33 --host 0.0.0.0 --port 8080

docker run --gpus all --rm -p 8084:8080 -v $(realpath models):/models ghcr.io/ggerganov/llama.cpp:server-cuda--b1-3fe847b \
  -m /models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -c 2048 --n-gpu-layers 33 --host 0.0.0.0 --port 8080

docker run --gpus all --rm -p 8089:8080 -v $(realpath models):/models ghcr.io/ggerganov/llama.cpp:server-cuda--b1-3fe847b \
  -m /models/nexusraven-v2-13b.Q8_0.gguf --n-gpu-layers 41 --host 0.0.0.0 --port 8080
```

### [JSON mode](example_json.ipynb) example

```shell
mkdir -p models

wget https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/resolve/main/llama-2-70b-chat.Q4_0.gguf?download=true \
  -O models/llama-2-70b-chat.Q4_0.gguf

wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf?download=true \
  -O models/mistral-7b-instruct-v0.1.Q8_0.gguf
```

```shell
docker run --gpus all --rm -p 8080:8080 -v $(realpath models):/models ghcr.io/ggerganov/llama.cpp:server-cuda-052051d8ae4639a1c3c61e7da3237bcc572469d4 \
  -m /models/llama-2-70b-chat.Q4_0.gguf --n-gpu-layers 83 --host 0.0.0.0 --port 8080

docker run --gpus all --rm -p 8081:8080 -v $(realpath models):/models ghcr.io/ggerganov/llama.cpp:server-cuda-052051d8ae4639a1c3c61e7da3237bcc572469d4 \
  -m /models/mistral-7b-instruct-v0.1.Q8_0.gguf --n-gpu-layers 33 --host 0.0.0.0 --port 8080
```

### Create environment

```shell
conda env create -f environment.yml
conda activate grammar-based-agents
```

If you additionally want to finetune the planner module:

```shell
conda env create -f environment-autotrain.yml
conda activate grammar-based-agents-autotrain
```
