## Open LLM agents with schema-guided generation of function calls

LLM agents use large language models (LLMs) to decide which tools to use for interacting with their 
environment during multi-step task solving. Tool usage is done via function calling. The LLM is 
prompted or fine-tuned to generate JSON output representing one or more function calls. To ensure 
that the JSON output of an LLM follows a tool-specific schema one can use constrained decoding 
methods to control next token generation.

Here, [grammar-based sampling](https://github.com/ggerganov/llama.cpp/pull/1773) available in llama.cpp
is used for constrained decoding. A JSON schema of available tools is converted to a grammar used for 
generating instances of that schema (= tool calls) during decoding. This is used for implementing a 
function calling interface for a Llama-2-70b model, an LLM with (limited) tool usage capabilities. 

The [implementation](https://github.com/krasserm/grammar-based-agents) uses LangChain interfaces and 
is compatible LangChain's [agent framework](https://python.langchain.com/docs/modules/agents/). In its 
current state, it is a simple prototype for demonstrating schema-based generation in LangChain agents. 
It is general enough to be used with many other language models supported by llama.cpp, after some 
tweaks to the prompt templates.

More details in [example_agent.ipynb](example_agent.ipynb).

### Tools

The tools used in [example_agent.ipynb](example_agent.ipynb) are mainly [mockups](example_tools.py) at the moment, 
except the `calculate` tool which is backed by [Llama2Math](gba/math.py) for interpreting and
evaluating mathematical queries using an LLM. 

See [example_math.ipynb](example_math.ipynb) for more details.

### JSON mode

Components in this repository can also be used to run LLMs in so-called JSON mode. JSON mode is
a narrower concept than tool calling and can be used to constrain model output to a user-defined
schema (which can be the signature of a tool but also any other JSON schema) whereas tool calling 
relies on an LLM to select an appropriate tool out of several provided tools, or even to decide
not calling a function at all.

See [example_json.ipynb](example_json.ipynb) for more details.

## Getting started

### Download models

```shell
mkdir models

# Download agent LLM Llama-2-70B
wget https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/resolve/main/llama-2-70b-chat.Q4_0.gguf?download=true -O models/llama-2-70b-chat.Q4_0.gguf

# Download math LLM CodeLlama-7B
wget https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf?download=true -O models/codellama-7b-instruct.Q4_K_M.gguf

# Download math Mistral-7B-Instruct-v0.1
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf?download=true -O model/mistral-7b-instruct-v0.1.Q8_0.gguf
```

### Docker image

Either build a [CUDA-enabled llama.cpp Docker image](https://github.com/ggerganov/llama.cpp/blob/master/README.md#docker-with-cuda) yourself (`full-cuda` variant) or use the pre-build
[ghcr.io/krasserm/llama.cpp:full-cuda](https://github.com/krasserm/grammar-based-agents/pkgs/container/llama.cpp) Docker image as done in the next section .

### Run LLM servers

#### Llama-2

```shell
docker run --gpus all --rm -p 8080:8080 -v $(realpath models):/models ghcr.io/krasserm/llama.cpp:full-cuda \
  --server -m /models/llama-2-70b-chat.Q4_0.gguf --n-gpu-layers 83 --host 0.0.0.0 --port 8080
```

#### Mistral

```shell
docker run --gpus all --rm -p 8081:8080 -v /home/martin/Models:/models ghcr.io/krasserm/llama.cpp:full-cuda \
  --server -m /models/mistral-7b-instruct-v0.1.Q8_0.gguf --n-gpu-layers 43 --host 0.0.0.0 --port 8080
```

#### Math LLM

```shell
docker run --gpus all --rm -p 8088:8080 -v $(realpath models):/models ghcr.io/krasserm/llama.cpp:full-cuda \
  --server -m /models/codellama-7b-instruct.Q4_K_M.gguf --n-gpu-layers 0 --host 0.0.0.0 --port 8080
```

Depending on available GPU memory you may want to decrease the `--n-gpu-layers` argument.

### Create environment

```shell
conda env create -f environment.yml
conda activate grammar-based-agents
```

### Run notebook server

```shell
jupyter notebook
```
