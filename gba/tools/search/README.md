# Search tools

This package contains tools for internet and Wikipedia searches.

## Search internet tool

The [SearchInternetTool](search_internet.py) is a RAG-based search tool that performs internet searches across multiple search engines and synthesizes concise responses using a large language model (LLM).

The tool does not require API keys as it utilizes a local [SearXNG](https://github.com/searxng/searxng) meta search instance to query various search engines.
Results from the search engines are retrieved from the internet, filtered and re-ranked by relevance and finally passed to the LLM to generate a concise response.

For a more detailed explanation of the implementation details refer to the section [search internet tool implementation](DETAILS.md#search-internet-tool-implementation).

To use the [SearchInternetTool](search_internet.py) you need to set up a local [SearXNG](https://github.com/searxng/searxng) instance and provide the endpoint URL to the tool.

### Setup

Setup a local SearXNG instance using the official docker container (see also [SearXNG docs](https://docs.searxng.org/admin/installation-docker.html#searxng-searxng)).

```shell
docker run \
  --name searxng \
  -d -p 8080:8080 \
  -v "${PWD}/.searxng:/etc/searxng" \
  -e "BASE_URL=http://localhost:8080" \
  -e "INSTANCE_NAME=my-instance" \
  searxng/searxng:2024.5.24-75e4b6512
```

The `.searxng/settings.yaml` file used in this project has been modified to additionally support `json` mode:

```yaml
   search:
     formats:
     - html
     - json
   ```

See [getting started](../../../README.md#getting-started) for instructions how to serve the Llama 3 model used by these tools.

### Usage

```python
from gba.tools.search import create_search_internet_tool
from gba.utils import Scratchpad

search_internet = create_search_internet_tool(
    llama3_endpoint="http://localhost:8084/completion",
    searxng_endpoint="http://localhost:8080",
)

response = search_internet.run(
    task="When was the video game 'The Last of Us' released",
    request="",
    scratchpad=Scratchpad(),
)
```

### Parameters

* `top_k_documents`: number of top-k webpages to select from the search results (default: `3`)
* `top_k_nodes_per_document`: number of top-k relevant text nodes to select from each webpage for generating the response (default: `5`)
* `top_k_snippets`: number of top-k webpage snippets to include (default: `top_k_documents`)

The complete list of parameters can be found in the [SearchInternetTool](search_internet.py) class.

## Search Wikipedia tool

The [SearchWikipediaTool](search_wikipedia.py) is a RAG-based search tool designed for efficient search in a local Wikipedia dataset.

The tool utilizes multiple locally stored, quantized search indices for memory and runtime-efficient nearest neighbor searches in the dataset.
Given a search query, the tool retrieves the most relevant text nodes from Wikipedia articles and synthesizes responses using a large language model (LLM).

_Note: the dataset has a knowledge cutoff of November 2023._

For a more detailed explanation of the implementation details refer to the section [search wikipedia tool implementation](DETAILS.md#search-wikipedia-tool-implementation).

### Usage

```python
from gba.tools.search import create_search_wikipedia_tool
from gba.utils import Scratchpad

search_wikipedia = create_search_wikipedia_tool(
   llama3_endpoint="http://localhost:8084/completion",
)

response = search_wikipedia.run(
   task="Search Wikipedia for the launch date of the first iPhone.",
   request="",
   scratchpad=Scratchpad(),
)
```

### Parameters

* `top_k_nodes`: number of top-k text nodes to select from the initial search results (default: `10`)
* `top_k_related_documents`: number of top-k related documents to select from the initial search results (default: `1`)
* `top_k_related_nodes`: number of top-k text nodes to select from the related documents (default: `3`)

The complete list of parameters can be found in the [SearchWikipediaTool](search_wikipedia.py) class.
