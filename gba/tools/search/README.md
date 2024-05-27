# Search tools

This package contains search tools for internet and Wikipedia searches.

## Search internet tool

The [SearchInternetTool](search_internet.py) is a RAG-based search tool that performs internet searches across multiple search engines
and synthesizes concise responses using a large language model (LLM).

The tool does not require API keys as it utilizes a local [SearXNG](https://github.com/searxng/searxng) meta search instance to query various search engines.
Results from the search engines are retrieved from the internet, filtered and ranked by relevance and finally passed to the LLM to generate a concise response.

For a more detailed explanation of the implementation details refer to the section [search internet tool implementation](#search-internet-tool-implementation).

To use the [SearchInternetTool](search_internet.py) you need to set up a local [SearXNG](https://github.com/searxng/searxng) instance and provide the endpoint URL to the tool.

### Setup

1. Setup a local SearXNG instance using the official docker container by following the instructions outlined the [SearXNG docs](https://docs.searxng.org/admin/installation-docker.html#searxng-searxng).
2. Start the docker container.
3. Locate the `settings.yml` file created by the container in the volume mount directory and enable `json` mode by modifying the `search.formats` section as follows:
   ```yaml
   search:
     formats:
     - html
     - json
   ```
4. Restart the docker container.

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

The [SearchWikipediaTool](../tools/search/search_wikipedia.py) is a RAG-based search tool designed for efficient search in a local Wikipedia dataset.

The tool utilizes multiple locally stored, quantized search indices for memory and runtime-efficient nearest neighbor searches in the dataset.
Given a search query, the tool retrieves the most relevant text nodes from Wikipedia articles and synthesizes responses using a large language model (LLM).

_Note: the dataset has a knowledge cutoff of November 2023._

For a more detailed explanation of the implementation details refer to the section [search wikipedia tool implementation](#search-wikipedia-tool-implementation).

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

## Implementation details

### Search internet tool implementation

The [SearchInternetTool](search_internet.py) is a RAG-based search tool that performs internet searches across multiple search engines and generates responses using a large language model (LLM).

The tool leverages the [SearXNG](https://github.com/searxng/searxng) meta search engine to query multiple search engines simultaneously.
Each search result includes the URL, webpage title, and webpage snippet.

_Note_: In the context of RAG search, a webpage is considered a _document_ consisting of multiple text _nodes_.

The tool selects the top-k documents (webpages) (`top_k_documents`, default `3`) from the initial search results and retrieves their URL, title, snippet, and content.

The document (webpage) content is fetched using the [Unstructured](https://unstructured-io.github.io/unstructured/core/partition.html) library,
which extracts text nodes from the webpage and filters out boilerplate content such as headers, footers, and ads.

For each document (webpage), the tool then extracts the first top-k text nodes (`num_nodes_rerank_per_document`, default `100`),
reranks the text nodes based on their relevance to the search query, and selects the top-k relevant text nodes to use in the response (`top_k_nodes_per_document`, default `5`).
The selected text nodes are concatenated with the webpage snippet and passed to the LLM to extract the most relevant information as a single summary per document.

In addition to the document (webpage) content, the tool allows for the inclusion of up to top-k webpage snippets (parameter: `top_k_snippets_per_document`, default `None`) without the corresponding webpage content.
These snippets are not summarized as they are considered to be already concise.

In the last step the tool passes the summarized documents (and optional snippets) to an LLM to generate a concise response.

### Search Wikipedia tool implementation

The [SearchWikipediaTool](search_wikipedia.py) searches a static Wikipedia dataset with a knowledge cutoff of November 2023, identifying relevant articles and synthesizing responses using a large language model (LLM).

The dataset consists of documents (Wikipedia pages) and their corresponding text nodes (paragraphs), derived from the English subset of the [Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary) dataset, containing 41M text nodes.

To utilize custom embeddings for similarity search, the dataset is processed using the [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) embedding
model, creating embeddings for each text node. For fast and efficient nearest neighbor search, we follow the approach outlined in the blog post
[Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization).

In line with this approach, the original `float32` embeddings from the model are quantized into `binary` and `int8` embeddings.
The `binary` embeddings create an in-memory binary [Faiss](https://github.com/facebookresearch/faiss) index for the initial nearest neighbor search, returning `top_k` * `rescore_multiplier` results.
This index is small (~4GB in memory) and fast but may not return the most accurate results.

To improve the accuracy of the search, the `int8` embeddings are used to create a memory-mapped [usearch](https://github.com/unum-cloud/usearch) index for a second stage,
where the results of stage 1 are re-scored using the query embedding and the respective `int8` embeddings, yielding the final `top_k` results.

Search settings are adjustable using the `similarity_search_top_k_nodes` (default: `100`) and `similarity_search_rescore_multiplier` (default: `4`) parameters.

As we want to use our own embeddings for similarity search, the dataset is preprocessed using the [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) embedding model, creating embeddings for each text node.
To perform fast nearest neighbor search for the 41M text nodes, with minimal resource requirements and minimal loss of accuracy, we follow the approach outlined in
the blog post [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization).

The original `float32` embeddings retrieved from the model are quantized into respective `binary` and an `int8` embeddings.
The `binary` embeddings are used to create a binary index, which is kept in memory and is used to perform the initial nearest neighbor search in a first stage, returning `top_k` * `rescore_multiplier` results.
The `int8` embeddings are used to create a memory-mapped [usearch](https://github.com/unum-cloud/usearch) index, which is facilitated in a second stage. Here, the `int8` embeddings of all
stage 1 results are loaded from the index and re-scored using the query embedding. The final `top_k` results are returned.

The settings for this search stage can be adjusted using the `similarity_search_top_k_nodes` (default: `100`) and `similarity_search_rescore_multiplier` (default: `4`) parameters.

After the initial search, the tool reranks results using a cross-encoder model, selecting the top-k most relevant text nodes (`top_k_nodes`, default: `10`).
These nodes are grouped by document (Wiki page).

To provide additional relevant information for the query, the tool loads all text nodes from the top-k documents (`top_k_related_documents`, default: `1`),
reranks these nodes by relevance, and selects the top-k related nodes (`top_k_related_nodes`, default: `3`) to include into the response generation.
The top-k documents usually contain the relevant information from the search query and the overall performance can be improved by including additional information from these documents.

All resulting text nodes are then grouped and concatenated by document and each resulting document is summarized by the LLM.
Finally, the summarized documents are passed to the LLM to generate a concise response.

### Search Wikipedia dataset and index creation

_Note: the following steps are optional as the [SearchWikipediaTool](../tools/search/search_wikipedia.py) uses the pre-built dataset and index files from Hugging Face by default._

#### Dataset creation

To create the dataset containing the `binary` and `int8` embeddings of the Wikipedia articles the following script can be used.

_Note: This step is optional. You can use the pre-built dataset from [krasserm/wikipedia-2023-11-en-embed-mxbai-int8-binary](https://huggingface.co/datasets/krasserm/wikipedia-2023-11-en-embed-mxbai-int8-binary) instead._

This script downloads the [Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary) dataset
and creates a new dataset containing the `binary` and `int8` embeddings of the English Wikipedia articles using the [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) embedding model.
The new dataset contains the following columns:
* `_id`: unique identifier of the Wikipedia text chunk
* `title`: title of the Wikipedia article
* `url`: URL of the Wikipedia article
* `text`: text node of the Wikipedia article
* `emb_ubinary`: `binary` embeddings of the Wikipedia text node
* `emb_int8`: `int8` embeddings of the Wikipedia text node

```shell
python gba/tools/search/create_wikipedia_dataset.py \
  --output_dir=output/wikipedia-2023-11-en
```

#### Index creation

To manually create the index files required for the [SearchWikipediaTool](../tools/search/search_wikipedia.py), the following script
can be used.

_Note: This step is optional as the tool uses the pre-built dataset and index files from [krasserm/wikipedia-2023-11-en-index](https://huggingface.co/datasets/krasserm/wikipedia-2023-11-en-index/tree/main) by default._

The script uses the [krasserm/wikipedia-2023-11-en-embed-mxbai-int8-binary](https://huggingface.co/datasets/krasserm/wikipedia-2023-11-en-embed-mxbai-int8-binary) Wikipedia dataset containing
the `binary` and `int8` embeddings of the Wikipedia articles. These embeddings were created using the [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) embedding model.

The script downloads the dataset and creates the index files required for
the [SearchWikipediaTool](../tools/search/search_wikipedia.py):
* `faiss-ubinary.index`: [Faiss](https://github.com/facebookresearch/faiss) index file containing the `binary`
  embeddings
* `usearch-int8.index`: [usearch](https://github.com/unum-cloud/usearch) index file containing the `int8` embeddings
* `document-url-mappings.sqlite`: [SQLite](https://www.sqlite.org/) database file containing mappings from document URLs
  to text node indices
* `wikipedia-en-text`: Wikipedia text-only dataset

```shell
python gba/tools/search/create_wikipedia_search_index.py \
  --output_dir=output/wikipedia_search_tool
```
