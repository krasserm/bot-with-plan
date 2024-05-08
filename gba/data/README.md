## Wikipedia search tool dataset and index creation

The following script uses the `krasserm/wikipedia-2023-11-en-embed-mxbai-int8-binary` Wikipedia dataset containing
the `binary` and `int8` embeddings of the Wikipedia created using the `mixedbread-ai/mxbai-embed-large-v1` embedding model.

The script downloads the dataset and creates the index files required for
the [SearchWikipediaTool](../tools/search/search_wikipedia.py):

* `faiss-ubinary.index`: [Faiss](https://github.com/facebookresearch/faiss) index file containing the `binary`
  embeddings
* `usearch-int8.index`: [usearch](https://github.com/unum-cloud/usearch) index file containing the `int8` embeddings
* `document-url-mappings.sqlite`: [SQLite](https://www.sqlite.org/) database file containing mappings from document URLs
  to text node indices
* `wikipedia-en-text`: Wikipedia text-only dataset

```shell
python gba/data/wiki_dataset.py \
  --output_dir=output/wikipedia_search_tool
```
