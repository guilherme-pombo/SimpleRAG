# SimpleRAG
This repo aims to be a very *simple* and *didactical* implementation of Retrieval Augmented Generation (RAG) using two simple-to-use and well maintained libraries: HuggingFace's _transformers_ and Meta's _faiss_ (Facebook AI similarity search).

It doesn't require any OpenAI keys and therefore you can run it for free on your local machine and without having to share your data. It can run on CPU, but for speed purposes it might be best if you have a GPU.

RAG libraries like LangChain or LlamaIndex provide a bunch of useful abstractions but they are constantly changing to the point where their own Tutorials don't work because the library has already changed. Also, they often rely on using OpenAI's API which is not ideal.

This implementation is very simple and inneficient but aims to be more of a "teaching" tool than anything else. Ideally the code should be used as a starting point to something more complex.

## Running the code

First install the dependencies in requirements.txt with:

```
pip install -r requirements.txt
```

then to run the code with default parameters simply do:

```
python simple_rag.py
```

This code uses the documents provided in the *rag_documents* folder about 3 fictional characters to answer the question:

```
Who is Alice Smith? What artifical drug could have been the cause of her disappearance?
```

If you want to change what embedding or generative models are used simply use the command line arguments. For example if you wanted to use Mistral instead of Camel you could simply do:

```
python simple_rag.py --embedding_model mistralai/Mistral-7B-v0.1
```

Or if you wanted to change the embedding model and use 3 documents instead of only 2 for context:

```
python simple_rag.py --embedding_model mistralai/Mistral-7B-v0.1 --number_of_docs 3
```

Have a look at the code for a better grasp of what to do.

## What the code does

1. Read in all the documents from a particular folder (default *rag_documents*) and split them into sentences
2. Embed each of the documents into a single vector using a HuggingFace embedding model (default "sentence-transformers/all-MiniLM-L6-v2") and mean pooling. Mean pooling is a choice made for simplicity, but ideally you would want to use whichever method is most appropriate for the nature of your documents
3. Create a vector index using FAISS, so that for our question, we can look for which documents might provide the most relevant context
4. Use the retrieved documents to generate a prompt that we'll be fed into a generative model (default: "Writer/camel-5b-hf")
5. Feed the prompt into a generative model to answer the question
