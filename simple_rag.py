import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm


def process_file(file_path):
    '''
    Load in a text file and split it into sentences using Spacy
    Spacy is not a requirement, use whatever chunking you want
    but spacy is decent
    '''
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
        # Split document into sentences
        # You'll probably want to use something more clever than splitting by line
        # Spacy provides a good sentence splitter
        sentences = text.split('\n')
        return text, sentences


def generate_rag_prompt(data_point):
    return f"""### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
"""


class DocumentEmbedder:

    def __init__(self,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 max_length=128,
                 max_number_of_sentences=20
                 ):
        # Load AutoModel from huggingface model repository
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # This parameter dictates the maximum number of tokens per sentence
        self.max_length = max_length
        # This dictates the maximum number of sentences to be considered
        self.max_number_of_sentences = max_number_of_sentences

    def get_document_embeddings(self, sentences):
        # Keep only the first K sentences for GPU purposes
        sentences = sentences[:self.max_number_of_sentences]
        # Tokenize the sentences
        encoded_input = self.tokenizer(sentences,
                                       padding=True,
                                       truncation=True,
                                       max_length=128,
                                       return_tensors="pt")
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # The document's embedding is the average of all sentences
        # If there's only one sentence, then it's just it's embedding
        return torch.mean(model_output.pooler_output, dim=0, keepdim=True)


class GenerativeModel:

    def __init__(self,
                 model_path="Writer/camel-5b-hf",
                 max_input_length=200,
                 max_generated_length=200
                 ):
        # Load 4-bit quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = max_input_length
        self.max_generated_length = max_generated_length
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def answer_prompt(self, prompt):
        # Tokenize the sentences
        encoded_input = self.tokenizer([prompt],
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_input_length,
                                       return_tensors="pt")

        outputs = self.model.generate(input_ids=encoded_input['input_ids'].to(self.device),
                                      attention_mask=encoded_input['attention_mask'].to(self.device),
                                      max_new_tokens=self.max_generated_length,
                                      do_sample=False)

        decoder_text = self.tokenizer.batch_decode(outputs,
                                                   skip_special_tokens=True)
        return decoder_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_directory",
                        help="The directory that has the documents",
                        default='rag_documents'
                        )
    parser.add_argument("--embedding_model",
                        help="The HuggingFace path to the embedding model to use",
                        default="sentence-transformers/all-MiniLM-L6-v2"
                        )
    parser.add_argument("--generative_model",
                        help="The HuggingFace path to the generative model to use",
                        default="Writer/camel-5b-hf"
                        )
    parser.add_argument("--number_of_docs",
                        help="The number of relevant documents to use for context",
                        default=2
                        )
    args = parser.parse_args()

    '''
    Process all the files in the directory by chunking them into sentences
    Keep track of the original documents filepath
    '''
    print('Splitting documents into sentences...')
    documents = {}
    for idx, file in enumerate(tqdm(os.listdir(args.documents_directory)[:10])):
        current_filepath = os.path.join(args.documents_directory, file)
        text, sentences = process_file(current_filepath)
        documents[idx] = {'file_path': file,
                          'sentences': sentences,
                          'document_text': text
                          }

    '''
    Now for all sentences get embeddings
    '''
    print('Getting document embeddings...')
    document_embedder = DocumentEmbedder(model_name=args.embedding_model,
                                         max_length=128,
                                         max_number_of_sentences=20)
    embeddings = []
    for idx in tqdm(documents):
        # Embed the document
        embeddings.append(document_embedder.get_document_embeddings(documents[idx]['sentences']))
    # Concatenate all embeddings
    embeddings = torch.concat(embeddings, dim=0).data.cpu().numpy()
    embedding_dimensions = embeddings.shape[1]

    '''
    Use FAISS to build an index we can use to search through the embeddings
    Ideally you'll want to cache this index so you don't have to build it every time
    '''
    faiss_index = faiss.IndexFlatIP(int(embedding_dimensions))
    faiss_index.add(embeddings)

    question = "Who is Alice Smith? What artifical drug could have been the cause of her disappearance?"
    query_embedding = document_embedder.get_document_embeddings([question])
    distances, indices = faiss_index.search(query_embedding.data.cpu().numpy(),
                                            k=int(args.number_of_docs))

    '''
    We use the K-closest documents to provide context to the generative model's answer
    '''
    context = ''
    for idx in indices[0]:
        context += documents[idx]['document_text']

    rag_prompt = generate_rag_prompt({'instruction': question,
                                      'input': context})

    '''
    Use the generative model to give an answer to the question
    Use the retrieved documents for context
    '''
    print('Generating answer...')
    generative_model = GenerativeModel(model_path=args.generative_model,
                                       max_input_length=200,
                                       max_generated_length=200)
    answer = generative_model.answer_prompt(rag_prompt)[0].split('### Response:')[1]
    print(answer)
