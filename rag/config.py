import datasets, os
import torch
from tqdm import tqdm
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline
from typing import Tuple, List

DATASET = "rajpurkar/squad"
LLM = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "thenlper/gte-base"
FAISS_PATH = "faiss_store"

def load_dataset(split: str, num_tokens: int) -> datasets.Dataset:
    ds = datasets.load_dataset(DATASET, split=split)
    tokenizer = AutoTokenizer.from_pretrained(LLM)
    ds = ds.filter(lambda sample: len(tokenizer.encode(sample['context']) < num_tokens))
    return ds

def create_vector_db(dataset: datasets.Dataset=None) -> Tuple[FAISS, HuggingFaceEmbeddings]:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        show_progress=True,
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True} 
    )
    if os.path.isdir(FAISS_PATH):
        return FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True), embedding_model

    if dataset is None:
        raise ValueError("Could not find local vector db and the dataset was not provided!")
    
    docs = [
        LangchainDocument(
            page_content=doc['context'],
            metadata={'title': doc['title']}
        ) for doc in tqdm(dataset)
    ]
    return FAISS.from_documents(
       docs, embedding_model, distance_strategy=DistanceStrategy.COSINE 
    ), embedding_model

def get_llm() -> Tuple[AutoTokenizer, Pipeline]:
    tokenizer = AutoTokenizer.from_pretrained(LLM)
    model = AutoModelForCausalLM.from_pretrained(
        LLM,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    llm_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )
    return tokenizer, llm_pipeline
    
def apply_prompt(tokenizer: AutoTokenizer):
    template = [
        {
            "role": "user",
            "content": """Context: {context}\nHere is the question you need to answer. Question: {question}\nInstruction: Answer with the shortest, most precise answer possible. Example: When did the first world war started?\nAnswer: 1914"""
        }
    ]
    return tokenizer.apply_chat_template(
        template, tokenize=False, add_generation_prompt=True
    )

def select_documents(documents: List[LangchainDocument], tokenizer: AutoTokenizer) -> int:
    MAX_TOKENS = 512 - 52
    current, count = 0, 0
    for doc in documents:
        current += int(tokenizer(doc.page_content, return_tensors="pt")['input_ids'].shape[-1])
        if current < MAX_TOKENS:
            count += 1
        else:
            return count
    return count

def concat_context_documents(documents: list) -> str:
    return "".join([f"Document {i}:::\n{doc.page_content}\n" for i, doc in enumerate(documents, 1)])

def prepare_docs(documents: List[LangchainDocument], tokenizer: AutoTokenizer) -> str:
    num_docs = select_documents(documents, tokenizer)
    return concat_context_documents(documents[: num_docs])


def create_prompt(question: str, vector_db: FAISS, template: str, tokenizer: AutoTokenizer) -> Tuple[str, List[LangchainDocument]]:
    print("Looking for similar topics...")
    simmilar_topics = vector_db.similarity_search(question, k=5)
    context_documents = prepare_docs(simmilar_topics, tokenizer)
    return template.format(question=question, context=context_documents), context_documents

def run(pipeline: Pipeline, question: str, vector_db: FAISS, tokenizer: AutoTokenizer, template: str):
    prompt, relevant_docs = create_prompt(question, vector_db, template, tokenizer)
    print("Generating response...")
    llm_response = pipeline(prompt)[0]['generated_text']
    return llm_response, relevant_docs
   


if __name__ == "__main__":
    db, embedding_model = create_vector_db()
    tokenizer, llm_pipeline = get_llm()
    prompt_template = apply_prompt(tokenizer)
    while True:
        question = input("Provide question:")
        print(run(llm_pipeline, question, db, tokenizer, prompt_template)) 
        print(llm_pipeline(question)[0]['generated_text'])