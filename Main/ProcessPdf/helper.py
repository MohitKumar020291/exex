import os
import warnings
import re
from typing import Dict
from pathlib import Path

from ollama import chat, ChatResponse
from langchain_core.documents import Document
from transformers import AutoTokenizer, tokenization_utils_fast
from huggingface_hub import login

from Main.models import DocumentSplitterLangChain


def read_prompts(file_name):
    prompt = None
    file_name, extension = os.path.splitext(file_name)
    if len(extension) == 0:
        file_name = file_name + ".txt"
        warnings.warn(f"No file_name extension is provided, assuming txt, new file_name = {file_name}")

    full_path = os.path.join("Main", "ProcessPdf", "Prompts", file_name)
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as fh:
                prompt = fh.read()
        except Exception as e:
            raise Exception(e)

    return prompt

def classify_chunk_relationship(text):
    system_prompt = read_prompts(file_name="classify_chunk_relationship_2.txt")
    response = chat(model='llama3:8b-instruct-q4_0',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ])
    return response["message"]["content"]

def clean_question_chunk(
        text, 
        system_prompt = None,
        model_name = None
) -> str:
    system_prompt = system_prompt or read_prompts(file_name="clean_question_chunk")
    model_name = model_name or 'llama3:8b-instruct-q4_0'
    response: ChatResponse = chat(model = model_name,
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": text}])
    return response.get("message", {}).get("content", "").strip()

def merge_docs(documents: DocumentSplitterLangChain) -> DocumentSplitterLangChain:
    merged_docs = {}
    for file_name, docs in documents.documents.items():
        merged_text = "\n".join(doc.page_content for doc in docs)
        merged_docs[file_name] = [Document(page_content=merged_text)]
    return DocumentSplitterLangChain(documents=merged_docs)

def get_hf_auth_token() -> str:
    token_file = os.path.join(Path.home(), '.cache', "huggingface", "token")
    if os.path.exists(token_file):
        with open(token_file, "r") as file:
            token = file.read().strip()
            return token
    else:
        raise ValueError(f"token_file: {token_file} does not exists!")

def get_hf_tokenizer(
    model_name: str
) -> tokenization_utils_fast.PreTrainedTokenizerFast:
    token = get_hf_auth_token()
    try:
        login(token=token)
    except Exception as e:
        raise e
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer