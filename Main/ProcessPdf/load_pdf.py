import os
import subprocess
import hashlib, base64
import re
from functools import partial
from typing import Dict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
import torch
import numpy as np
from ollama import chat, ChatResponse

from Main.models import DocumentSplitterLangChain
from Main.Embeddings.embedding import CustomEmbeddings
from Main.ProcessPdf.helper import (
                                read_prompts, 
                                classify_chunk_relationship, 
                                clean_question_chunk,
                                merge_docs,
                                get_hf_tokenizer
                                )


async def get_documents_langchain(input_dir: str) -> DocumentSplitterLangChain:
    os.path.exists(input_dir)
    input_dir_pdfs = subprocess.run(
                        ['find', input_dir, '-type', 'f', '-name', '*.pdf'], 
                        capture_output=True).stdout.strip().decode('utf-8').split('\n')
    documents = dict() # file_name.pdf: documents
    for file_pdf_path in input_dir_pdfs:
        documents_idx = []
        loader = PyPDFLoader(file_pdf_path)
        # These is pdf page
        async for page in loader.alazy_load():
            documents_idx.append(page)
        documents[file_pdf_path] = documents_idx

    return DocumentSplitterLangChain(documents=documents)


class EmbeddingSplitter:
    def __init__(self, 
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                threshold=0.70,
                normalize: bool = True
                ):
        """
        threshold: lower = more aggressive splitting (smaller chunks)
        """
        self.embed_model = CustomEmbeddings(model_name=model_name, normalize=normalize)
        self.threshold = threshold

    @staticmethod
    def _unique_marker(text: str) -> str:
        h = hashlib.sha256(text.encode()).digest()
        return "=========" + base64.urlsafe_b64encode(h)[:16].decode() + "========="

    def _embed(self, sentences):
        embs = self.embed_model.embed_documents(sentences)
        return embs

    def embedding_based_splitting(self, 
                                text: str, 
                                metadata_document: dict, 
                                chunks_covered_pdf: int,
                                lookBackPower: int = 5,
                                ) -> str:
        """Splits text into semantically distinct chunks and inserts markers."""
        if not isinstance(chunks_covered_pdf, int):
            ...
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return text
        embs = self._embed(lines)
        embs = torch.from_numpy(embs)

        relns = {
            i: [(i, i-j) for j in range(1, lookBackPower+1) if i-j >= 0]
            for i in range(len(lines))
        }

        # Does each of the chunk have any relation?
        relns_power = []
        calculated = dict()
        for _, reln_pair in relns.items():
            for i, j in reln_pair:
                sim_idx_idxx = calculated.get((i, j), None) or\
                      torch.nn.functional.cosine_similarity(embs[i].reshape(1,-1), embs[j].reshape(1,-1)).item() # <-- float conversin
                calculated[(i, j)] = sim_idx_idxx
                relns_power.append((i, j, sim_idx_idxx))
        
        relns_power.sort(key = lambda x: -x[-1])
        reln_lookup = {
            (i, j): float(sim)
            for i, j, sim in relns_power
        }
        strong_reln_threshold = max(self.threshold, 0.15)
        strong_reln_threshold = 0.40

        # Finding semantic gaps
        sims = torch.nn.functional.cosine_similarity(embs[:-1], embs[1:])

        boundaries = [0]
        for i, s in enumerate(sims):
            # Check if current line i+1 has any strong relation with its lookback context
            has_strong_relation = False
            candidates = relns[i + 1] if (i + 1) in relns else []
            for c in candidates:
                reln_sim = reln_lookup.get(c, 0.0)
                # print(lines[c[0]], lines[c[1]], reln_sim, strong_reln_threshold, reln_sim >= strong_reln_threshold)
                if reln_sim >= strong_reln_threshold:
                    has_strong_relation = True
                    break

            # Boundary = weak local sim AND no strong relation
            if s < self.threshold and not has_strong_relation:
                boundaries.append(i + 1)

        boundaries.append(len(lines))

        # Building chunks
        result_chunks = self._resolve_boundaries(lines, boundaries, metadata_document, chunks_covered_pdf)
        return result_chunks, len(boundaries) - 1

    def regex_based_splitting(self, text: str, metadata_document: dict, subject: str, pattern: str = r"Question"):
        if pattern == None:
            # Use chat models to generate the regex
            ...
        lines = [l.strip() for l in text.splitlines('\n') if l.strip()]
        boundaries = [0]
        for idx, line in enumerate(lines):
            matched = re.match(pattern, line)
            print(matched, pattern, line)
            if re.match(pattern, line):
                boundaries.append(idx)
        boundaries.append(len(lines))
        result_chunks = self._resolve_boundaries(lines, boundaries, metadata_document)
        return result_chunks

    # The most I will feed is three pages
    def llm_based_splitter(self, text: str, metadata_document: dict, *args, **kwargs):
        """
            The only problem I see here is what if the text is too long.
        """
        system_prompt = read_prompts(file_name="llm_based_splitter.txt")
        llm_model_name: str = 'llama3:8b-instruct-q2_K'
        response: ChatResponse = chat(
            model=llm_model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ])

        return response["message"]["content"], None
    
    def token_based_splitter(
        self,
        chunk_overlap: int,
        model_name: str,
        tokens_per_chunk: int,
        documents: DocumentSplitterLangChain,
        subject: str = "AI",
        **kwargs
    ) -> DocumentSplitterLangChain:
        # I don't need metadata = {"source": "file_name"}
        documents = documents.documents.copy() # <-- pydant problem - should be a better way of typing
        for file_name, docs in documents.items():
            docs[0].page_content = re.split(pattern=subject, string=docs[0].page_content)[-1]
            documents[file_name] = docs
        # merge docs - so that can utilize the full tokenization limit
        documents = merge_docs(DocumentSplitterLangChain(documents=documents)).documents

        # splitting on the basis of the token length
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap ,
            model_name=model_name,
            tokens_per_chunk=tokens_per_chunk
        )

        for file_name, docs in documents.items():
            try:
                splitted_docs = splitter.transform_documents(docs)
            except Exception as e:
                raise Exception(e)
            documents[file_name] = splitted_docs

        return DocumentSplitterLangChain(documents=documents)

    def _resolve_boundaries(self, lines, boundaries, metadata_document, chunks_covered_pdf):
        result_chunks = []
        unique_marker  = self._unique_marker(lines[0])
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk = "\n".join(lines[start:end])
            result_chunks.append(chunk + "\n" + unique_marker)

        for idx, result_chunk in enumerate(result_chunks):
            metadata_document["chunk_number"] = chunks_covered_pdf + idx
            result_chunks[idx] = Document(
                page_content=result_chunk,
                metadata=metadata_document # <-- this is for each page in a pdf
            )

        return result_chunks

model_2_context_length = {
            "meta-llama/Meta-Llama-3-8B-Instruct": 8192
        }

class Clean:
    def __init__(self):
        ...

    @staticmethod
    def clean_chunks(
        splitted_documents: DocumentSplitterLangChain
    ) -> DocumentSplitterLangChain:
        r"""
            Definition:
                splitted docs of a file are combined if required (defined below).
                required: if we need some prev doc for a doc we ask for the prev doc, sometimes
                we may require next doc, and something like.
                How do we decide we need any other chunk/doc? Ahh... we passes current text to the LLM
                and there is system prompt in "Main/prompts/classify_chunk_relationship_2.txt" which helps
                the LLM to decide whether we need any other chunk or we have the full question.
                then we are using a model (right now llama3:8b-instruct-q4_0).
                When we have the full question, we might want to remove the metadata about the question,
                this is done by the EmbeddingSpliiter.clean_question_chunk, see above.
            Args:
                splitted_documents: These are splitted documents (right now on the basis of the tokens).
            Usefulness:
                No (read limitations)
            Limitations:
                Models sucks (dilution, context length), needs more configuration and over
                engineered when compared to the normal cleaning of the text (formed through some different method).
        """
        splitted_documents = splitted_documents.documents
        cleaned_documents = dict()
        for file_name, docs in splitted_documents.items():
            cleaned_documents[file_name] = []
            num_docs = len(docs)
            upper_range = 0
            for idx, doc in enumerate(docs):
                if idx < upper_range:
                    continue
                print(f"\ndoc {idx}\n", remove_hash(doc.page_content))

                response = "None"
                prev_question = cleaned_documents[file_name][-1] if len(cleaned_documents[file_name]) > 0 else "No previous question yet"
                text = f"\ntext:\n{remove_hash(doc.page_content)}"
                cannot_ask_previous = True if idx == 0 else False
                current_idx = idx

                while response.strip() != "done":
                    if cannot_ask_previous:
                        send_text: str = f"cannot_ask_previous = True\nprevious question:\n{prev_question}\n" + text
                    else:
                        send_text: str = f"previous question:\n{prev_question}\n" + text
                    # send_text = send_text.replace("\n", " ")
                    response = classify_chunk_relationship(text=send_text)
                    if response == "prev":
                        current_idx = max(current_idx-1, 0)
                        print("prev", f"current_idx: {current_idx}", f"doc idx: {idx}")
                        if current_idx == 0:
                            cannot_ask_previous = True
                        else:
                            text =  f"{remove_hash(docs[upper_range].page_content)}\n" + text
                    elif response == "next":
                        upper_range = min(upper_range+1, num_docs)
                        text = text + f"\n{remove_hash(docs[upper_range].page_content)}"
                    elif response == "try_next_chunk_for_options":
                        text = text + f"\n{remove_hash(docs[idx+1].page_content)}"
                    elif response == "done":
                        print("done", f"current_idx: {current_idx}", f"doc idx: {idx}")
                    else:
                        print(response)

                upper_range += 1

                cleaned_question = clean_question_chunk(text=text)
                if not("no question found" in cleaned_question):
                    cleaned_documents[file_name].append(cleaned_question)
        
        return DocumentSplitterLangChain(documents=cleaned_question)
    
    @staticmethod
    def clean_docs(
        documents: DocumentSplitterLangChain
    ) -> DocumentSplitterLangChain:
        """
            This function is calling model everytime for a doc.
            Which is slow according to me.
        """
        tokenizer_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        llm_model_name = "llama3:8b-instruct-q4_0"
        context_length = model_2_context_length.get(tokenizer_model_name)
        leave_apart = 500 #changes according to models
        tokenizer = get_hf_tokenizer(model_name=tokenizer_model_name)
        documents = documents.documents
        system_prompt = read_prompts(file_name="clean_question_chunk")
        system_prompt_tokens = tokenizer(system_prompt)["input_ids"]
        max_model_limit = context_length - len(system_prompt_tokens) - leave_apart
        PromptDominanceThreshold = 10
        prompt_dilution_limit = len(system_prompt_tokens) * PromptDominanceThreshold


        clean_docs = dict()
        for file_name, docs in documents.items():
            clean_docs[file_name] = []
            text = ""
            for idx, doc in enumerate(docs):
                text_input = text + "\n" + doc.page_content if text != "" else doc.page_content
                tokenized_text = tokenizer(text_input) #testing
                if len(tokenized_text["input_ids"]) <= max_model_limit \
                    and idx != len(docs) - 1 \
                    and len(tokenized_text["input_ids"]) <= prompt_dilution_limit:
                    text = text + "\n" + doc.page_content #updating
                else:
                    response: str = clean_question_chunk(
                                        text=text, 
                                        system_prompt=system_prompt,
                                        model_name=llm_model_name
                                    )
                    doc.page_content = response
                    clean_docs[file_name].append(doc)
                    text = ""

        return DocumentSplitterLangChain(documents=clean_docs)


def remove_hash(text):
    return "\n".join(text.split("\n")[:-1])

def split(
    documents: DocumentSplitterLangChain,
    subject: str,
    embedding_splitter: EmbeddingSplitter,
    split_type: str,
    lookBackPower: str,
    pattern: str,
    **kwargs
) -> DocumentSplitterLangChain:
    documents = documents.documents
    splitted_documents = dict()
    if split_type == "embedding":
        split_method = partial(embedding_splitter.embedding_based_splitting, lookBackPower=lookBackPower)
    elif split_type == "regex":
        split_method = partial(embedding_splitter.regex_based_splitting, subject=subject, pattern=pattern)
    elif split_type == "llm":
        split_method = partial(embedding_splitter.llm_based_splitter)
    elif split_type == "token":
        chunk_overlap = kwargs.get("chunk_overlap", None)
        if chunk_overlap == None:
            raise ValueError("chunk_overlap could not be None")
        model_name = kwargs.get("model_name", None)
        if model_name == None:
            raise ValueError("model_name could not be None")
        tokens_per_chunk = kwargs.get("tokens_per_chunk", None)
        if tokens_per_chunk == None:
            raise ValueError("tokens_per_chunk could not be None")
        split_method = partial(
                            embedding_splitter.token_based_splitter,
                            chunk_overlap=chunk_overlap,
                            model_name=model_name,
                            tokens_per_chunk=tokens_per_chunk
                        )
        splitted_documents: DocumentSplitterLangChain = split_method(documents=DocumentSplitterLangChain(documents=documents))
        return splitted_documents

    for file_name, list_docs in documents.items():
        splitted_documents[file_name] = list()
        chunks_covered_pdf = 0
        for idx, document in enumerate(list_docs):
            if idx == 0:
                document.page_content = re.split(subject, document.page_content)[-1]
            splitted_document, chunks_covered_page = split_method(
                                    text=document.page_content, 
                                    metadata_document=document.metadata,
                                    chunks_covered_pdf=chunks_covered_pdf
                                )
            chunks_covered_pdf += chunks_covered_page
            splitted_documents[file_name].extend(splitted_document)

        assert len(splitted_documents[file_name]) == chunks_covered_pdf, "Total number of chunks are not equal to chunks covered pdf"
    
    return DocumentSplitterLangChain(documents=splitted_document)

# this is a run kinda function
# better to be part of the pipeline
def document_splitter(
    documents: DocumentSplitterLangChain,
    sentence_embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    split_type: str = 'embedding',
    lookBackPower: int = 3,
    subject: str = "AI",
    pattern: str = r"Question",
    **kwargs
):
    embedding_splitter = EmbeddingSplitter(sentence_embed_model_name)

    token_based_splitter_args = {
        "chunk_overlap": 50,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "tokens_per_chunk": 256
    }

    splitted_documents: DocumentSplitterLangChain = split(
        documents=documents,
        subject=subject,
        split_type=split_type,
        embedding_splitter=embedding_splitter,
        lookBackPower=lookBackPower,
        pattern=pattern,
        **token_based_splitter_args
    )

    cleaned_questions = Clean.clean_docs(
            documents=splitted_documents
        )
    
    return cleaned_questions