import os
import subprocess
import hashlib, base64
import re
from functools import partial

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import torch
import numpy as np
from ollama import chat
from ollama import ChatResponse

from Main.models import DocumentSplitterLangChain
from Main.Embeddings.embedding import CustomEmbeddings


async def get_documents_langchain(input_dir: str):
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

    return documents


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
        system_prompt = """
                You have been provided the raw text metadata, IDs, hashes, and repeated blocks.\n
                I want to divide the page content into chunks of each question\n.
                Your task is to\n
                1. Remove any extra metadata not required from the question like\n:
                "   Question Id, Question Type, Correct Marks, Question Label,\n"
                "   Sub-Section, Max. Selectable Options, Group Comprehension Questions, Question Pattern Type.\n"
                "   Also remove numeric prefixes like '6406532787170.' before text.\n"
                "   Remove duplicates or repeated question blocks.\n\n"
                2." Keep\n:"
                "- The actual question statements and question number, options, and meaningful instructions.\n"
                "- Examples:\n"
                '   - "Based on the above data, answer the given subquestions."\n'
                '   - "Select the valid adjacency representations of the tour."\n'
                '   - Lists like "M, U, S, K, O, V, R, T, P, L, N, Q"\n\n'"
                "- label each chunk with a chunk_number on that page"
                3. Sometimes you may encounter that the question is not complete for a page - just leave that question there but mention INCOMPLETE in the end of that chunk.
                "such as -\n"
                "Question ...\n"
                "chunk_number 2 (if this second question)"
                "INCOMPLETE (if question is incomplete)"
                "### OUTPUT FORMAT\n"
                "Return only clean, human-readable text without markdown or commentary.\n"
            """
        llm_model_name: str = 'llama3:8b-instruct-q2_K'
        response: ChatResponse = chat(
            model=llm_model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ])
        
        print(response["message"]["content"])
        return response["message"]["content"], None

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
    
    def classify_chunk_relationship(self, text):
        system_prompt = """
You are a strict classifier that analyzes a text chunk from an exam question.

Your task is to decide whether this chunk:
1. Belongs with the previous chunk,
2. Belongs with the next chunk,
3. Is a complete question,
4. Ends with options but not all options are visible (partial options),
5. Contains a question following a previously completed question, or
6. Should skip linking to previous if explicitly disallowed.

We process **exactly one question at a time**. 
If the text clearly includes a full question (identified by "Question Number", "Question Label", and a complete set of options), classify it as done.

### Decision Logic
- If the chunk starts mid-sentence, lacks a clear "Question Number" or "Question Label", or looks like a continuation of a previous chunk → return prev.
- If the chunk ends abruptly, or options or the question text appear incomplete → return next.
- If the chunk ends with options but not all options (A/B/C/D or numeric options) are present → return try_next_chunk_for_options.
- If the chunk includes the full question (including all options), or includes multiple question blocks, return done.
- If the chunk contains text of a new question immediately after a previously completed one → classify based on the new question and return done.

### Notes
- Treat each "Question Number" as a signal for a new question boundary.
- If any question appears fully contained before another "Question Number" starts → return **done** for that question.
- Do not overextend context beyond visible question text and options.
- If the chunk includes “cannot_ask_previous = True” → never return prev.

### Possible outputs (exactly one word, no italic and no bold):
- prev
- next
- done
- try_next_chunk_for_options
        """

        system_prompt = """
You are a strict rule-based classifier. Output only one of:
["prev", "next", "done", "try_next_chunk_for_options"]

Use the following ordered decision tree to decide:

{
    "priority_order": [
        "cannot_ask_previous = True",
        "continuation_or_incomplete_question",
        "partial_options",
        "complete_question",
        "new_question_after_completed"
    ],
    "rules": {
        "cannot_ask_previous = True": {
            "If the chunk ends abruptly, or options or the question text appear incomplete": "next",
            "If the chunk ends with options but not all options (A/B/C/D or numeric options) are present": "try_next_chunk_for_options",
            "If the chunk includes the full question (including all options), or includes multiple question blocks": "done",
            "If the chunk contains text of a new question immediately after a previously completed one": "done"
        },
        "continuation_or_incomplete_question": {
            "If the chunk starts mid-sentence, lacks a clear 'Question Number' or 'Question Label', or looks like a continuation of a previous chunk": "prev",
            "If the chunk ends abruptly or options are incomplete": "next"
        },
        "partial_options": {
            "If the chunk ends with options but not all options are present": "try_next_chunk_for_options"
        },
        "complete_question": {
            "If the chunk includes the full question with all options": "done"
        },
        "new_question_after_completed": {
            "If the chunk includes text of a new question after a completed one": "done"
        }
    }
}


Follow the order in `priority_order`. Once a condition matches, stop and output its value.

Output format:
"<one of: prev | next | done | try_next_chunk_for_options>"
No explanations. No reasoning text.
"""
        print("\nText Received\n", text)
        # text = f"system prompt:\n{system_prompt}" + f"\nprompt:\n{text}"
        response = chat(model='llama3:8b-instruct-q4_0',
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text}
                            ])
        return response["message"]["content"]
    
    def clean_question_chunk(self, text):
        system_prompt = \
        """
            "You are a text-cleaning model specialized in exam data.\n\n"
            "Your task:\n"
            "- Receive raw question text mixed with metadata, IDs, hashes, and repeated blocks and a previous question.\n"
            "- Output only human-readable, non-technical question and option text.\n\n"
            "### OUTPUT RULES\n"
            "1. Keep:\n"
            "- The actual question statements and question number, options, and meaningful instructions.\n"
            "- Examples:\n"
            '   - "Based on the above data, answer the given subquestions."\n'
            '   - "Select the valid adjacency representations of the tour."\n'
            '   - Lists like "M, U, S, K, O, V, R, T, P, L, N, Q"\n\n'
            "2. Remove lines starting with:\n"
            "   Question Id, Question Type, Correct Marks, Question Label,\n"
            "   Sub-Section, Max. Selectable Options, Group Comprehension Questions, Question Pattern Type.\n"
            "   Also remove numeric prefixes like '6406532787170.' before text.\n"
            "   Remove duplicates or repeated question blocks.\n\n"
            "3. Normalize:\n"
            "- Collapse extra blank lines.\n"
            "- Ensure options are separated by newlines.\n"
            "- Preserve order.\n\n"
            4. Strictly I don't want answer from you. If you don't find any question return - 'no question found', nothing extra no comments etc.
            "### OUTPUT FORMAT\n"
            "Return only clean, human-readable text without markdown or commentary.\n"
        """
        response = chat(model='llama3:8b-instruct-q4_0',
                        messages=[{"role": "system", "content": system_prompt},
                                {"role": "user", "content": text}])
        return response.get("message", {}).get("content", "").strip()

def remove_hash(text):
    return "\n".join(text.split("\n")[:-1])

def document_splitter(
    documents: DocumentSplitterLangChain,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    split_type: str = 'embedding',
    lookBackPower: int = 3,
    subject: str = "AI",
    pattern: str = r"Question"
):
    embedding_splitter = EmbeddingSplitter(model_name)
    splitted_documents = dict()
    if split_type == "embedding":
        split_method = partial(embedding_splitter.embedding_based_splitting, lookBackPower=lookBackPower)
    elif split_type == "regex":
        split_method = partial(embedding_splitter.regex_based_splitting, subject=subject, pattern=pattern)
    elif split_type == "llm":
        split_method = partial(embedding_splitter.llm_based_splitter)

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

    cleaned_documents = dict()
    for file_name, docs in splitted_documents.items():
        cleaned_documents[file_name] = []
        num_docs = len(docs)
        upper_range = 0
        for idx, doc in enumerate(docs):
            print("\n", idx, upper_range)
            input("Press Enter")
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
                response = embedding_splitter.classify_chunk_relationship(text=send_text)
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
                    print("next", f"upper_range: {upper_range}", f"doc idx: {idx}")
                elif response == "try_next_chunk_for_options":
                    text = text + f"\n{remove_hash(docs[idx+1].page_content)}"
                    print("try_next_chunk_for_options")
                elif response == "done":
                    print("done", f"current_idx: {current_idx}", f"doc idx: {idx}")
                else:
                    print(response)
                    print("No condition met")

                print(f"\nTextD:\n{text}\n")
                input("Press Enter")
            upper_range += 1

            cleaned_question = embedding_splitter.clean_question_chunk(text=text)
            if not("no question found" in cleaned_question):
                cleaned_documents[file_name].append(cleaned_question)


def document_splitter_cs(chunk_size): ...