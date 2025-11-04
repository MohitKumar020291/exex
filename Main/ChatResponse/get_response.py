import re
from ollama import chat, ChatResponse


def get_response(prompt: str):
    """
    Clean raw exam data: remove noise, extract useful human-readable question content.
    Handles long context by splitting the document into smaller chunks
    while preserving limited context from previous outputs.
    """

    system_prompt = (
        "You are a text-cleaning model specialized in exam data.\n\n"
        "Your task:\n"
        "- Receive raw question text mixed with metadata, IDs, hashes, and repeated blocks.\n"
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
        "### OUTPUT FORMAT\n"
        "Return only clean, human-readable text without markdown or commentary.\n"
    )

    # Split document by visual delimiters like =======HASH=======
    chunks = re.split(r"=+[A-Za-z0-9]+=+", prompt)
    # for chunk in chunks:
    #     print(chunk)
    # print("\nprompt")
    # print(prompt)
    # return

    cleaned_outputs = []
    print("\n")

    for i, chunk in enumerate(chunks):
        prev_context = (
            "Previous cleaned content (for continuity):\n"
            + "\n".join(cleaned_outputs[-2:])  # keep only last 2 chunks for context
            if cleaned_outputs else "No previous context."
        )

        print("previous context\n", prev_context)

        current_input = (
            f"{prev_context}\n\n"
            f"Now clean this new part of the document:\n{chunk.strip()}\n"
        )
        print("current input\n", current_input)

        response: ChatResponse = chat(
            model="llama3:8b-instruct-q2_K",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_input},
            ],
        )

        content = response["message"]["content"].strip()
        cleaned_outputs.append(content)

        print("content\n", content)

    print("=======================")
    final_system_prompt = """
            You were given some chunks of data which you have cleaned and extracted out the important info.
            Your next task is to give the final output from the contexts you have generated and I am also attaching the prompt which
            led you to generate these context.
        """
    final_prompt = "Context we have built:" + "\n\n".join(cleaned_outputs) + "prompt that led to generate these:" + f"\n{prompt}"
    final_output: ChatResponse = chat(
            model="llama3:8b-instruct-q2_K",
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": final_prompt},
            ],
        )
    return final_output["message"]["content"]