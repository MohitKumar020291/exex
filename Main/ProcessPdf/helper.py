import os
import warnings

def read_prompts(file_name):
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