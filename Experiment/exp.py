# Main\ProcessPdf\load_pdf.py
def get_documents_llama(input_dir: str):
    os.path.exists(input_dir)
    reader = SimpleDirectoryReader(input_dir=input_dir)
    documents = reader.load_data()
    return documents
