import os
import unittest

from Main.Embeddings.embedding import CustomEmbeddings
from Main.ProcessPdf.load_pdf import (
    get_documents_langchain,
    document_splitter
)
from Main.VectorStore.vector_database import Store
from Main.ChatResponse.get_response import get_response 

class ChatResponse(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir =  os.path.join("Test", "ProcessPdf", "TestPdfDir") # <-- we can change this test_dir for AWS file system
        self.embeddings = CustomEmbeddings()
        self.vector_store = Store(embeddings=self.embeddings)
        ...

    def test_cleaning_retrieval(self):
        # documents = get_documents_langchain(input_dir=self.test_dir)
        # splitted_documents = document_splitter(documents)
        # self.vector_store.add_data(data=splitted_documents)
        query = "what is question 37"
        results = self.vector_store.query(query=query)
        results_text = [result.page_content for result in results]
        prompt = "".join(results_text)
        response = get_response(prompt)
        print(response)