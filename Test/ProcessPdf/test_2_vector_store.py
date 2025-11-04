import unittest
import os

import torch

from Main.VectorStore.vector_database import Store
from Main.Embeddings.embedding import CustomEmbeddings
from Main.models import VDAdd
from Main.ProcessPdf.load_pdf import (
                                    get_documents_langchain, 
                                    document_splitter
                                    )


class TestVectorStore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model_name =  "BAAI/bge-m3" # <-- this should be an embedding for text - splitting is a different thing
        self.embeddings = CustomEmbeddings(model_name=self.model_name)
        self.vector_store = Store(embeddings=self.embeddings) # <- where is this vector store actually living on my pc
        self.test_dir = os.path.join("Test", "ProcessPdf", "TestPdfDir")
        self.documents = await get_documents_langchain(input_dir=self.test_dir)

    def test_setup_vector_store(self):
        self.assertIsNotNone(self.vector_store, "vector_store is initialized to None")

    def test_add_and_delete_data(self):
        texts = ["This is first chunk.", "This is second chunk.",\
                "This is third chunk.", "This is fourth chunk."]
        metadata = [1, 2, 3, 4]
        data_input = VDAdd(data=texts, metadata=metadata)
        uuids = self.vector_store.add_data(data=data_input)
        # delete data before running test
        self.assertTrue(expr=self.vector_store.delete(ids=uuids), msg="vectors not deleted successfully")
        self.assertTrue(expr=self.vector_store.check_if_ids_deleted(ids=uuids), msg="deleted ids are still present in the vector store")

    async def test_splitter_and_vs(self):
        documents = self.documents
        file_names = tuple(documents.keys())
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        splitted_documents = document_splitter(documents=documents, model_name=model_name)

        splitted_documents_file_1 = splitted_documents[file_names[0]]
        uuids = self.vector_store.add_data(data=splitted_documents_file_1)
        # Handle this deletion better
        self.assertTrue(expr=self.vector_store.delete(ids=uuids), msg="vectors not deleted successfully")
        self.assertTrue(expr=self.vector_store.check_if_ids_deleted(ids=uuids), msg="deleted ids are still present in the vector store")

    async def test_query(self):
        
        query = "what is question 37"
        results = self.vector_store.query(query=query)

        for result in results:
            print(result.page_content)


if __name__ == "__main__":
    unittest.main()