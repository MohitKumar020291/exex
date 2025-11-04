# Folder to check the loading of a directory or file
# I will add the visual testing - the next test only runs when the visualizer passes - they can pass all test at once

import os
import unittest

from Main.ProcessPdf.load_pdf import (
                                    get_documents_langchain, 
                                    document_splitter
                                    )


class TestProcessPdf(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = os.path.join("Test", "ProcessPdf", "TestPdfDir")
        self.assertTrue(os.path.exists(self.test_dir), f"{self.test_dir} not found!")
        self.documents = await get_documents_langchain(input_dir=self.test_dir)
        self.file_names = tuple(self.documents.keys())
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    async def test_dir_reading_langchain(self):
        documents = await get_documents_langchain(input_dir=self.test_dir)
        self.assertGreater(len(documents), 0, f"No document is loaded in {self.test_dir}")
        for _, list_docs in documents.items():
            for doc in list_docs:
                print(doc.page_content)
            break
    
    async def test_embedding_splitter(self):
        self.splitted_documents = document_splitter(documents=self.documents, model_name=self.model_name)
        file_name_1 = self.file_names[0]
        self.assertGreater(len(self.splitted_documents[file_name_1]), len(self.documents[file_name_1]),\
                        "The documents are not splitted further")
        self.assertEqual(first=self.splitted_documents[file_name_1][0].metadata["chunk_number"], second=0, msg="correct chunk_number is not assigned")

        for doc in self.splitted_documents[file_name_1]:
            print("\n", doc.page_content)
    
    async def test_llm_splitter(self):
        self.splitted_documents = document_splitter(documents=self.documents, model_name=self.model_name, split_type="llm")
        print(self.splitted_document)
        ...


if __name__ == "__main__":
    unittest.main()