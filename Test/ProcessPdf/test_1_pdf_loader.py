# Folder to check the loading of a directory or file
# I will add the visual testing - the next test only runs when the visualizer passes - they can pass all test at once

import os
import unittest

from Main.ProcessPdf.load_pdf import (
                                    get_documents_langchain,
                                    document_splitter,
                                    EmbeddingSplitter,
                                    Clean
                                    )
from Main.models import DocumentSplitterLangChain



class TestProcessPdf(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = os.path.join("Test", "ProcessPdf", "TestPdfDir")
        self.assertTrue(os.path.exists(self.test_dir), f"{self.test_dir} not found!")
        self.documents: DocumentSplitterLangChain = await get_documents_langchain(input_dir=self.test_dir)
        self.file_names = tuple(self.documents.documents.keys())
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    async def test_embedding_splitter(self):
        self.splitted_documents = document_splitter(documents=self.documents, sentence_embed_model_name=self.model_name)
        file_name_1 = self.file_names[0]
        self.assertGreater(len(self.splitted_documents[file_name_1]), len(self.documents[file_name_1]),\
                        "The documents are not splitted further")
        self.assertEqual(first=self.splitted_documents[file_name_1][0].metadata["chunk_number"], second=0, msg="correct chunk_number is not assigned")

        for doc in self.splitted_documents[file_name_1]:
            print("\n", doc.page_content)
    
    async def test_llm_splitter(self):
        self.splitted_documents = document_splitter(documents=self.documents, sentence_embed_model_name=self.model_name, split_type="llm")
        print(self.splitted_document)

    def test_splitting_tokens(self):
        # I am not happy with the current settings
        splitted_docs: DocumentSplitterLangChain = document_splitter(
            documents=self.documents,
            split_type="token"
        )
        old_docs_len = len(self.documents.documents[self.file_names[0]])
        new_docs_len = len(splitted_docs.documents[self.file_names[0]])
        self.assertGreaterEqual(new_docs_len, old_docs_len)

        for _, docs in splitted_docs.documents.items():
            for doc in docs:
                print(f"\n{doc.page_content}")

    def test_clean_docs(self):
        print(type(self.documents))
        splitted_docs: DocumentSplitterLangChain = document_splitter(
            documents=self.documents,
            split_type="token"
        )
        cleaned_docs: DocumentSplitterLangChain = Clean.clean_docs(documents=splitted_docs)
        for _, docs in cleaned_docs.documents.items():
            for doc in docs:
                print(doc)


if __name__ == "__main__":
    unittest.main()