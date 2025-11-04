import unittest

from Main.Embeddings.embedding import CustomEmbeddings


class TestCustomEmbed(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = CustomEmbeddings(model_name=self.model_name)
        self.embedding_dim = 384

    def test_embed_doc(self):
        texts = ["This is a dummy text"]
        embeddings = self.embedding_model.embed_documents(texts)
        self.assertEqual(tuple(embeddings.shape), (1, 384))
    
    def test_embed_query(self):
        texts = "This is a dummy text"
        embeddings = self.embedding_model.embed_query(texts)
        self.assertEqual(tuple(embeddings.shape), (384,))

if __name__ == "__main__":
    unittest.main()