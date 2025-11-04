from typing import List

from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings.embeddings import Embeddings
import torch


class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-m3", normalize: bool = True):
        if not isinstance(model_name, str):
            raise ValueError(f"model_name should be of type str, got {type(model_name)}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.normalize = normalize

        torch.set_grad_enabled(False)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (texts)."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device) # <-- moving tensors to device
        with torch.no_grad():
            out = self.model(**inputs)
            embeddings = out.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]
