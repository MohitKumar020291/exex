from typing import Dict, List, Union
from langchain_core.documents import Document
from pydantic import BaseModel


class DocumentSplitterLangChain(BaseModel):
    documents: Dict[str, List[Document]]


class VDAdd(BaseModel):
    data: Union[str, Document, List[Document], List[str]]
    metadata: List = []


if __name__ == "__main__":

    ...