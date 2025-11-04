import uuid
from typing import List, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document

from Main.models import VDAdd
from Main.Embeddings.embedding import CustomEmbeddings


class Store:
    def __init__(self, embeddings: CustomEmbeddings):
        # Ensure not use sentence transformers here - they are for splitting
        if not isinstance(embeddings, CustomEmbeddings):
            raise TypeError("embeddings must be a CustomEmbeddings instance")
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )

    def _preprocess_add_data(self, data: Union[str, Document], metadata):
        """
            metadata is just chunk number here
        """
        # The only use of this function for data being Document is this part
        # verifying the chunk number
        if isinstance(data, Document) and metadata == None:
            if not("chunk_number" in data.metadata):
                raise ValueError("Document with metadata without chunk number is not allowed")
            if not isinstance(data.metadata.get("chunk_number"), int):
                raise ValueError("chunk_number of Document's metadata should be an integer.")

        document = Document(page_content=data, metadata={"chunk_number": metadata} if not isinstance(metadata, dict) else metadata)\
                    if isinstance(data, str) else data
        return document

    def add_data(self, data: VDAdd):
        """
            metadata[chunk_number]
        """
        # Do not add data with source and chunk number being same

        if isinstance(data, dict):
            # data[file_name] = list[doc] or list[text]
            for _, docs_or_texts in data.items():
                self.add_data(data=docs_or_texts)
        if isinstance(data, VDAdd):
            metadata = data.metadata
            data = data.data
        else:
            # We will never like to provide metadata to be None in case of 
            # data being list[str]
            data = data
            metadata = None
        if isinstance(data, str):
            data = [data]
        if isinstance(data, list) and isinstance(data[0], str):
            assert len(data) == len(metadata), "metadata must have equal number of keys as chunks in data"

        documents = []
        for idx, data_ in enumerate(data):
            metadata_ = metadata[idx] if isinstance(data_, str) and not isinstance(data, Document) else None
            documents.append(self._preprocess_add_data(data_, metadata_))

        uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
        # clean documents
        self.vector_store.add_documents(documents=documents, ids=uuids)
        return uuids

    def query(self, query):
        # the most similar chunk
        most_similar_chunk = self.vector_store.similarity_search(
            query=query,
            k=1
        )
        chunk_number = most_similar_chunk[0].metadata["chunk_number"]

        # new query should consist the info about most similar chunk
        query = f"most_similar_chunk\n: {most_similar_chunk[0].page_content} \n {query}" # <-- risky approach 
        neighbor_chunks = [chunk_number + i for i in range(1, 5)] + [chunk_number + i for i in range(-1, -5, -1)]
        results = self.vector_store.similarity_search(
                    query,
                    k=4,
                    filter={"chunk_number": {"$in": neighbor_chunks}}
                )

        results.append(most_similar_chunk[0])
        # arrange chunks with the order of chunk_number -> could there be chunks with the 
        # different sources? Divide into sources
        results.sort(key=lambda x: x.metadata["chunk_number"])
        return results
    
    def delete(self, ids):
        self.vector_store.delete(ids=ids)
        return True
    
    def get_num_vectors(self):
        all_documents = self.vector_store.get()["documents"]
        total_records = len(all_documents)
        return total_records

    def check_if_exists(self, ids: Union[List[str], str]):
        if isinstance(ids, str):
            ids = [ids]

        not_found_ids = []
        for id in ids:
            retrieved_doc = self.vector_store.get_by_ids([id])
            if retrieved_doc == []:
                not_found_ids.append(id)

        return not_found_ids
    
    def check_if_ids_deleted(self, ids):
        not_found_ids = self.check_if_exists(ids)
        not_found_ids.sort(key=lambda x: x)
        ids.sort(key=lambda x: x)
        return not_found_ids == ids