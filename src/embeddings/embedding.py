from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

class E5Embeddings(Embeddings):

    """
    Обертка Ембендинга из HuggingFaceEmbeddings
    - добавляет 'passage' к документам
    - добавляет 'query' к запросам 
    """

    def __init__(
            self, model_name: str='intfloat/multilingual-e5-base', 
            query_prefix: str = "query:", 
            doc_prefix: str = "passage:",
        ):
        self._hf = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True} # включаем нормализацию
        )
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        docs = [
            t if t.startswith(self.doc_prefix) else f"{self.doc_prefix} {t}"
            for t in texts
        ]
        return self._hf.embed_documents(docs)

    def embed_query(self, text: str) -> List[float]:
        q = text if text.startswith(self.query_prefix) else f"{self.query_prefix} {text}"
        return self._hf.embed_query(q)