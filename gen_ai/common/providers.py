"""Provides classes for retrieving documents based on semantic analysis.

This module contains the `DocumentRetrieverProvider` for selecting the appropriate document
retriever based on a given criterion (e.g., semantic analysis) and the abstract base class
`DocumentRetriever` alongside its implementation, `SemanticDocumentRetriever`.
"""

from gen_ai.common.document_retriever import SemanticDocumentRetriever
from gen_ai.custom_client_functions import CustomSemanticDocumentRetriever, CustomStorage
from gen_ai.common.storage import DefaultStorage


class DocumentRetrieverProvider:
    def __call__(self, name: str) -> "DocumentRetriever":
        if name == "semantic":
            return SemanticDocumentRetriever()
        elif name == "custom":
            return CustomSemanticDocumentRetriever()
        else:
            raise ValueError("Not implemented document retriver")


class StorageProvider:
    def __call__(self, name: str) -> "Storage":
        if name == "default":
            return DefaultStorage()
        elif name == "custom":
            return CustomStorage()
        else:
            raise ValueError("Not implemented storage")