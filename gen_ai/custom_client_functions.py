from gen_ai.deploy.model import QueryState
from langchain.schema import Document
from gen_ai.common.storage import Storage
import os
from collections import defaultdict
import gen_ai.common.common as common

from gen_ai.common.argo_logger import trace_on
import copy
from langchain_community.vectorstores.chroma import Chroma
from gen_ai.common.document_retriever import SemanticDocumentRetriever
from typing import Any


def default_fill_query_state_with_doc_attributes(query_state: QueryState, post_filtered_docs: list[Document]) -> QueryState:
    """Updates the provided query_state object with attributes extracted from documents after filtering.

    This function iterates through each document in the `post_filtered_docs` list.  For each key-value pair in a document's metadata, it adds the value to the corresponding list in the `custom_fields` field of the `query_state`. If the key doesn't exist yet, a new list is created.

    Args:
        query_state: The QueryState object to be modified.
        post_filtered_docs: A list of Document objects containing metadata.

    Returns:
        The modified QueryState object, with custom_fields updated based on document metadata.
    """
    for document in post_filtered_docs:
        for key, value in document.metadata.items():
            if key not in query_state.custom_fields:
                query_state.custom_fields[key] = []
            query_state.custom_fields[key].append(value)

    return query_state


def custom_fill_query_state_with_doc_attributes(query_state: QueryState, post_filtered_docs: list[Document]) -> QueryState:
    """
    Updates the provided query_state object with attributes extracted from documents after filtering.

    This function modifies the query_state object by setting various attributes based on the metadata of documents
    in the post_filtered_docs list. It processes documents to categorize them by their data source
    (B360, KM or MP from KC), and updates the query_state with URLs, and categorized attributes for each type.

    Args:
        query_state (QueryState): The query state object that needs to be updated with document attributes.
        post_filtered_docs (list[Document]): A list of Document objects that have been filtered and whose attributes
        are to be extracted.

    Returns:
        QueryState: The updated query state object with new attributes set based on the provided documents.

    Side effects:
        Modifies the query_state object by setting the following attributes:
        - urls: A set of unique URLs extracted from the document metadata.
        - attributes_to_b360: A list of dictionaries with attributes from B360 documents.
        - attributes_to_kc_km: A list of dictionaries with attributes from KC KM documents.
        - attributes_to_kc_mp: A list of dictionaries with attributes from KC MP documents.

    """
    query_state.urls = list(set(document.metadata["url"] for document in post_filtered_docs))

    # B360 documents
    b360_docs = [x for x in post_filtered_docs if x.metadata["data_source"] == "b360"]
    attributes_to_b360 = [
        {"set_number": x.metadata["set_number"], "section_name": x.metadata["section_name"]} for x in b360_docs
    ]

    # KC documents, they can be of two types: from KM (dont have policy number) and from MP (have policy number)
    kc_docs = [x for x in post_filtered_docs if x.metadata["data_source"] == "kc"]
    kc_km_docs = [x for x in kc_docs if not x.metadata["policy_number"]]
    kc_mp_docs = [x for x in kc_docs if x.metadata["policy_number"]]

    attributes_to_kc_km = [
        {
            "doc_type": "km",
            "doc_identifier": x.metadata["doc_identifier"],
            "url": x.metadata["url"],
            "section_name": x.metadata["section_name"],
        }
        for x in kc_km_docs
    ]
    attributes_to_kc_mp = [
        {
            "doc_type": "mp",
            "original_filepath": x.metadata["original_filepath"],
            "policy_number": x.metadata["policy_number"],
            "section_name": x.metadata["section_name"],
        }
        for x in kc_mp_docs
    ]
    query_state.custom_fields["attributes_to_b360"] = attributes_to_b360
    query_state.custom_fields["attributes_to_kc_km"] = attributes_to_kc_km
    query_state.custom_fields["attributes_to_kc_mp"] = attributes_to_kc_mp

    return query_state


def default_extract_doc_attributes(docs_and_scores: list[Document]) -> list[tuple[str]]:
    """Extracts all metadata attributes from a list of Document objects.

    Args:
        docs_and_scores: A list of Document objects, typically returned from a search operation.

    Returns:
        A list of tuples where each tuple contains all metadata attributes from a Document, in an arbitrary order. The order of the attributes in each tuple may vary depending on the underlying dictionary implementation.
    """
    return [
        tuple([value for _, value in x.metadata.items()])
        for x in docs_and_scores
    ]


def custom_extract_doc_attributes(docs_and_scores: list[Document]) -> list[tuple[str]]:
    """Extracts specific metadata attributes from a list of Document objects.

    Args:
        docs_and_scores: A list of Document objects, typically returned from a search operation.

    Returns:
        A list of tuples where each tuple contains the following metadata attributes from a Document:
            - original_filepath: The original file path of the document.
            - doc_identifier: A unique identifier for the document.
            - section_name: The name of the section within the document.
    """
    return [
        (x.metadata["original_filepath"], x.metadata["doc_identifier"], x.metadata["section_name"])
        for x in docs_and_scores
    ]


def remove_member_and_session_id(metadata: dict[str, Any]) -> dict[str, Any]:
    """Removes the "member_id" key and "session_id" from a metadata dictionary.

    This function creates a copy of the input dictionary, deletes the "member_id" and "session_id" key from
    the copy, and returns the modified copy.

    Args:
        metadata (dict): The input metadata dictionary.

    Returns:
        dict: A new dictionary with the "member_id" and "session_id" key removed.
    """
    new_metadata = copy.deepcopy(metadata)
    if "member_id" in new_metadata:
        del new_metadata["member_id"]
    if "session_id" in new_metadata:
        del new_metadata["session_id"]
    return new_metadata

class CustomStorage(Storage):
    """
    Provides a customized document storage strategy for Woolworth-specific document processing.
    This class handles text files by extracting their content, augmenting each document with
    metadata from a corresponding JSON file named similarly to the text file but with a '_metadata.json' suffix.

    Designed to work within a fixed plan name 'se', this class assumes all documents belong to a single
    organizational unit, making it specialized for scenarios where document categorization by plan name
    is uniform and predefined.
    """

    def process_directory(self, content_dir: str, woolworth_extract_data: callable) -> dict[str, list[Document]]:
        """
        Go through files in content_dir and parse their content if the filename ends with ".txt".
        Generate a document object from each file and store it in a hashmap where the key is the
        plan name and the value is a list of document objects. Return the hashmap.
        """

        documents_hashmap = defaultdict(list)
        plan_name = "se"
        for filename in os.listdir(content_dir):
            if filename.endswith(".txt") and "_metadata.json" not in filename:
                file_path = os.path.join(content_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                document = woolworth_extract_data(content)
                filename_metadata = file_path.replace(".txt", "_metadata.json")
                metadata = common.read_json(filename_metadata)
                for k, v in metadata.items():
                    document.metadata[k] = v

                documents_hashmap[plan_name].append(document)
        return documents_hashmap


class CustomSemanticDocumentRetriever(SemanticDocumentRetriever):
    """Implements document retrieval based on semantic similarity from a Chroma store.

    This retriever utilizes semantic similarity searches and max marginal relevance (MMR)
    algorithms to identify and rank documents from a Chroma vector store that are most
    relevant to a given query string. The process can be optionally refined using
    metadata filters to narrow down the search results further.

    Attributes:
        store (Chroma): The Chroma vector store instance from which documents are retrieved.
        questions_for_search (str): The query string used for finding related documents.
        metadata (dict, optional): Additional metadata for filtering the documents in the
            search query.
    """

    @trace_on("Retrieving documents from semantic store", measure_time=True)
    def get_related_docs_from_store(
        self, store: Chroma, questions_for_search: str, metadata: dict[str, str] | None = None
    ) -> list[Document]:
        # Very custom method
        metadata = remove_member_and_session_id(metadata)
        if metadata is None or "set_number" not in metadata:
            custom_metadata = {"data_source": "kc"}
            return self._get_related_docs_from_store(store, questions_for_search, custom_metadata)
        
        b360_metadata = copy.deepcopy(metadata)
        b360_metadata["data_source"] = "b360"

        kc_metadata = copy.deepcopy(metadata)
        kc_metadata["data_source"] = "kc"
        if "set_number" in kc_metadata:
            del kc_metadata["set_number"]
        metadatas = [b360_metadata, kc_metadata]
        docs = []
        for metadata in metadatas:
            docs.extend(self._get_related_docs_from_store(store, questions_for_search, metadata))

        return docs


def default_build_doc_title(metadata: dict[str, str]) -> str:
    """Constructs a document title string based on provided metadata.

    This function takes a dictionary containing various metadata fields,
    including "set_number," "section_name," "doc_identifier," and "policy_number."
    It concatenates these values to form a document title string.

    Args:
        metadata (dict[str, str]): A dictionary with potential metadata fields.
            - "set_number": An identifier representing the set number.
            - "section_name": The name of the relevant section.
            - "doc_identifier": A unique identifier for the document.
            - "policy_number": The specific number of the associated policy.
            - "symbols": The symbols of the document.

    Returns:
        str: A concatenated string containing the document title information
        based on the provided metadata fields.

    """
    doc_title = ""
    if metadata.get("set_number"):
        doc_title += metadata["set_number"] + " "
    if metadata.get("section_name"):
        doc_title += metadata["section_name"] + " "
    if metadata.get("doc_identifier"):
        doc_title += metadata["doc_identifier"] + " "
    if metadata.get("policy_number"):
        doc_title += metadata["policy_number"] + " "
    if metadata.get("symbols"):
        doc_title += metadata["symbols"] + " "
    return doc_title

build_doc_title = default_build_doc_title
extract_doc_attributes = default_extract_doc_attributes
fill_query_state_with_doc_attributes = default_fill_query_state_with_doc_attributes
