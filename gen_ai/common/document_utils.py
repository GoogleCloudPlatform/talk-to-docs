"""This module provides functions for converting documents to Langchain and dictionaries, as well as
generation of contexts from the list of docs"""

from dependency_injector.wiring import inject
from langchain.schema import Document


from typing import Any

from gen_ai.common.measure_utils import trace_on
from gen_ai.common.common import TokenCounter, split_large_document, update_used_docs
from gen_ai.common.ioc_container import Container
from gen_ai.custom_client_functions import build_doc_title, extract_doc_attributes
from gen_ai.deploy.model import QueryState


def convert_langchain_to_json(doc: Document) -> dict[str, Any]:
    """
    Converts a `Document` object to a JSON-serializable dictionary.

    This function is specifically designed for converting documents from the custom `Document`
    format used within the langchain schema to a JSON format for storage or further processing.

    Args:
        doc (Document): The `Document` object to convert.

    Returns:
        dict: A dictionary representation of the `Document`, suitable for JSON serialization.
    """
    doc_json = {}
    doc_json["page_content"] = doc.page_content
    doc_json["metadata"] = doc.metadata
    return doc_json


def convert_json_to_langchain(doc: dict[str, Any]) -> Document:
    """
    Converts a JSON-serializable dictionary to a `Document` object.

    This function allows for the reverse operation of `convert_langchain_to_json`, enabling
    the reconstruction of a `Document` object from its JSON dictionary representation.

    Args:
        doc (dict): The dictionary representation of a `Document`.

    Returns:
        Document: The reconstructed `Document` object.
    """
    return Document(page_content=doc["page_content"], metadata=doc["metadata"])


def convert_dict_to_summaries(doc: dict) -> dict[str, Any]:
    """
    Converts a `Document` object to a JSON-serializable dictionary.

    This function is specifically designed for converting documents from the custom `Document`
    format used within the langchain schema to a JSON format for storage or further processing.

    Args:
        doc (Document): The `Document` object to convert.

    Returns:
        dict: A dictionary representation of the `Document`, suitable for JSON serialization.
    """
    doc_json = {}
    doc_json["summary"] = doc["metadata"]["summary"]
    doc_json["summary_reasoning"] = doc["metadata"]["summary_reasoning"]
    return doc_json


def convert_dict_to_relevancies(doc: dict) -> dict[str, Any]:
    """
    Converts a `Document` object to a JSON-serializable dictionary.

    This function is specifically designed for converting documents from the custom `Document`
    format used within the langchain schema to a JSON format for storage or further processing.

    Args:
        doc (Document): The `Document` object to convert.

    Returns:
        dict: A dictionary representation of the `Document`, suitable for JSON serialization.
    """
    doc_json = {}
    doc_json["relevancy_score"] = doc["metadata"]["relevancy_score"]
    doc_json["relevancy_reasoning"] = doc["metadata"]["relevancy_reasoning"]
    return doc_json


@inject
@trace_on("Generating context from documents", measure_time=True)
def generate_contexts_from_docs(docs_and_scores: list[Document], query_state: QueryState | None = None) -> list[str]:
    """
    Generates textual contexts from a list of documents, preparing them for input to a language model.

    This function processes each document to extract content up to a specified token limit, organizing this content
    into manageable sections that fit within the maximum context size for a language model. It handles large documents
    by splitting them into chunks and maintains a count of used tokens and documents to optimize the subsequent
    language model processing.

    Args:
        docs_and_scores (list[Document]): A list of Document objects, each containing metadata and content
            to be used in generating context. These documents are assumed to be scored and potentially filtered
            by relevance to the query.
        query_state (QueryState): The current state of the query, including details like previously used tokens
            and documents. This state is updated during the function execution to include details about the
            documents and tokens used in this invocation.

    Returns:
        list[str]: A list of strings, where each string represents a textual context segment formed from the
            document content. Each segment is designed to be within the token limits suitable for processing
            by a language model.

    Raises:
        ValueError: If any document does not contain the necessary content or metadata for processing.

    Examples:
        >>> docs = [Document(page_content="Content of the document.",
        metadata={"section_name": "Section 1", "summary": "Summary of the document.", "relevancy_score": 0.95})]
        >>> query_state = QueryState(question="What is the purpose of the document?",
        answer="To provide information.", additional_information_to_retrieve="")
        >>> contexts = generate_contexts_from_docs(docs, query_state)
        >>> print(contexts[0])
        "Content of the document."

    Note:
        The function modifies the `query_state` object in-place, updating it with details about the tokens and
        documents used during the context generation process. Ensure that the `query_state` object is appropriately
        handled to preserve the integrity of the conversation state.
    """
    num_docs_used = [0]
    contexts = ["\n"]
    token_counts = [0]
    used_articles = []
    token_counter: TokenCounter = Container.token_counter()
    max_context_size = Container.config.get("max_context_size", 1000000)

    for doc in docs_and_scores:
        filename = doc.metadata["section_name"]

        doc_content = doc.page_content if Container.config.get("use_full_documents", False) else doc.metadata["summary"]
        doc_tokens = token_counter.get_num_tokens_from_string(doc_content)
        if doc_tokens > max_context_size:
            doc_chunks = split_large_document(doc_content, max_context_size)
        else:
            doc_chunks = [(doc_content, doc_tokens)]

        for doc_chunk, doc_tokens in doc_chunks:

            if token_counts[-1] + doc_tokens >= max_context_size:
                token_counts.append(0)
                contexts.append("\n")
                num_docs_used.append(0)

            used_articles.append((f"{filename} Context: {len(contexts)}", doc.metadata["relevancy_score"]))
            token_counts[-1] += doc_tokens

            contexts[-1] += "DOCUMENT TITLE: "
            contexts[-1] += build_doc_title(doc.metadata) + "\n"
            contexts[-1] += "DOCUMENT CONTENT: "
            contexts[-1] += doc_chunk
            contexts[-1] += "\n" + "-" * 12 + "\n"
            num_docs_used[-1] += 1

    contexts[-1] += "\n"

    Container.logger().info(msg=f"Docs used: {num_docs_used}, tokens used: {token_counts}")

    if query_state:
        query_state.input_tokens = token_counts
        query_state.num_docs_used = num_docs_used
        query_state.used_articles_with_scores = update_used_docs(used_articles, query_state)
        Container.logger().info(msg=f"Doc names with relevancy scores: {query_state.used_articles_with_scores}")

    doc_attributes = extract_doc_attributes(docs_and_scores)
    Container.logger().info(msg=f"Doc attributes: {doc_attributes}")
    return contexts
