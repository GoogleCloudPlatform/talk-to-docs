"""
Defines data models and state management classes for the Gen AI project.

This module provides the following:

* **Data Models:**  
    * `PersonalizedData`:  Represents personalized member information.
    * `ItemInput`:  Structures input for question-answering requests.
    * `ResponseInput`:  Encapsulates data for feedback on responses.
    * `ResponseOutput`, `ResetInput`, `ResetOutput`: Models for handling feedback and reset interactions.
    * `LLMOutput`:  Structures the output format of language model responses.

* **State Management:**
    * `QueryState`:  Tracks the details and evolution of a single query for thorough analysis.
    * `Conversation`: Manages the history of a user's interaction with the Gen AI system. 
"""

from dataclasses import dataclass, field
from pydantic import BaseModel, HttpUrl
from datetime import datetime
from typing import List, Optional, Any
from fastapi import UploadFile

from pydantic import BaseModel


def transform_to_dictionary(base_model: BaseModel) -> dict:
    """
    Transform a Pydantic BaseModel instance into a dictionary containing only
    attributes that do not have their default values.

    Args:
        base_model (BaseModel): An instance of a Pydantic BaseModel.

    Returns:
        dict: A dictionary with keys and values from the BaseModel instance where
              the values are not equal to their defined default values.
    """
    return {k: v for k, v in base_model.model_dump().items() if v != base_model.model_fields[k].default}


class PersonalizedData(BaseModel):
    """Represents personalized policy and patient data.

    This class models the personalized information associated with an insurance
    policy and the patient covered by the policy.

    Attributes:
        member_id (str): Unique identifier for the policy member.
        policy_title (str): Title or name of the insurance policy.
        policy_holder_name (str): Full name of the policyholder.
        patient_first_name (str): First name of the patient.
        patient_last_name (str): Last name of the patient.
        patient_age (int): Age of the patient in years.
        patient_gender (str): Gender of the patient (e.g., "M", "F", "Other").
        effective_date (str): Date the policy becomes effective (YYYY-MM-DD).
        agent_name (str): Name of the insurance agent.
        set_number (str): A set identifier for the policy.
        policy_number (str): Unique identifier for the insurance policy.
        session_id (str): Identifier for the current session or interaction.
        asof_date (str): The date as of the claim/request si to be made (YYYY-MM-DD).
    """

    member_id: str = ""
    policy_title: str = ""
    policy_holder_name: str = ""
    patient_first_name: str = ""
    patient_last_name: str = ""
    patient_age: int = 0
    patient_gender: str = ""
    effective_date: str = ""
    agent_name: str = ""
    set_number: str = ""
    policy_number: str = ""
    session_id: str = ""
    cob_status: str = ""
    asof_date: str = ""


class IndexDocumentsResponse(BaseModel):
    status: bool
    message: str
    lro_id: str


class DocumentsRequest(BaseModel):
    user_id: str
    client_project_id: str


class DocumentsList(BaseModel):
    document_id: str
    document_uri: str
    document_filename: str
    document_client_project_id: str


class ListDocumentsResponse(BaseModel):
    user_id: str
    client_project_id: str
    documents:list[DocumentsList]


class RemoveDocumentsRequest(BaseModel):
    user_id: str
    document_ids: list[str]


class RemoveDocumentsResponse(BaseModel):
    status: bool
    message: str


class ViewExtractedDocumentResponse(BaseModel):
    status: bool
    document_id: str
    context: str


class ItemInput(BaseModel):
    question: str
    member_context_full: PersonalizedData


class ResponseInput(BaseModel):
    question: str
    answer: str
    response_id: str
    rank: int


class ResponseOutput(BaseModel):
    success: bool
    response_id: str


class ResetInput(BaseModel):
    person_info: PersonalizedData


class ResetOutput(BaseModel):
    success: bool


class VAISConfig(BaseModel):
    branch: str = "default_branch"
    bucket_name: str
    collection: str = "default_collection"
    company_name: str
    data_store_id: str
    dataset_name: str
    engine_id: str
    location: str
    metadata_filename: str
    metadata_folder: str
    source_folder: str


from pydantic import BaseModel
from typing import Optional
import base64

class LocalDocument(BaseModel):
    file_object: str
    file_title: str

    @classmethod
    def from_bytearray(cls, file_object: bytearray, file_title: str):
        return cls(
            file_object=base64.b64encode(file_object).decode("utf-8"),
            file_title=file_title
        )

    def to_bytearray(self) -> bytearray:
        return bytearray(base64.b64decode(self.file_object))


class ExternalDocument(BaseModel):
    document_url: HttpUrl
    created_on: datetime
    document_name: str


class CreateProjectInput(BaseModel):
    project_name: str
    user_id: str
    local_documents: Optional[List[UploadFile]] = []
    external_documents: Optional[List[ExternalDocument]] = []

class ChatOutput(BaseModel):
    is_ai: bool
    message: str
    prediction_id: str
    like_status: str | None
    icon: str | None

class LLMOutput(BaseModel):
    """
    A model representing the structured output of a conversational AI response.

    This class is used to encapsulate the response from a conversational AI model, organizing
    the information into a structured format that can be easily returned from an API endpoint.
    It extends Pydantic's BaseModel, leveraging Pydantic's data validation and serialization
    capabilities.

    Attributes:
        round_number (str): The conversation round number, indicating the sequence of the
                            current interaction within the conversation.
        answer (str): The textual response or answer generated by the AI model for the current
                      interaction.
        response_id (str): A unique identifier for the response. This can be used for tracking
                           and referencing specific interactions.
        plan_and_summaries (str): A string containing any plans or summaries that were generated
                                  as part of the AI's processing. This could be empty if not applicable.
        additional_information_to_retrieve (str): Any additional information that the AI model
                                                  suggests should be retrieved or looked up. This field
                                                  could be used to prompt further actions or queries.
        context_used (str): A description or representation of the context that was used by the AI
                            model to generate its response. This helps in understanding the basis
                            of the AI's response.
        urls_to_kc (list[str]): A list of URLs to knowledge components that were referenced or
                                could be relevant to the response. This can help users to explore
                                related topics or verify information.
        attributes_to_kc_km (list[dict[str, str, str]]): A list of quatrlets:
                                - Document type (can be "KM" or "MP"), both come from KC
                                - Document identifier (doc_identifier for KM, original_filepath for MP)
                                - URL (exists only for KM, is empty for MP)
                                - Section Name (exists for all: B360, KC KM, KC MP)
                                This can help users to explore related topics or verify information.
        attributes_to_kc_mp (list[dict[str, str, str]]): A list of quatrlets:
                                - Document type (can be "KM" or "MP"), both come from KC
                                - Original filepath (original_filepath for MP)
                                - Policy number (exists only for MP, is empty for KM)
                                - Section Name (exists for all: B360, KC KM, KC MP)
                                This can help users to explore related topics or verify information.
        attributes_to_b360 (list[dict[str, str]]): A list of tuples:
                                - Set number, exists only in b360
                                - Section name, name of the chunk from B360
                                This is useful for integrating the AI's responses with other systems or databases.
        confidence_score (str): The conversation confidence_score, indicating how confident was the answer.
        session_id (str): The conversation session_id, which can be used to track the answers in BigQuery.

    Note:
        The attributes `plan_and_summaries`, `additional_information_to_retrieve`, `context_used`,
        `urls_to_kc`, and `sections_to_b360` are designed to be flexible and may contain various types
        of information depending on the specific implementation and use case of the conversational AI.
    """

    round_number: str
    answer: str
    response_id: str
    plan_and_summaries: str
    additional_information_to_retrieve: str
    context_used: str
    urls_to_kc: list[str]
    attributes_to_kc_km: list[dict]
    attributes_to_kc_mp: list[dict]
    attributes_to_b360: list[dict]
    confidence_score: str
    session_id: str


@dataclass
class QueryState:
    """A class representing the state of a query.

    Attributes:
        question: The question being asked.
        all_sections_needed: A list of all the sections that are needed to answer the question.
        gt_answer: The ground truth answer to the question.
        output: The output of the query.
        tokens_used: The number of tokens used to generate the answer.
        relevant_context: The context that is relevant to the question.
        urls: A list of urls to the documents used in the answer.
        react_rounds: A list of the react rounds that were used to generate the answer.
        input_tokens: A list of the input tokens that were used to generate the answer.
        num_docs_used: A list of the number of documents that were used to generate the answer.
        used_articles_with_scores: A list of the articles that were used to generate the answer with the scores.
        additional_information_to_retrieve: Any additional information that needs to be retrieved.
        time_taken: The time taken to generate the answer.
        confidence_score: Confidence score of the answer of the final round
        custom_fields: Dictionary from list of dictionaries, that represents information specific to use case
        original_question: Original question before enhancements
    """

    question: str
    all_sections_needed: list[str]
    gt_answer: str | None = field(default=None)
    answer: str | None = field(default=None)
    tokens_used: int | None = field(default=None)
    relevant_context: str | None = field(default=None)
    urls: list[str] = field(default_factory=list)
    react_rounds: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    num_docs_used: list[int] = field(default_factory=list)
    used_articles_with_scores: list[tuple[str, float]] = field(default_factory=list)
    additional_information_to_retrieve: str | None = field(default=None)
    time_taken: int = field(default=0)
    confidence_score: int = field(default=0)
    custom_fields: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    original_question: str | None = field(default=None)


@dataclass
class Conversation:
    """
    Represents a single conversation instance, tracking the exchanges between a user
    and an AI model, along with additional metadata.

    This class is designed to encapsulate the state and progression of a conversation,
    including all exchanges, participant information, and any other relevant details that
    emerge during the interaction.

    Attributes:
        exchanges (list[QueryState]): A list of QueryState objects representing the sequence
                                      of exchanges between the user and the system within this
                                      conversation.
        conversation_num (int): An identifier for the conversation, typically used for tracking
                                or referencing purposes. Defaults to 1.
        user (str | None): The username or identifier of the user involved in the conversation.
                           Defaults to None if not specified.
        date_time (str | None): The date and time when the conversation was initiated or last
                                updated. Defaults to None if not specified.
        plan_name (str | None): The name of the plan or strategy being followed during the
                                conversation. This field's necessity is under consideration,
                                and it may be included for future use or integration with other
                                systems. Defaults to None.
        correct (str | None): Indicates whether the responses provided by the system during the
                              conversation were correct or met the user's expectations. Defaults
                              to None. This could be used for feedback or quality control.
        articles_not_relevant (str | None): Tracks any articles or resources that were deemed
                                             not relevant during the conversation. This can be
                                             used for improving resource recommendations in the
                                             future. Defaults to None.
        round_number (int): The current round number of the conversation, indicating how many
                            exchanges have taken place. Defaults to 1. Note: the attribute name
                            has been corrected from `round_numder` to `round_number`.
        session_id (str): The session id that we use to track between responses.
        member_info (list[PersonalizedData] | None): A list of PersonalizedData objects that
                                                     contain information tailored to the member
                                                     involved in the conversation. Defaults to None.

    Note:
        - The `QueryState` and `PersonalizedData` types should be defined elsewhere in the codebase,
          with `QueryState` capturing the state of individual queries and `PersonalizedData` encapsulating
          personalized information relevant to the user.
        - This class uses Python's dataclass decorator for boilerplate code reduction, automatically
          generating methods like `__init__`.
    """

    exchanges: list[QueryState]
    conversation_num: int = field(default=1)
    user: str | None = field(default=None)
    date_time: str | None = field(default=None)
    plan_name: str | None = field(default=None)  # do we need it??
    correct: str | None = field(default=None)
    articles_not_relevant: str | None = field(default=None)
    round_numder: int = field(default=1)
    session_id: str | None = field(default=None)
    member_info: list[PersonalizedData] | None = field(default=None)
    prediction_id: str | None = field(default=None)
