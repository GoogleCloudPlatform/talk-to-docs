"""
This module defines functionality for initializing and managing language model chains with robust features like 
exponential retry logic, vector indexing, and output parsing. It includes utility functions for creating configured 
instances of language model chains, output parsers, logging services, and Redis connections. Additionally, it provides 
a dependency injection container to manage and reuse these components efficiently across different parts 
of the application.

The module leverages the dependency_injector package to manage dependencies cleanly and ensure that resources like 
models and database connections are instantiated in a controlled manner. It supports configurations that are 
dynamically loaded and applied to various components such as LLMChains and vector stores.

Functions:
    provide_output_parser() -> BooleanOutputParser: Configures and returns a parser for correcting outputs.
    provide_chain(template_name, input_variables, output_key, llm=None) -> Chain: Returns a configured LLMChain.
    provide_vector_indices(regenerate=False) -> Chroma: Manages and provides Chroma vector indices.
    provide_logger() -> Logger: Configures and returns a standard Python logger for application-wide use.
    provide_redis() -> redis.Redis: Initializes and provides a Redis connection based on predefined settings.

Classes:
    Container: A dependency injection container that provides singletons and resources like LLMChains, Redis and logger

Usage:
    Use the provided functions to obtain configured instances of required components such as LLMChains and Redis.
    The Container class can be used to access these components as singletons throughout the application
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from logging import Logger

import google.auth
import redis
from dependency_injector import containers, providers
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores.chroma import Chroma

import gen_ai.common.common as common
from gen_ai.common.embeddings_provider import EmbeddingsProvider
from gen_ai.common.exponential_retry import LLMExponentialRetryWrapper
from gen_ai.common.storage import DefaultStorage
from gen_ai.common.vector_provider import VectorStrategy, VectorStrategyProvider

LLM_YAML_FILE = "gen_ai/llm.yaml"

from gen_ai.user_context import get_current_client_project_id

class PromptManager:
    def __init__(self):
        self.prompts = {}

    def get_prompt(self, prompt_name: str) -> str:
        client_project_id = get_current_client_project_id()
        if client_project_id is None:
            return None
        if client_project_id not in self.prompts:
            self.prompts[client_project_id] = self._load_prompts(client_project_id)
        return self.prompts[client_project_id].get(prompt_name, "")

    def _load_prompts(self, user_id: str) -> dict:
        return bq_get_prompts(user_id)
    
    def invalidate_cache(self, client_project_id):
        if client_project_id in self.prompts:
            del self.prompts[client_project_id]


def get_dataset_id() -> str:
    project_id = common.load_yaml(LLM_YAML_FILE).get("bq_project_id")
    dataset_name = common.load_yaml(LLM_YAML_FILE)["dataset_name"]
    if not project_id:
        _, project_id = google.auth.default()
    return f"{project_id}.{dataset_name}"


def bq_get_prompts(project_id: str):
    dataset_id = get_dataset_id()

    query = f"""
    WITH ProjectDetails AS (
        -- Fetch project details from the projects table
        SELECT 
            p.project_name,
            p.created_on,
            p.updated_on,
            dp.prompt_name AS default_prompt_name,
            dp.prompt_value AS default_prompt_value,
            dp.prompt_display_name AS default_prompt_display_name,
            pr.prompt_name AS custom_prompt_name,
            pr.prompt_value AS custom_prompt_value
        FROM `{dataset_id}.projects` p
        JOIN `{dataset_id}.project_user` pu 
            ON p.project_id = pu.project_id
        LEFT JOIN `{dataset_id}.default_prompts` dp 
            ON dp.vertical_id = p.vertical_id
        LEFT JOIN `{dataset_id}.prompts` pr
            ON pr.project_id = p.project_id
            AND pr.prompt_name = dp.prompt_name
        WHERE p.project_id = '{project_id}'
    ),
    FilteredPrompts AS (
        SELECT
            project_name,
            created_on,
            updated_on,
            COALESCE(custom_prompt_name, default_prompt_name) AS prompt_name,
            COALESCE(custom_prompt_value, default_prompt_value) AS prompt_value,
            default_prompt_display_name AS prompt_display_name
        FROM ProjectDetails
    )
    SELECT 
        project_name, 
        created_on, 
        updated_on,
        ARRAY_AGG(STRUCT(prompt_name, prompt_value, prompt_display_name)) AS prompt_configuration
    FROM FilteredPrompts
    GROUP BY project_name, created_on, updated_on
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
        ]
    )

    client = Container.logging_bq_client()
    query_job = client.query(query, job_config=job_config)

    results = list(query_job.result())

    project_details = None
    for row in results:
        project_details = {prompt["prompt_name"]: prompt["prompt_value"] for prompt in row.prompt_configuration }

    return project_details


def create_bq_client(project_id: str | None = None) -> bigquery.Client | None:
    """Creates a BigQuery client.
    If project_id is not specified, the default project ID will be used.
    If the default project ID cannot be determined, an error will be raised.
    Args:
        project_id (str, optional): The project ID to use. Defaults to None.
    Returns:
        A BigQuery client.
    """
    if project_id is None:
        try:
            _, project_id = google.auth.default()
        except GoogleAPIError as e:
            print(f"Failed to authenticate: {e}")
            return None
    try:
        client = bigquery.Client(project=project_id)
    except GoogleAPIError as e:
        print(f"Failed to create BigQuery client: {e}")
        return None
    return client


def provide_chain(template_name: str, input_variables: list[str], output_key: str, llm: LLMChain = None) -> Chain:
    """
    Provides an LLMChain instance wrapped with retry logic for specified templates and configurations.

    This function configures and returns an LLMChain based on the specified template configuration and input variables.
    The chain is automatically wrapped with an exponential retry mechanism for added robustness in operation.

    Args:
        template_name (str): The name of the template from configuration to be used for generating prompts.
        input_variables (list[str]): List of strings specifying the variables to be included in the prompt template.
        output_key (str): The key used to retrieve the output from the chain's response.
        llm (LLMChain, optional): An existing LLMChain instance to use; if not provided, one is retrieved from the
        Container.

    Returns:
        Chain: An instance of LLMChain wrapped with exponential retry logic configured to use specified template.
    """
    llm = llm or Container.llm
    template = Container.prompt_manager().get_prompt(template_name)
    if not template:
        template = Container.config[template_name].strip()
    answer_template = PromptTemplate(input_variables=input_variables, template=template)
    chain = LLMChain(
        llm=llm,
        prompt=answer_template,
        output_key=output_key,
        verbose=False,
        llm_kwargs={"response_mime_type": "application/json"},
    )
    return LLMExponentialRetryWrapper(chain)


def provide_vector_indices(regenerate: bool = False) -> Chroma:
    """
    Provides or regenerates vector indices for embeddings using a specified vector strategy.

    This function initializes or updates vector indices based on the configuration specified in LLM_YAML_FILE.
    It manages embeddings and vector strategies to create a Chroma vector store instance suitable for semantic
    operations.

    Args:
        regenerate (bool, optional): If true, existing vector indices are regenerated; otherwise, the current indices
        are used. Defaults to False.

    Returns:
        Chroma: An instance of Chroma vector store populated with the appropriate vector indices for the configured
        embeddings and vector strategy.
    """
    config = common.load_yaml(LLM_YAML_FILE)
    embeddings_name = config.get("embeddings_name")
    embeddings_model_name = config.get("embeddings_model_name")
    vector_name = config.get("vector_name")
    dataset_name = config.get("dataset_name")
    processed_files_dir = config.get("processed_files_dir").format(dataset_name=dataset_name)
    vectore_store_path = config.get("vector_store_path")

    embeddings_provider = EmbeddingsProvider(embeddings_name, embeddings_model_name)
    embeddings: Embeddings = embeddings_provider()

    vector_strategy_provider = VectorStrategyProvider(vector_name)
    vector_strategy: VectorStrategy = vector_strategy_provider(
        storage_interface=DefaultStorage(), config=config, vectore_store_path=vectore_store_path
    )

    local_vector_indices = {}
    return vector_strategy.get_vector_indices(regenerate, embeddings, local_vector_indices, processed_files_dir)


def provide_logger() -> Logger:
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

    return logger


def provide_redis() -> redis.Redis:
    """
    Provides a Redis database connection using predefined settings.

    This function initializes and returns a connection to a Redis database specified by the memory_store_ip constant.
    It sets up the connection with the default port and database index.

    Returns:
        redis.Redis: A Redis client instance connected to the specified Redis server.
    """
    config = common.load_yaml(LLM_YAML_FILE)
    memory_store_ip = config.get("memory_store_ip")
    redis_db = redis.Redis(host=memory_store_ip, port=6379, db=0, decode_responses=True)
    return redis_db


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container that provides singletons and resources for the application.

    This class uses the dependency_injector package to manage and provide configured instances of various components
    such as LLMChains, Redis databases, and loggers. It ensures that components like the LLMChain or vector indices
    are initialized only once and reused throughout the application, providing consistency and efficiency in
    resource usage.

    Attributes:
        config (dict): Configuration loaded from LLM_YAML_FILE.
        llm (LLMChain): Default LLMChain model initialized based on the configuration.
        scoring_llm (LLMChain): Scoring LLMChain model for evaluation purposes.
        react_chain (Provider): Provides a Chain instance for reacting to input.
        json_corrector_chain (Provider): Provides a Chain instance for correcting JSON input.
        aspect_based_summary_chain (Provider): Provides a Chain instance for aspect-based summarization.
        answer_scoring_chain (Provider): Provides a Chain instance for scoring answers.
        retriever_scoring_chain (Provider): Provides a Chain instance for scoring retrievals.
        similar_questions_chain (Provider): Provides a Chain instance for finding similar questions.
        output_parser (Provider): Provides an output parser.
        token_counter (Provider): Provides a token counter utility.
        logger (Provider): Provides a logger configured for console output.
        vector_indices (Chroma): Chroma vector indices initialized based on configuration.
        redis_db (Provider): Provides a Redis database connection.
        debug_info (bool): Indicates whether debugging is enabled.
        system_state_id (str | None): System state id
        question_id (str | None): Question id
        logging_bq_executor (Provider): Provides Thread Pool Executor
        logging_bq_client (Provider): Provides BigQuery client


    Usage:
        Components from the container can be accessed as attributes and are instantiated as needed with configurations
        derived from LLM_YAML_FILE.
    """

    config = common.load_yaml(LLM_YAML_FILE)
    llm = common.get_or_create_model(config["model_name"])
    scoring_llm = common.get_or_create_model(config["scoring_model_name"])

    prompt_manager = providers.Singleton(PromptManager)

    _input_variables_react = [
        "question",
        "previous_conversation",
        "context",
        "previous_rounds",
        "round_number",
        "final_round_statement",
    ]
    react_chain = providers.Factory(provide_chain, "react_chain_prompt", _input_variables_react, "text")
    json_corrector_chain = providers.Factory(provide_chain, "json_corrector_prompt", ["json"], "text")
    aspect_based_summary_chain = providers.Factory(
        provide_chain, "aspect_based_summary_prompt", ["retrieved_doc", "question"], "text"
    )
    answer_scoring_chain = providers.Factory(
        provide_chain, "answer_scoring_prompt", ["question", "answer"], "text", scoring_llm
    )
    retriever_scoring_chain = providers.Factory(
        provide_chain, "retriever_scoring_prompt", ["retrieved_doc", "question"], "text", scoring_llm
    )
    similar_questions_chain = providers.Factory(
        provide_chain, "similar_questions_prompt", ["question", "similar_questions_number"], "similar_questions"
    )

    enhance_question_chain = providers.Factory(
        provide_chain, "enhanced_prompt", ["question", "member_context"], "text"
    )

    string_matcher_chain = providers.Factory(
        provide_chain, "substring_matching_prompt", ["left_string", "right_string"], "text", scoring_llm
    )

    golden_answer_scoring_chain = providers.Factory(
        provide_chain,
        "golden_answer_scoring_prompt",
        ["question", "actual_answer", "expected_answer"],
        "text",
        scoring_llm,
    )

    previous_conversation_relevancy_chain = providers.Factory(
        provide_chain,
        "previous_conversation_scoring_prompt",
        ["previous_question", "previous_answer", "previous_additional_information_to_retrieve", "question"],
        "text",
    )

    token_counter = providers.Singleton(common.provide_token_counter)

    logger = providers.Singleton(provide_logger)

    vector_indices = provide_vector_indices()

    debug_info = config.get("debug_info", False)
    redis_db = providers.Singleton(provide_redis)
    comments = "None"
    system_state_id = None
    question_id = None
    logging_bq_executor = providers.Singleton(ThreadPoolExecutor, max_workers=1)
    logging_bq_client = providers.Singleton(create_bq_client, config.get("bq_project_id"))
