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
from logging import Logger

import redis
from dependency_injector import containers, providers
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores.chroma import Chroma

import gen_ai.common.common as common
from gen_ai.common.embeddings_provider import EmbeddingsProvider
from gen_ai.common.exponential_retry import LLMExponentialRetryWrapper
from gen_ai.common.storage import UhgStorage
from gen_ai.common.vector_provider import VectorStrategy, VectorStrategyProvider
from gen_ai.constants import LLM_YAML_FILE, MEMORY_STORE_HOST


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
    template = Container.config[template_name].strip()
    answer_template = PromptTemplate(input_variables=input_variables, template=template)
    chain = LLMChain(llm=llm, prompt=answer_template, output_key=output_key, verbose=False, llm_kwargs={"response_mime_type":"application/json"})
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

    embeddings_provider = EmbeddingsProvider(embeddings_name, embeddings_model_name)
    embeddings: Embeddings = embeddings_provider()

    vector_strategy_provider = VectorStrategyProvider(vector_name)
    vector_strategy: VectorStrategy = vector_strategy_provider(storage_interface=UhgStorage())

    local_vector_indices = {}
    return vector_strategy.get_vector_indices(regenerate, embeddings, local_vector_indices)


def provide_logger() -> Logger:
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stdout_handler],
        format="%(asctime)s: %(levelname)s: %(message)s",
    )
    return logging.getLogger()


def provide_redis() -> redis.Redis:
    """
    Provides a Redis database connection using predefined settings.

    This function initializes and returns a connection to a Redis database specified by the MEMORY_STORE_HOST constant.
    It sets up the connection with the default port and database index.

    Returns:
        redis.Redis: A Redis client instance connected to the specified Redis server.
    """
    redis_db = redis.Redis(host=MEMORY_STORE_HOST, port=6379, db=0, decode_responses=True)
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

    Usage:
        Components from the container can be accessed as attributes and are instantiated as needed with configurations
        derived from LLM_YAML_FILE.
    """

    config = common.load_yaml(LLM_YAML_FILE)
    llm = common.get_or_create_model(config["model_name"])
    scoring_llm = common.get_or_create_model(config["scoring_model_name"])

    _input_variables_react = [
        "question",
        "context",
        "previous_rounds",
        "round_number",
        "final_round_statement",
    ]
    react_chain = providers.Singleton(provide_chain, "simple_react_chain_prompt", _input_variables_react, "text")
    json_corrector_chain = providers.Singleton(provide_chain, "json_corrector_prompt", ["json"], "text")
    aspect_based_summary_chain = providers.Singleton(
        provide_chain, "aspect_based_summary_prompt", ["retrieved_doc", "question"], "text"
    )
    answer_scoring_chain = providers.Singleton(
        provide_chain, "answer_scoring_prompt", ["question", "answer"], "text", scoring_llm
    )
    retriever_scoring_chain = providers.Singleton(
        provide_chain, "retriever_scoring_prompt", ["retrieved_doc", "question"], "text", scoring_llm
    )
    similar_questions_chain = providers.Singleton(
        provide_chain, "similar_questions_prompt", ["question", "similar_questions_number"], "similar_questions"
    )

    token_counter = providers.Singleton(common.provide_token_counter)

    logger = providers.Singleton(provide_logger)

    vector_indices = provide_vector_indices()

    debug_info = config.get("debug_info", False)
    redis_db = providers.Singleton(provide_redis)
    comments = "None"
    system_state_id = None
    question_id = None
