"""Module for managing vectorization, vector storage, and retrieval strategies.

This module provides classes and functions for:

* **VectorStore:** Abstract base class defining vector search interfaces.
* **ChromaVectorStore:** Concrete implementation using Chroma for vector storage and search.
* **VertexVectorStore:**  Concrete implementation leveraging Vertex AI for vector storage and search.
* **VectorStrategy:** Abstract base class for vector embedding and index creation logic.
* **VectorStrategyProvider:** Provides a factory for selecting appropriate VectorStrategy.
* **ChromaVectorStrategy:**  Concrete implementation of VectorStrategy for Chroma.
* **VertexAIVectorStrategy:** Concrete implementation of VectorStrategy for Vertex AI.
"""

import glob
import os
import random
import shutil
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from google.cloud import aiplatform, storage
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import Chroma

from gen_ai.common.common import default_extract_data
from gen_ai.common.inverted_index import InvertedIndex
from gen_ai.common.storage import Storage

@dataclass
class DeployedEndpoint:
    """Class for keeping track of Deployed Endpoints."""

    index: str
    index_endpoint_name: str
    deployed_index_endpoint_name: str
    plan: str


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Defines the interface for similarity search and maximum marginal relevance search.
    """

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        """Performs a similarity search on the vector store.

        Args:
            query (str): The query text.
            k (int, optional): The number of results to return. Defaults to 4.
            **kwargs: Additional arguments to be passed to the underlying implementation.

        Returns:
            List[Document]: A list of LangChain Document objects representing the results.
        """
        pass

    @abstractmethod
    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs
    ) -> list[Document]:
        """Performs a maximum marginal relevance (MMR) search on the vector store.

        Args:
            query (str): The query text.
            k (int, optional): The number of results to return. Defaults to 4.
            fetch_k (int, optional):  The number of documents to initially retrieve. Defaults to 20
            lambda_mult (float, optional): Multiplier for controlling MMR tradeoff. Defaults to 0.5.
            **kwargs: Additional arguments to be passed to the underlying implementation.

        Returns:
            List[Document]: A list of LangChain Document objects representing the results.
        """
        pass


class VectorStrategy(ABC):
    """Abstract base class for defining vector embedding and index creation strategies.

    Attributes:
        storage_interface (Storage): An interface for interacting with storage systems.
    """

    def __init__(self, storage_interface: Storage):
        self.storage_interface = storage_interface

    @abstractmethod
    def get_vector_indices(
        self, regenerate: bool, embeddings: Embeddings, vector_indices: dict[str, str], processed_files_dir: str
    ):
        """Retrieves or creates vector indices based on the provided configuration.

        Args:
            regenerate (bool): If True, forces index regeneration.
            embeddings (Embeddings): The embedding model to use for generating vector representations.
            vector_indices (dict[str, str]): Existing vector indices (if any).
            processed_files_dir (str): directory where files are stored

        Returns:
            dict[str, str]: A dictionary mapping plan names to their corresponding vector indices.
        """
        raise NotImplementedError("Cannot be invoked directly from abstract class")


class ChromaVectorStore(VectorStore):
    """Concrete implementation of VectorStore using Chroma for vector storage and search.

    Attributes:
        chroma (Chroma): The Chroma instance for managing the vector store.
    """

    def __init__(self, chroma):
        self.chroma = chroma

    def similarity_search(self, query: str, k: int = 4, **kwargs):
        return self.chroma.similarity_search(query, k, **kwargs)

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs
    ):
        return self.chroma.max_marginal_relevance_search(query, k, fetch_k, lambda_mult, **kwargs)


class VertexVectorStore(VectorStore):
    """Concrete implementation of VectorStore using Vertex AI for vector storage and search.

    Attributes:
        vertex (aiplatform.MatchingEngineIndexEndpoint): Vertex AI endpoint instance.
        index_id (str): The ID of the Vertex AI index.
        embeddings (Embeddings): The embedding model used for generating vectors.
        doc_mapping (dict):  Mapping between document IDs and their textual content/metadata.
    """

    def __init__(self, vertex, index_id, embeddings, doc_mapping) -> None:
        self.vertex = vertex
        self.index_id = index_id
        self.embeddings = embeddings
        self.doc_mapping = doc_mapping

    def convert_to_langchain(self, neighbours):
        docs = []

        for match in neighbours[0]:
            if match.id in self.doc_mapping:
                text = self.doc_mapping[match.id][0]
                metadata = self.doc_mapping[match.id][1]
                doc = Document(page_content=text, metadata=metadata)
                docs.append(doc)
        return docs

    def similarity_search(self, query: str, k: int = 4, **kwargs):
        embs = self.embeddings.embed_documents([query])
        neighbours = self.vertex.find_neighbors(num_neighbors=k, queries=embs, deployed_index_id=self.index_id)
        docs = self.convert_to_langchain(neighbours)
        return docs

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs
    ):
        # vertex AI does not support max marginal search, so we just use similarity search under the hood
        return self.similarity_search(query, k, **kwargs)


class VectorStrategyProvider:
    """Factory class for selecting the appropriate VectorStrategy implementation."""

    def __init__(self, vector_name):
        self.vector_name = vector_name

    def __call__(self, **kwargs) -> VectorStrategy:
        """Provides the vector strategy.
        Vector strategy consists of two functionalities: vector embeddings and vector indices creation

        Raises:
            ValueError: currently supports openai and vertexai strategies. Raises exception if other types are
            specified

        Returns:
            Embeddings: langchain class of base embeddings
        """
        print(f"Loading {self.vector_name} Vector Strategy...")
        if "vertexai" in self.vector_name:
            return VertexAIVectorStrategy(**kwargs)
        elif "chroma" in self.vector_name:
            return ChromaVectorStrategy(**kwargs)
        else:
            raise ValueError("Not supported embeddings name in config")


class ChromaVectorStrategy(VectorStrategy):
    """Concrete implementation of VectorStrategy for using Chroma as the vector store."""

    def __init__(self, storage_interface: Storage, vectore_store_path: str) -> None:
        super().__init__(storage_interface)
        self.vectore_store_path = f"{vectore_store_path}_chroma"

    def get_vector_indices(
            self, regenerate: bool, embeddings: Embeddings, vector_indices: dict[str, str], processed_files_dir: str
    ):
        if not os.path.exists(self.vectore_store_path) or regenerate:
            if os.path.exists(self.vectore_store_path):
                try:
                    os.rmdir(self.vectore_store_path)
                except OSError as _:
                    os.removedirs(self.vectore_store_path)

            docs = self.storage_interface.process_directory(processed_files_dir, default_extract_data)
            plan_store = Chroma.from_documents(docs, embeddings, persist_directory=self.vectore_store_path)
            plan_store.persist()
            vector_indices = plan_store
        else:
            plan_store = Chroma(persist_directory=self.vectore_store_path, embedding_function=embeddings)
            vector_indices = plan_store

        return vector_indices


class VertexAIVectorStrategy(VectorStrategy):
    """Concrete implementation of VectorStrategy for using Vertex AI as the vector store."""

    DEPLOYED_INDEX_ID = "article_index_endpoint_deployed"
    ARTICLE_INDEX = "article_index"
    ARTICLE_INDEX_ENDPOINT = "article_index_endpoint"

    def __init__(self, storage_interface: Storage, vectore_store_path: str) -> None:
        super().__init__(storage_interface)
        self.vectore_store_path = f"{vectore_store_path}_vertexai"

    def __create(self, embeddings: Embeddings, processed_files_dir: str):
        if not os.path.exists(self.vectore_store_path):
            print("Creating the directory...")
            os.makedirs(self.vectore_store_path)
        else:
            print("Removing & creating the directory...")
            shutil.rmtree(self.vectore_store_path, ignore_errors=True)
            os.makedirs(self.vectore_store_path)

        docs = self.storage_interface.process_directory(processed_files_dir, default_extract_data)
        all_jsons = {}
        for plan, documents in docs.items():
            store_path = os.path.join(self.vectore_store_path, plan)
            embs = embeddings.embed_documents([x.page_content for x in documents])
            embs_plan = [(f"{plan}_{i}", x) for i, x in enumerate(embs)]
            embs_df = pd.DataFrame(embs_plan, columns=["id", "embedding"])
            plan_output_jsonl = store_path + "_df.json"
            with open(plan_output_jsonl, "w", encoding="utf-8") as f:
                f.write(embs_df.to_json(orient="records", lines=True, force_ascii=False))
            all_jsons[plan] = plan_output_jsonl

        return all_jsons

    def create_bucket_and_copy(self, json_file, random_string):
        prefix = os.path.splitext(os.path.basename(json_file))[0]
        bucket_path = f"{prefix}-vertex-ai-search-{random_string}"
        client = storage.Client()
        bucket = client.bucket(bucket_path)
        bucket.location = "us-central1"
        bucket.create()

        blob = bucket.blob(os.path.basename(json_file))
        blob.upload_from_filename(json_file)
        return bucket_path

    def __copy(self, all_jsons):
        bucket_json_names = {}
        random_string = "".join(random.choices(string.ascii_lowercase, k=6))

        for plan, json_file in all_jsons.items():
            bucket_path = self.create_bucket_and_copy(json_file, random_string)
            print(f"File {json_file} copied to bucket: {bucket_path}")
            bucket_json_names[(json_file, plan)] = bucket_path

        return bucket_json_names

    def __deploy(self, bucket_json_names):
        deployed_endpoints = []
        random_string = "".join(random.choices(string.ascii_lowercase, k=4))
        for key, bucket_path in bucket_json_names.items():
            json_file, plan = key
            prefix = os.path.splitext(os.path.basename(json_file))[0]
            file_index_name = f"{prefix}_{self.ARTICLE_INDEX}_{random_string}"
            file_index_endpoint_name = f"{prefix}_{self.ARTICLE_INDEX_ENDPOINT}_{random_string}"
            file_index_deployed_name = f"{prefix}_{self.DEPLOYED_INDEX_ID}_{random_string}"

            file_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=file_index_name,
                contents_delta_uri=f"gs://{bucket_path}/",
                dimensions=768,
                approximate_neighbors_count=10,
            )
            print(file_index)

            file_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=file_index_endpoint_name, public_endpoint_enabled=True
            )
            print(file_index_endpoint)

            file_index_endpoint.deploy_index(index=file_index, deployed_index_id=file_index_deployed_name)
            print(file_index_deployed_name)
            deployed_endpoints.append(
                DeployedEndpoint(file_index_name, file_index_endpoint.resource_name, file_index_deployed_name, plan)
            )

        endpoints_dir = f"{self.vectore_store_path}/endpoints"
        if not os.path.exists(endpoints_dir):
            os.makedirs(endpoints_dir)
        for endpoint in deployed_endpoints:
            endpoint_index = endpoint.index.replace("/", "_")
            with open(f"{endpoints_dir}/{endpoint_index}.txt", "w+", encoding="utf-8") as f:
                f.write(
                    "\n".join(
                        [
                            endpoint.index,
                            endpoint.index_endpoint_name,
                            endpoint.deployed_index_endpoint_name,
                            endpoint.plan,
                        ]
                    )
                )

        return deployed_endpoints

    def get_endpoints(self):
        deployed_endpoints = []
        endpoints_dir = f"{self.vectore_store_path}/endpoints"
        endpoint_files = glob.glob(endpoints_dir + "/*.txt")

        for endpoint_file in endpoint_files:
            with open(endpoint_file, "r+", encoding="utf-8") as f:
                endpoint_lines = f.readlines()
                endpoint_lines = [x.strip().replace("\n", "") for x in endpoint_lines]
                deployed_endpoint = DeployedEndpoint(
                    endpoint_lines[0], endpoint_lines[1], endpoint_lines[2], endpoint_lines[3]
                )
                deployed_endpoints.append(deployed_endpoint)
        return deployed_endpoints

    def get_vector_indices(
            self, regenerate: bool, embeddings: Embeddings, vector_indices: dict[str, str], processed_files_dir: str
            ):
        aiplatform.init()
        if not os.path.exists(self.vectore_store_path):
            all_jsons = self.__create(embeddings, processed_files_dir)
            bucket_json_names = self.__copy(all_jsons)
            deployed_endpoints = self.__deploy(bucket_json_names)
        else:
            deployed_endpoints = self.get_endpoints()

        docs = self.storage_interface.process_directory(processed_files_dir, default_extract_data)
        doc_mapping = InvertedIndex().build_map(docs)

        for deployed_endpoint in deployed_endpoints:
            real_endpoint_object = aiplatform.MatchingEngineIndexEndpoint(deployed_endpoint.index_endpoint_name)
            vector_store = VertexVectorStore(
                real_endpoint_object, deployed_endpoint.deployed_index_endpoint_name, embeddings, doc_mapping
            )
            vector_indices[deployed_endpoint.plan] = vector_store
        return vector_indices
