"""
This module provides tools for interacting with Google BigQuery, including functions for creating clients,
datasets, and tables, as well as loading data. It leverages Google Cloud BigQuery to manage large-scale data
and analytics. The module contains utility functions to facilitate the creation and management of BigQuery
resources such as datasets and tables, and it provides a method to directly load data from a pandas DataFrame
into BigQuery, handling schema and client initialization. Additionally, it includes a specialized class
for converting structured data related to query states into a format suitable for analytics in BigQuery.

Classes:
    BigQueryConverter - Converts query state data into a pandas DataFrame for upload to BigQuery.

Functions:
    create_bq_client(project_id)
    create_dataset(client, dataset_id, location, recreate_dataset)
    create_table(client, table_id, schema, recreate_table)
    load_data_to_bq(client, table_id, schema, df)

Exceptions:
    GoogleAPIError - Handles API errors that may occur during interaction with Google services.
"""

import datetime
import getpass
import json
import os
import re
import uuid
from typing import Any

import git
import google.auth
import pandas as pd
from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import bigquery
from google.cloud.bigquery.schema import SchemaField

from gen_ai.common.document_utils import convert_dict_to_relevancies, convert_dict_to_summaries
from gen_ai.common.ioc_container import Container
from gen_ai.deploy.model import Conversation, QueryState

medical_vertical_id = "20e264fd-7c30-4d99-8292-f02f5e92461b"  # hardcoded for October demo, we show just 1 vertical now


def create_dataset(
    client: bigquery.Client, dataset_id: str, location: str = "US", recreate_dataset: bool = False
) -> None:
    """Creates a BigQuery dataset.
    If the dataset already exists, it will be deleted and recreated if recreate_dataset is True.
    Otherwise, an error will be raised.
    Args:
        client (bigquery.Client): The BigQuery client.
        dataset_id (str): The ID of the dataset to create.
        location (str, optional): The location of the dataset. Defaults to "US".
        recreate_dataset (bool, optional): Whether to recreate the dataset if it already exists. Defaults to False.
    """
    if recreate_dataset:
        client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=True)
        print(f"Dataset {dataset_id} and its contents have been deleted.")
    try:
        client.get_dataset(dataset_id)
        print(f"Dataset {client.project}.{dataset_id} already exists")
    except NotFound:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = location
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"Created dataset {client.project}.{dataset.dataset_id}")


def create_table(
    client: bigquery.Client, table_id: str, schema: list[SchemaField], recreate_table: bool = False
) -> None:
    """Creates a BigQuery table.
    If the table already exists, it will be deleted and recreated if recreate_table is True.
    Otherwise, an error will be raised.
    Args:
        client (bigquery.Client): The BigQuery client.
        table_id (str): The ID of the table to create.
        schema (List[bigquery.SchemaField]): The schema of the table.
        recreate_table (bool, optional): Whether to recreate the table if it already exists. Defaults to False.
    """
    if recreate_table:
        try:
            client.get_table(table_id)
            client.delete_table(table_id)
            print(f"Table {table_id} deleted.")
        except NotFound:
            print(f"Table {table_id} does not exist. Skipping deletion.")

    table = bigquery.Table(table_id, schema=schema)
    try:
        client.get_table(table_id)
        print(f"Table {table_id} already exists.")
    except NotFound:
        table = client.create_table(table)
        print(f"Table {table_id} created.")


def load_data_to_bq(conversation: Conversation, log_snapshots: list[dict[str, Any]], client_project_id):
    """Loads prediction data, question and exeriments information to BigQuery.

    This function prepares and loads relevant data from a conversation into BigQuery.
    The process involves extracting the latest question from the conversation,
    logging it for reference, and transforming the data into a format
    suitable for BigQuery using the `BigQueryConverter`.

    Args:
        conversation: A Conversation object containing the full conversation history.
        log_snapshots: A list of log snapshot objects containing relevant metadata.
    """
    query_state = conversation.exchanges[-1]
    question = query_state.question
    log_question(question)
    df = BigQueryConverter.convert_query_state_to_prediction(
        conversation.exchanges[-1], log_snapshots, conversation.session_id, client_project_id
    )
    load_status = load_prediction_data_to_bq(df)
    if load_status:
        Container.logger().info(msg="Successfully wrote into BQ Prediction table")
    else:
        Container.logger().info(msg="Error in writing into BQ Prediction table")


def load_prediction_data_to_bq(df: pd.DataFrame) -> None:
    """Loads data from a pandas DataFrame to a BigQuery table.
    The table will be created if it does not already exist.
    If the table already exists, it will be overwritten.
    Args:
        df (pandas.DataFrame): The DataFrame to load data from.
    """
    client = Container.logging_bq_client()
    dataset_id = get_dataset_id()

    table_id = f"{dataset_id}.prediction"
    table = client.get_table(table_id)
    schema = table.schema

    job_config = bigquery.LoadJobConfig(schema=schema)
    job = None
    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        print(f"Loaded {job.output_rows} rows into {table_id}.")

    except GoogleAPIError as e:
        Container.logger().error(msg="Crashed on writing into BQ Prediction table")
        Container.logger().error(msg=str(e))
        if job and job.errors:
            for error in job.errors:
                print(f"Error: {error['message']}")
                if "location" in error:
                    print(f"Field that caused the error: {error['location']}")
        return False
    return True


def log_system_status(session_id: str) -> str:
    """
    Logs the current system status and pipeline parameters to a BigQuery table for tracking and reproducibility.

    This function gathers essential information about the current execution context, including Git commit hash,
    GCS bucket location, model configuration, and optional user comments.
    It then generates a unique system state ID and inserts this data into an 'experiment' BigQuery table.

    Args:
        session_id (str): A unique identifier for the current user session.

    Returns:
        str: The generated system state ID.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        git_hash = str(repo.head.object.hexsha)
    except git.exc.InvalidGitRepositoryError:
        print("Error: git repo not found.")
        git_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, os.getcwd()))

    gcs_bucket = Container.config["gcs_source_bucket"]
    model_name = Container.config["model_name"]
    temperature = Container.config["temperature"]
    max_output_tokens = Container.config.get("max_output_tokens", 4000)
    pipeline_parameters = f"model: {model_name}; temperature: {temperature}; max_tokens: {max_output_tokens}"

    comments = Container.comments
    system_state_id = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"{git_hash}-{gcs_bucket}-{pipeline_parameters}-{comments or ''}")
    )

    data = {
        "system_state_id": system_state_id,
        "session_id": session_id,
        "github_hash": git_hash,
        "gcs_bucket_path": gcs_bucket,
        "pipeline_parameters": pipeline_parameters,
        "comments": comments,
    }
    data = {str(x): str(v) for x, v in data.items()}
    insert_status = insert_data_to_table("experiment", data)
    if not insert_status:
        print(f"Error while logging system state id to bq table. Github hash: {git_hash}; GCS bucket: {gcs_bucket}")
    Container.system_state_id = system_state_id
    return system_state_id


def bq_add_lro_entry(user_id: str, client_project_id: str, lro_id: str) -> bool:
    lro_data = {"user_id": user_id, "client_project_id": client_project_id, "lro_id": lro_id, "status": "INPROGRESS"}

    insert_status = insert_data_to_table("lros", lro_data)
    if not insert_status:
        print(f"Error while logging lro {lro_id} to bq table.")
    return insert_status


def bq_get_lro_entries(user_id: str, client_project_id: str) -> list[str]:
    dataset_id = get_dataset_id()

    query = f"""
    SELECT *
    FROM `{dataset_id}.lros`
    WHERE user_id='{user_id}' 
            AND client_project_id='{client_project_id}' 
            AND status='INPROGRESS'
    """
    client = Container.logging_bq_client()
    query_job = client.query(query)

    results = query_job.result()
    lro_data = [row.lro_id for row in results]

    return lro_data


def bq_get_previous_chat(user_id: str, client_project_id: str):
    dataset_id = get_dataset_id()

    query = f"""
    SELECT pred.response_id, pred.response, proj.project_name
    FROM (SELECT response_id, response, client_project_id 
            FROM `{dataset_id}.prediction`
            WHERE client_project_id='{client_project_id}'
            ORDER BY timestamp
    ) as pred
    LEFT JOIN `{dataset_id}.projects` as proj
    ON pred.client_project_id=proj.project_id    
    """
    client = Container.logging_bq_client()
    query_job = client.query(query)
    results = query_job.result()

    chat_list = []
    project_name = None
    for row in results:
        project_name = row.project_name
        chat_list.append(
            {
                "is_ai": True,
                "message":row.response, 
                "response_id":row.response_id,
            }
        )

    response = {
        "project_name": project_name,
        "chat_list": chat_list
    }

    return response



def bq_create_project(project_name: str, user_id: str):
    project_id = str(uuid.uuid4())

    timestamp = datetime.datetime.now().isoformat()

    project_data = {
        "project_id": project_id,
        "project_name": project_name,
        "created_on": timestamp,
        "updated_on": timestamp,
        "vertical_id": medical_vertical_id,
    }

    insert_status_project = insert_data_to_table("projects", project_data)
    if not insert_status_project:
        print(f"Error while inserting project {project_name} to projects table.")
        return None

    project_user_data = {"id": str(uuid.uuid4()), "project_id": project_id, "user_id": user_id}

    insert_status_project_user = insert_data_to_table("project_user", project_user_data)
    if not insert_status_project_user:
        print(f"Error while inserting project-user relationship for project {project_name}.")
        return None

    return project_id


def bq_project_details(project_id: str, user_id: str):
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
        AND pu.user_id = '{user_id}'
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
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )

    client = Container.logging_bq_client()
    query_job = client.query(query, job_config=job_config)

    results = list(query_job.result())

    project_details = None
    for row in results:
        project_details = {
            "project_name": row.project_name,
            "created_on": row.created_on,
            "updated_on": row.updated_on,
            "prompt_configuration": [
                {
                    "prompt_name": prompt["prompt_name"],
                    "prompt_value": prompt["prompt_value"],
                    "prompt_display_name": prompt["prompt_display_name"]
                }
                for prompt in row.prompt_configuration
            ],
        }

    return project_details


def bq_all_projects(user_id: str):
    dataset_id = get_dataset_id()

    query = f"""
    WITH ProjectDetails AS (
        -- Fetch project details from the projects table
        SELECT 
            p.project_id,
            p.project_name,
            p.created_on,
            p.updated_on
        FROM `{dataset_id}.projects` p
        JOIN `{dataset_id}.project_user` pu 
            ON p.project_id = pu.project_id
        WHERE pu.user_id = @user_id
        order by p.updated_on desc
    )
    SELECT 
        project_id, 
        project_name, 
        created_on, 
        updated_on
    FROM ProjectDetails
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )

    client = bigquery.Client()
    query_job = client.query(query, job_config=job_config)

    results = list(query_job.result())

    all_projects = []
    for row in results:
        all_projects.append(
            {
                "project_id": row.project_id,
                "project_name": row.project_name,
                "created_on": row.created_on,
                "updated_on": row.updated_on,
            }
        )

    return {"all_projects": all_projects}


def bq_change_prompt(project_id: str, user_id: str, prompt_name: str, prompt_value: str):
    dataset_id = get_dataset_id()

    query = f"""
    MERGE `{dataset_id}.prompts` AS target
    USING (SELECT @project_id AS project_id, @prompt_name AS prompt_name) AS source
    ON target.project_id = source.project_id AND target.prompt_name = source.prompt_name
    WHEN MATCHED THEN 
        UPDATE SET prompt_value = @prompt_value, created_on = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
        INSERT (id, project_id, vertical_id, prompt_name, prompt_value, created_on)
        VALUES (GENERATE_UUID(), @project_id, (SELECT vertical_id FROM `{dataset_id}.projects` WHERE project_id = @project_id), @prompt_name, @prompt_value, CURRENT_TIMESTAMP())
    """
    client = Container.logging_bq_client()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("prompt_name", "STRING", prompt_name),
            bigquery.ScalarQueryParameter("prompt_value", "STRING", prompt_value),
        ]
    )

    query_job = client.query(query, job_config=job_config)
    try:
        res = list(query_job.result())
    except Exception as e:
        return {"status": False}

    Container.prompt_manager().invalidate_cache(project_id)
    return {"status": True}


def bq_debug_response(response_id: str):
    dataset_id = get_dataset_id()

    query = f"""
    SELECT *
    FROM `{dataset_id}.prediction`
    WHERE response_id = @response_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("response_id", "STRING", response_id)]
    )

    client = Container.logging_bq_client()
    query_job = client.query(query, job_config=job_config)

    results = query_job.result()

    prediction_data = [dict(row) for row in results]

    formatted_data = format_prediction_data(prediction_data)
    return formatted_data


def format_prediction_data(data):
    formatted_output = {}

    previous_context = []
    for i, row in enumerate(data):
        previous_context.append(
            f"- Previous question #{i} was: {row['question']}\n"
            f"- Previous answer #{i} was: {row['response']}\n"
            f"- Previous additional information to retrieve #{i} was: {row.get('additional_question', 'N/A')}"
        )
    formatted_output["Previous Context"] = "\n".join(previous_context)

    formatted_output["Rounds Information"] = []
    for i, row in enumerate(data):
        document_details = []
        round_info = {
            "Round number": i,
            "Plan and Summaries": row["plan_and_summaries"],
            "Answer": row["response"],
            "Additional Info to Retrieve": row.get("additional_question", "N/A"),
            "Confidence Score": row["confidence_score"],
            "Context Used": row["context_used"],
        }

        try:
            doc_i_details = ""
            doc_metadata = eval(row["post_filtered_documents_so_far_all_metadata"])
            for j, x in enumerate(doc_metadata):
                doc_i_details += f"Document #{j} \n"
                doc_i_details += "Page content: "
                doc_i_details += x["page_content"][0:150] + "...\n"
                if "metadata" in x:
                    doc_i_details += "Section name: " + x["metadata"]["section_name"] + "\n"
                    if x["metadata"]["relevancy_reasoning"] != "The text could not be scored":
                        doc_i_details += "Relevancy score: " + x["metadata"]["relevancy_score"] + "\n"
                        doc_i_details += "Relevancy reasoning: " + x["metadata"]["relevancy_reasoning"] + "\n"
                    if x["metadata"]["summary_reasoning"] != "The text could not be summarized":
                        doc_i_details += "Summary score: " + x["metadata"]["summary_score"] + "\n"
                        doc_i_details += "Summary reasoning: " + x["metadata"]["summary_reasoning"] + "\n"
                        doc_i_details += "Summary: " + x["metadata"]["summary"] + "\n"
                doc_i_details += "################## \n"
            document_details.append(doc_i_details)
        except Exception as e:
            print(e)
            continue
        round_info["Retrieved Document Details"] = document_details

        formatted_output["Rounds Information"].append(round_info)

    time_taken = []
    for i, row in enumerate(data):
        time_taken.append(
            f"Round #{i}:\n"
            f"Total time taken: {row['time_taken_total']}s\n"
            f"Retrieval time taken: {row['time_taken_retrieval']}s\n"
            f"LLM time taken: {row['time_taken_llm']}s\n"
        )
    formatted_output["Time Taken"] = "\n".join(time_taken)

    return formatted_output


def log_question(question: str) -> str:
    """
    Logs a question into a BigQuery table and generates a unique question ID.

    This function does the following:

    * **Cleans the Question:** Removes non-alphanumeric characters from the question for ID generation.
    * **Generates Unique ID:** Creates a question ID using a UUID and the cleaned question text, ensuring uniqueness.
    * **Prepares Data:**  Formats the question and generated ID into a data structure for insertion.
    * **Inserts into BigQuery:** Inserts the formatted data into a 'questions' BigQuery table.
    * **Handles Errors:** Logs an error message if the BigQuery insertion fails.

    Args:
        question: The raw text of the question.

    Returns:
        str: The unique question ID.
    """
    question_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, re.sub(r"\W", "", question.lower())))
    data = {
        "question_id": question_id,
        "question": question,
        "parent_question_id": "",
    }

    insert_status = insert_data_to_table("questions", data)
    if not insert_status:
        print(f"Error while logging question {question} to bq table.")

    Container.question_id = question_id
    return question_id


def insert_data_to_table(table_name: str, data: dict[str, str]) -> bool:
    """
    Inserts a single row of data into a specified BigQuery table.

    This function assumes the data dictionary contains only string values.

    Args:
        table_name: The name of the target BigQuery table.
        data: A dictionary containing the data to be inserted, with keys as column names and values as strings.

    Returns:
        bool: True if the insertion was successful, False otherwise.
    """
    client = Container.logging_bq_client()
    dataset_id = get_dataset_id()
    table = client.get_table(f"{dataset_id}.{table_name}")

    errors = client.insert_rows_json(table, [data])
    if not errors:
        print("New rows have been added.")
        return True
    print(f"Errors while inserting rows: {errors}")
    return False


def get_dataset_id() -> str:
    """
    Retrieves the BigQuery dataset ID for the current project.

    The dataset ID combines the project ID and a predefined dataset name
    (assumed to be globally defined as 'DATASET_NAME').

    Priority for determining the project ID:

    1. **Variable in llm.yaml:** Looks for the 'bq_project_id' config variable.
    2. **Google Application Default Credentials:** If the environment variable is not found, uses Google's default
    credentials mechanism.

    Returns:
        str: The fully constructed BigQuery dataset ID in the format 'project_id.DATASET_NAME'.

    Raises:
        ValueError: If the project ID cannot be determined from either source.
    """
    project_id = Container.config.get("bq_project_id")
    dataset_name = Container.config["dataset_name"]
    if not project_id:
        _, project_id = google.auth.default()
    return f"{project_id}.{dataset_name}"


class BigQueryConverter:
    """
    A utility class for converting query state data into a pandas DataFrame that can be uploaded to BigQuery.

    This class is used to convert structured data from various stages of query processing, encapsulating it into
    a DataFrame. The DataFrame format is suitable for analytics and can be directly uploaded to BigQuery for
    further analysis. It handles the extraction of relevant fields from log snapshots associated with each
    query state, transforming them into a tabular form.

    Methods:
        convert_query_state_to_prediction(query_state, log_snapshots) - Converts log snapshots and a query state
                                                                        into a DataFrame structured for BigQuery.

    Usage:
        converter = BigQueryConverter()
        dataframe = converter.convert_query_state_to_prediction(query_state, log_snapshots)
    """

    @staticmethod
    def convert_query_state_to_prediction(
        query_state: QueryState, log_snapshots: list[dict], session_id: str, client_project_id: str
    ) -> pd.DataFrame:
        data = {
            "user_id": [],
            "prediction_id": [],
            "timestamp": [],
            "system_state_id": [],
            "session_id": [],
            "question_id": [],
            "question": [],
            "react_round_number": [],
            "response": [],
            "retrieved_documents_so_far": [],
            "post_filtered_documents_so_far": [],
            "retrieved_documents_so_far_content": [],
            "post_filtered_documents_so_far_content": [],
            "post_filtered_documents_so_far_all_metadata": [],
            "confidence_score": [],
            "response_type": [],
            "run_type": [],
            "time_taken_total": [],
            "time_taken_retrieval": [],
            "time_taken_llm": [],
            "tokens_used": [],
            "summaries": [],
            "relevance_score": [],
            "additional_question": [],
            "plan_and_summaries": [],
            "original_question": [],
            "client_project_id": [],
            "response_id": [],
            "context_used": []
        }
        max_round = len(log_snapshots) - 1
        system_state_id = Container.system_state_id or log_system_status(session_id)
        for round_number, log_snapshot in enumerate(log_snapshots):
            react_round_number = round_number
            response = query_state.answer or ""
            retrieved_documents_so_far = json.dumps(
                [
                    {"original_filepath": x["metadata"].get("original_filepath")}
                    for x in log_snapshot["pre_filtered_docs"]
                ]
            )
            post_filtered_documents_so_far = json.dumps(
                [
                    {"original_filepath": x["metadata"].get("original_filepath")}
                    for x in log_snapshot["post_filtered_docs"]
                ]
            )
            retrieved_documents_so_far_content = json.dumps(
                [{"page_content": x["page_content"]} for x in log_snapshot["pre_filtered_docs"]]
            )
            post_filtered_documents_so_far_content = json.dumps(
                [{"page_content": x["page_content"]} for x in log_snapshot["post_filtered_docs"]]
            )
            post_filtered_documents_so_far_all_metadata = json.dumps([x for x in log_snapshot["post_filtered_docs"]])
            time_taken_total = query_state.time_taken
            time_taken_retrieval = 0
            time_taken_llm = 0
            response_type = "final" if react_round_number == max_round else "intermediate"
            tokens_used = query_state.tokens_used if query_state.tokens_used is not None else 0
            prediction_id = log_snapshot["prediction_id"]
            response_id = log_snapshot["response_id"]
            context_used = str(log_snapshot['context_used'])

            timestamp = datetime.datetime.now()
            confidence_score = query_state.confidence_score
            summary = json.dumps([convert_dict_to_summaries(x) for x in log_snapshot["pre_filtered_docs"]])
            relevance_score = json.dumps([convert_dict_to_relevancies(x) for x in log_snapshot["pre_filtered_docs"]])
            additional_question = log_snapshot["additional_information_to_retrieve"]
            plan_and_summaries = str(log_snapshot["plan_and_summaries"])

            data["user_id"].append(getpass.getuser())
            data["prediction_id"].append(prediction_id)
            data["timestamp"].append(timestamp)
            data["system_state_id"].append(system_state_id)
            data["session_id"].append(session_id)
            data["question_id"].append(Container.question_id)
            data["question"].append(query_state.question)
            data["react_round_number"].append(str(react_round_number))
            data["response"].append(response)
            data["retrieved_documents_so_far"].append(retrieved_documents_so_far)
            data["post_filtered_documents_so_far"].append(post_filtered_documents_so_far)
            data["retrieved_documents_so_far_content"].append(retrieved_documents_so_far_content)
            data["post_filtered_documents_so_far_content"].append(post_filtered_documents_so_far_content)
            data["post_filtered_documents_so_far_all_metadata"].append(post_filtered_documents_so_far_all_metadata)
            data["confidence_score"].append(confidence_score)
            data["response_type"].append(response_type)
            data["run_type"].append("test")
            data["time_taken_total"].append(time_taken_total)
            data["time_taken_retrieval"].append(time_taken_retrieval)
            data["time_taken_llm"].append(time_taken_llm)
            data["tokens_used"].append(tokens_used)
            data["summaries"].append(summary)
            data["relevance_score"].append(relevance_score)
            data["additional_question"].append(additional_question)
            data["plan_and_summaries"].append(plan_and_summaries)
            data["original_question"].append(query_state.original_question)
            data["client_project_id"].append(client_project_id)
            data["response_id"].append(response_id)
            data["context_used"].append(context_used)

        df = pd.DataFrame(data)
        return df
