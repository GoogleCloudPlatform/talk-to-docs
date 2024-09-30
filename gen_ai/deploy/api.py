"""
This module defines the FastAPI endpoints for the Gen AI project, handling 
requests related to question answering, feedback, and conversational state resets.
"""

import os
import posixpath
import requests

from fastapi import FastAPI, File, UploadFile, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List

import google.auth

from gen_ai.common.de_tools import (
    create_metadata_jsonl,
    create_layout_search_datastore,
    create_search_engine,
    import_datastore_documents,
    list_datastore_documents,
    get_operation_status,
    purge_datastore_documents,
    flush_redis_cache,
)
from gen_ai.common.ioc_container import Container
from gen_ai.deploy.model import (
    ItemInput,
    LLMOutput,
    ResetInput,
    ResetOutput,
    ResponseInput,
    ResponseOutput,
    VAISConfig,
    ListDocumentsRequest,
    ListDocumentsResponse,
    CreateProjectInput,
    RemoveDocumentsRequest,
    RemoveDocumentsResponse,
    ViewExtractedDocumentResponse,
    IndexDocumentsResponse,
)
from gen_ai.llm import respond_api
from gen_ai.extraction_pipeline.vais_import_tools import VaisImportTools
from gen_ai.common.bq_utils import (
    bq_create_project,
    bq_project_details,
    bq_change_prompt,
    bq_debug_response,
    bq_all_projects,
)
from starlette.responses import JSONResponse


# Get ADC creds and project ID.
_, PROJECT = google.auth.default()
Container.logger().info(f"ADC Project ID: {PROJECT}")

app = FastAPI()

items_db = {}


@app.post("/documents")
async def get_list_documents(view_documents_request: ListDocumentsRequest) -> ListDocumentsResponse:
    vait = VaisImportTools(Container.config)
    return ListDocumentsResponse(
        user_id=view_documents_request.user_id,
        project_name=view_documents_request.project_name,
        documents=vait.list_documents(view_documents_request),
    )


@app.post("/index_files")
async def upload_files(user_id: str, project_name: str, files: list[UploadFile] = File(...)) -> IndexDocumentsResponse:
    vait = VaisImportTools(Container.config)
    return vait.processor(user_id, project_name, files)


@app.delete("/document")
async def remove_documents(remove_documents_request: RemoveDocumentsRequest) -> RemoveDocumentsResponse:
    vait = VaisImportTools(Container.config)
    return vait.remove_multiple_documents(remove_documents_request)


@app.get("/document/{document_id}")
async def view_document(document_id: str) -> ViewExtractedDocumentResponse:
    vait = VaisImportTools(Container.config)
    return vait.get_parsed_document(document_id)


# TODO: output class?
@app.get("/import_status/{lro_id}")
async def check_import_status(lro_id: str):
    vait = VaisImportTools(Container.config)
    return vait.get_import_status(lro_id)


@app.post("/create_project/")
async def create_project(project_name: str = Form(...), user_id: str = Form(...), files: List[UploadFile] = File(...)):
    project_id = bq_create_project(project_name, user_id)

    # uncomment when Uploading works
    vait = VaisImportTools(Container.config)
    process_files = vait.processor(user_id, project_name, files)
    if not process_files:
        return JSONResponse(
            status_code=500,
            content={"detail": "Files were not processed properly", "code": 500},
        )

    return {"project_id": project_id}


@app.post("/project_details/")
async def project_details(project_id: str = Form(...), user_id: str = Form(...)):
    project_details = bq_project_details(project_id, user_id)

    return project_details


@app.post("/all_projects/")
async def all_projects(user_id: str = Form(...)):
    project_details = bq_all_projects(user_id)

    return project_details


@app.post("/change_prompt/")
async def change_prompt(
    project_id: str = Form(...), user_id: str = Form(...), prompt_name: str = Form(...), prompt_value: str = Form(...)
):
    change_prompt = bq_change_prompt(project_id, user_id, prompt_name, prompt_value)

    return change_prompt


@app.post("/debug_response/")
async def debug_response(prediction_id: str = Form(...)):
    debug_info = bq_debug_response(prediction_id)

    return debug_info


@app.post("/respond/", response_model=LLMOutput)
def respond(query: ItemInput) -> LLMOutput:
    """
    Handles a POST request to the '/respond/' endpoint, processing the given query
    through a conversational model and generating a structured response.

    This function takes a query input, sends it to the respond_api for processing,
    and then formats the response into a structured LLMOutput object. It is part of
    a FastAPI application and is designed to work with asynchronous conversation models
    or chatbots.

    Args:
        query (ItemInput): An object containing the user's query and context for the
                           conversation. This object must have two attributes: `question`,
                           which is a string representing the user's query, and `member_context_full`,
                           which provides additional context necessary for generating a response.

    Returns:
        LLMOutput: An object containing the structured response from the conversational model.
                   This includes the round number, the latest answer generated by the model,
                   and placeholders for additional fields such as `response_id`,
                   `plan_and_summaries`, `additional_information_to_retrieve`, `context_used`,
                   `urls_to_kc`, and `sections_to_b360`. Most of these fields are initialized
                   with empty strings or arrays as placeholders.

    Raises:
        This function itself does not explicitly raise any exceptions but relies on the
        `respond_api` function's behavior. If `respond_api` encounters issues or if the input
        does not meet expected formats, the caller should handle potential exceptions
        according to the implementation of `respond_api`.

    Note:
        - The `LLMOutput` and `ItemInput` types should be properly defined elsewhere in the codebase,
          including their fields and types.
        - The actual implementation of `respond_api` is not shown here and must be implemented
          or imported from another module. This function's behavior, especially error handling,
          will depend on the `respond_api` implementation.
    """
    conversation = respond_api(query.question, query.member_context_full)
    response = LLMOutput(
        round_number=str(conversation.round_numder),
        answer=conversation.exchanges[-1].answer,
        response_id="",
        plan_and_summaries="",
        additional_information_to_retrieve="",
        context_used=conversation.exchanges[-1].relevant_context,
        urls_to_kc=conversation.exchanges[-1].urls,
        attributes_to_b360=conversation.exchanges[-1].custom_fields.get("attributes_to_b360", []),
        attributes_to_kc_km=conversation.exchanges[-1].custom_fields.get("attributes_to_kc_km", []),
        attributes_to_kc_mp=conversation.exchanges[-1].custom_fields.get("attributes_to_kc_mp", []),
        confidence_score=str(conversation.exchanges[-1].confidence_score),
        session_id=conversation.session_id,
    )

    return response


@app.post("/feedback/", response_model=ResponseOutput)
def feedback(query: ResponseInput) -> ResponseOutput:
    """Submits feedback using the object ResponseInput
    Returns: ResponseOutput
    """
    _ = query
    response = ResponseOutput(success=True, response_id="123")

    return response


@app.post("/reset/", response_model=ResetOutput)
def reset(query: ResetInput) -> ResponseOutput:
    """Resets the state using the object ResetInput
    Returns: ResetOutput
    """
    _ = query
    response = ResetOutput(success=True)

    return response


@app.get("/health")
def health_check() -> dict:
    """Provides a health pulse for Cloud Deployment"""
    return {"status": "ok"}


# Discovery engine and redis routes.
@app.post("/create-metadata")
def create_metadata(request: VAISConfig) -> dict:
    """Create a metadata JSONL file in a GCS bucket."""
    return create_metadata_jsonl(
        project=PROJECT,
        bucket_name=request.bucket_name,
        source_folder=request.source_folder,
        metadata_folder=request.metadata_folder,
        dataset_name=request.dataset_name,
        metadata_filename=request.metadata_filename,
    )


@app.post("/create-layout-datastore")
def create_layout_datastore(request: VAISConfig) -> dict:
    """Create a data store using the layout parser."""
    return create_layout_search_datastore(
        project=PROJECT,
        location=request.location,
        collection=request.collection,
        data_store_id=request.data_store_id,
    )


@app.post("/create-search-engine")
def create_engine(request: VAISConfig) -> dict:
    """Create a search engine."""
    return create_search_engine(
        project=PROJECT,
        location=request.location,
        collection=request.collection,
        engine_id=request.engine_id,
        data_store_id=request.data_store_id,
        company_name=request.company_name,
    )


@app.post("/import-documents")
def import_docs(request: VAISConfig) -> dict:
    """Import documents to the data store."""
    # Compose the path and URI of the metadata JSONL file.
    metadata_path = posixpath.join(
        request.bucket_name,
        request.metadata_folder,
        request.dataset_name,
        request.metadata_filename,
    )
    metadata_uri = f"gs://{metadata_path}"

    return import_datastore_documents(
        project=PROJECT,
        location=request.location,
        data_store=request.data_store_id,
        branch=request.branch,
        metadata_uri=metadata_uri,
    )


@app.get("/get-operation")
def operation_status(location: str, operation_name: str) -> dict:
    """Check the status of an Operation."""
    return get_operation_status(
        location=location,
        operation_name=operation_name,
    )


@app.post("/purge-documents")
def purge_docs(request: VAISConfig) -> dict:
    """Purge data store documents."""
    return purge_datastore_documents(
        project=PROJECT,
        location=request.location,
        data_store=request.data_store_id,
        branch=request.branch,
        force=True,
    )


@app.get("/list-documents")
def list_docs(location: str, data_store_id: str, branch: str) -> dict:
    """List documents in the data store."""
    return list_datastore_documents(
        project=PROJECT,
        location=location,
        data_store=data_store_id,
        branch=branch,
    )


@app.get("/count-documents")
def count_docs(location: str, data_store_id: str, branch: str) -> dict:
    """Return only the document count in the data store."""
    list_response = list_datastore_documents(
        project=PROJECT,
        location=location,
        data_store=data_store_id,
        branch=branch,
    )
    return {"total_documents": list_response["total_documents"]}


@app.post("/flush-redis")
def flush_redis() -> dict:
    """Flush the redis cache."""
    return flush_redis_cache()


@app.get("/get-env-variable")
def get_env_variables(name: str) -> dict:
    """Return the value of an environment variable.
    Ref: https://cloud.google.com/run/docs/testing/local#cloud-code-emulator_1
    """
    return {name: os.environ.get(name, f"No variable set for '{name}'")}


@app.get("/get-instance-id")
def get_instance_id() -> dict:
    """Return the Cloud Run instance ID."""
    response = requests.get(
        url="http://metadata.google.internal/computeMetadata/v1/instance/id",
        headers={"Metadata-Flavor": "Google"},
        timeout=120,
    )

    if response.status_code == 200:
        instance_id = response.text
        return {"instance_id": instance_id}
    else:
        return (
            "Failed to retrieve instance id",
            404,
        )
