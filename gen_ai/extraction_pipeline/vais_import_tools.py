import json
import json5
import os
from typing import Any
import uuid

from gen_ai.deploy.model import DocumentsRequest, RemoveDocumentsRequest,RemoveDocumentsResponse, ViewExtractedDocumentResponse, IndexDocumentsResponse
from google.api_core.client_options import ClientOptions
import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import discoveryengine
from google.cloud import storage
import pandas as pd

import markdown


class VaisImportTools:
    def __init__(self, config: dict[str, str]):
        creds, default_project_id = google.auth.default()
        self.auth_session = AuthorizedSession(creds)

        self.bucket_address = config.get("vais_staging_bucket")
        self.project_id = config.get("vais_project_id", default_project_id)
        self.location = config.get("vais_location")
        self.datastore_id = config.get("vais_data_store")
        self.source_dir = config.get("SOURCE_DIR", "source-data")
        self.metadata_dir = config.get("METADATA_DIR", "metadata")


    def processor(self, user_id: str, client_project_id: str, files: Any) -> IndexDocumentsResponse:
        """
        Processes files for a given user and imports them into a datastore.

        This function orchestrates the following steps:
        1. Uploads source files to a Google Cloud Storage bucket.
        2. Creates a metadata JSONL file describing the uploaded files.
        3. Imports the documents into a specified datastore using the metadata file.

        Args:
            user_id: The ID of the user associated with the files.
            client_project_id: Client project id
            files: The files to be processed.

        Returns:
            The result of the document import operation.
        """
        # TODO: proper logs
        stage_response = self.upload_source_files_to_bucket(files, user_id, self.project_id, self.bucket_address, self.source_dir)
        uri_list = stage_response["success"]
        if not uri_list:
            return False

        metadata_filename = self.create_metadata_jsonl(uri_list, user_id, client_project_id, self.project_id, self.bucket_address, self.metadata_dir)
        if not metadata_filename:
            return False
        metadata_uri = f"gs://{self.bucket_address}/{metadata_filename}"

        import_result = self.import_documents_from_gcs_jsonl(
            authed_session=self.auth_session,
            project_id=self.project_id,
            location=self.location,
            datastore_id=self.datastore_id,
            metadata_uri=metadata_uri,
        )
        if import_result:
            return IndexDocumentsResponse(
                status=True,
                message="Index operation successful",
                lro_id=import_result
            )
        #TODO add proper error message
        return IndexDocumentsResponse(
            status=False,
            message="Error uploading files.",
            lro_id=""
        )


    def upload_source_files_to_bucket(
        self,
        files,
        user_id: str,
        project_id: str,
        bucket_address: str,
        source_dir: str
    ) -> dict[str, list[str]]:
        """
        Uploads files to a Google Cloud Storage bucket.

        Args:
            files: A list of files to upload.
            user_id: The ID of the user associated with the files.
            project_id: The ID of the Google Cloud project.
            bucket_address: The address of the Google Cloud Storage bucket.
            source_dir: The directory in the bucket to upload the files to.

        Returns:
            A dictionary containing two lists:
            - "success": A list of filenames that were successfully uploaded.
            - "errors": A list of filenames that failed to upload.
        """
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_address)
        success = []
        errors = []
        for file in files:
            if file.filename == "":
                continue
            try:
                blob_name = f"{source_dir}/{user_id}-{file.filename}"
                blob = bucket.blob(blob_name)
                blob.upload_from_file(file.file)
                success.append(blob_name)
            except Exception as e: # pylint: disable=W0718
                print(f"Error uploading file {file.filename}: {e}")
                errors.append(blob_name)
        response = {
            "success": success,
            "errors": errors
        }
        return response


    def upload_metadata_file(
        self,
        jsonl_data: list[dict[str, Any]],
        user_id: str,
        project_id: str,
        bucket_address: str,
        metadata_dir: str
    ) -> str | bool:
        """
        Uploads a metadata file to a Google Cloud Storage bucket.

        Args:
            jsonl_data: A list of dictionaries containing the metadata.
            user_id: The ID of the user associated with the metadata.
            project_id: The ID of the Google Cloud project.
            bucket_address: The address of the Google Cloud Storage bucket.
            metadata_dir: The directory within the bucket where the file should be saved.

        Returns:
            Metadata filename if the upload was successful, False otherwise.
        """
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_address)
        metadata_file_name = f"{metadata_dir}/metadata_{user_id}_{uuid.uuid4()}.json"
        data = pd.DataFrame(jsonl_data).to_json(orient="records", lines=True)
        try:
            blob = bucket.blob(metadata_file_name)
            blob.upload_from_string(data)
            return metadata_file_name
        except Exception as e: # pylint: disable=W0718
            print(f"Error uploading metadata file: {e}")
            return False


    def create_metadata_jsonl(
        self,
        uris: list[str],
        user_id: str,
        client_project_id: str, 
        project_id: str,
        bucket_address: str,
        metadata_dir: str
    ) -> str | bool:
        """
        Creates a JSONL file containing metadata for a list of URIs and uploads it to a Google Cloud Storage bucket.

        The function iterates through the provided URIs, extracts relevant information such as file extension and 
        generates a unique ID. It then constructs a metadata dictionary for each URI, including user ID, mime type,
        and a Google Cloud Storage URI pointing to the file. This metadata is then uploaded as a JSONL file to the
        specified bucket.

        Args:
            uris (list[str]): A list of URIs representing the files for which metadata will be generated.
            user_id (str): The ID of the user associated with the files.
            project_id (str): The ID of the Google Cloud project.
            bucket_address (str): The address of the Google Cloud Storage bucket.
            metadata_dir (str): The directory within the bucket where the metadata file will be uploaded.

        Returns:
            str | bool: The filename of the uploaded metadata file if successful, False otherwise.
        """
        jsonl_data = []
        for uri in uris:
            bucket_uri = f"gs://{bucket_address}/{uri}"
            basename = os.path.basename(uri)
            if basename.startswith(f"{user_id}-"):
                filename = basename[len(user_id)+1:]
            else:
                filename = basename
            struct_data = {"user_id": user_id, "client_project_id": client_project_id, "section_name": filename}
            file_extension = os.path.splitext(uri)[-1]
            doc_id = str(uuid.uuid4())
            mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if file_extension == ".docx" else "application/pdf"

            metadata = {
                "id": doc_id,
                "structData": struct_data,
                "content": {"mimeType": mimetype, "uri": bucket_uri}
            }
            jsonl_data.append(metadata)
        uploaded_metadata_filename = self.upload_metadata_file(jsonl_data, user_id, project_id, bucket_address, metadata_dir)
        if not uploaded_metadata_filename:
            return False

        return uploaded_metadata_filename


    def import_documents_from_gcs_jsonl(
        self,
        authed_session: AuthorizedSession,
        project_id: str,
        location: str,
        datastore_id: str,
        metadata_uri: str
    ) -> str:
        """Imports documents from a JSONL file in GCS."""
        if location == "global":
            base_url = "https://discoveryengine.googleapis.com/v1"
        else:
            base_url = f"https://{location}-discoveryengine.googleapis.com/v1"

        payload = {
            "reconciliationMode": "INCREMENTAL",
            "gcsSource": {"inputUris": [metadata_uri]},
        }
        header = {"Content-Type": "application/json"}
        es_endpoint = f"{base_url}/projects/{project_id}/locations/{location}/collections/default_collection/dataStores/{datastore_id}/branches/default_branch/documents:import"
        response = authed_session.post(es_endpoint, data=json.dumps(payload), headers=header)
        if "name" in response.json():
            return response.json()["name"]
        return False


    def list_documents(self, request: DocumentsRequest) -> list[dict[str, str]]:
        """
        Lists documents associated with a specific user ID from the Discovery Engine service.

        Args:
            user_id: The ID of the user whose documents should be retrieved.

        Returns:
            A list of dictionaries, where each dictionary represents a document and contains 
            the document's ID and URI. Returns an empty list if no documents are found.
        """
        client_options = (
            ClientOptions(api_endpoint=f"{self.location}-discoveryengine.googleapis.com")
            if self.location != "global"
            else None
        )
        client = discoveryengine.DocumentServiceClient(client_options=client_options)

        parent = client.branch_path(
            project=self.project_id,
            location=self.location,
            data_store=self.datastore_id,
            branch="default_branch",
        )
        try:
            response = client.list_documents(parent=parent)
        except Exception as e: # pylint: disable=W0718
            print(f"Error listing documents: {e}")
            return []

        documents = [
            {
                "document_id": doc.id, 
                "document_filename": doc.struct_data.get("section_name", ""),
                "document_client_project_id": doc.struct_data.get("client_project_id", ""),
                "document_uri": doc.content.uri,
            } 
            for doc in response if (
                doc.struct_data.get("user_id") == request.user_id and 
                doc.struct_data.get("client_project_id") == request.client_project_id
                )
        ]
        return documents


    def remove_document(self, document_id: str) -> bool:
        """
        Removes a document from a VAIS data store in Google Discovery Engine.

        Args:
            document_id: The ID of the document to remove.

        Returns:
            True if the document was removed successfully, False otherwise.
        """
        if self.location == "global":
            base_url = "https://discoveryengine.googleapis.com/v1"
        else:
            base_url = f"https://{self.location}-discoveryengine.googleapis.com/v1"

        endpoint = f"{base_url}/projects/{self.project_id}/locations/{self.location}/collections/default_collection/dataStores/{self.datastore_id}/branches/0/documents/{document_id}"
        header = {"Content-Type": "application/json"}
        response = self.auth_session.delete(endpoint, headers=header)
        print(response.json())
        if response.status_code == 200:
            return True
        return False

    def remove_multiple_documents(self, request: RemoveDocumentsRequest) -> RemoveDocumentsResponse:
        success = 0
        errors = 0
        for document_id in request.document_ids:
            removed = self.remove_document(document_id)
            if removed:
                success += 1
            else:
                errors += 1
        if success:
            return RemoveDocumentsResponse(
                status = True,
                message = f"Successfully removed: {success} out of {success+errors}"
            )
        return RemoveDocumentsResponse(
            status = False,
            message = "Failed to remove documents."
        )


    def reconstruct_document(self, chunked_document: dict[str, Any]) -> str:
        """Reconstructs a document from its chunks."""
        reconstructed_document = ""
        for chunk in chunked_document['chunks']:
            reconstructed_document += chunk["content"]
        reconstructed_document = markdown.markdown(reconstructed_document, extensions=['nl2br'])
        return reconstructed_document


    def get_parsed_document(self, document_id: str) -> ViewExtractedDocumentResponse:
        if self.location == "global":
            base_url = "https://discoveryengine.googleapis.com/v1alpha"
        else:
            base_url = f"https://{self.location}-discoveryengine.googleapis.com/v1alpha"

        endpoint = f"{base_url}/projects/{self.project_id}/locations/{self.location}/collections/default_collection/dataStores/{self.datastore_id}/branches/0/documents/{document_id}:getProcessedDocument?processed_document_type=CHUNKED_DOCUMENT"
        header = {"Content-Type": "application/json"}
        response = self.auth_session.get(endpoint, headers=header)

        if response.status_code == 200:
            response_json = json5.loads(response.json()["jsonData"])
            return ViewExtractedDocumentResponse(
                status = True,
                document_id = document_id,
                context = self.reconstruct_document(response_json)
            )
        return ViewExtractedDocumentResponse(
            status = False,
            document_id = document_id,
            context = ""
        )


    def get_import_status(self, lro_id: str) -> str:
        if self.location == "global":
            base_url = "https://discoveryengine.googleapis.com/v1"
        else:
            base_url = f"https://{self.location}-discoveryengine.googleapis.com/v1"
        endpoint = f"{base_url}/projects/{self.project_id}/locations/{self.location}/collections/default_collection/dataStores/{self.datastore_id}/branches/0/operations/{lro_id}"
        response = self.auth_session.get(endpoint)
        if response.status_code == 200:
            if "done" in response.json():
                return "SUCCESS"
            else:
                return "INPROGRESS"
        return "Failed to retrieve the import status."
    
    def get_import_works(self) -> list[str] | None:
        if self.location == "global":
            base_url = "https://discoveryengine.googleapis.com/v1"
        else:
            base_url = f"https://{self.location}-discoveryengine.googleapis.com/v1"
        endpoint = f"{base_url}/projects/{self.project_id}/locations/{self.location}/collections/default_collection/dataStores/{self.datastore_id}/operations/"
        response = self.auth_session.get(endpoint)
        if response.status_code == 200:
            pending_operations = []
            for operation in response.json()["operations"]:
                if "done" not in operation:
                    pending_operations.append(operation["name"])
            return pending_operations
        return None
        

