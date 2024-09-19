"""
This module provides an example on how to call Talk2Docs API. 
It uses fake data as member_context_full

You can run it either in the localhost mode (when T2X end point is running locally), or can access remote Server.


For the Remote Server you will need to set following env variables:

API_DOMAIN                - Talk2Docs Endpoint, for example: x.241.x.173.nip.io
DEVELOPER_SERVICE_ACCOUNT - Service account used for impersonation and token retrieval.
                            Must have roles/iam.serviceAccountTokenCreator and roles/run.invoker

In cloud shell:

export PROJECT_ID=...
export API_DOMAIN=...

export DEVELOPER_SERVICE_ACCOUNT_NAME="t2d-developer"
export DEVELOPER_SERVICE_ACCOUNT="$DEVELOPER_SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

gcloud iam service-accounts create $DEVELOPER_SERVICE_ACCOUNT \
  --display-name $DEVELOPER_SERVICE_ACCOUNT \
  --project $PROJECT_ID

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:$DEVELOPER_SERVICE_ACCOUNT" \
  --role "roles/run.invoker" \
  --role "roles/iam.serviceAccountTokenCreator"

python example.py

"""
import os

import google.auth.transport.requests
import requests
import google.auth
import google
import google.oauth2.credentials
from google.auth import impersonated_credentials
import google.auth.transport.requests

api_domain = os.environ.get("API_DOMAIN")
target_principal = os.environ.get("DEVELOPER_SERVICE_ACCOUNT")

if api_domain:
    audience = f"https://{api_domain}/t2x-api"
else:
    audience = "http://127.0.0.1:8080"

target_scopes = ['https://www.googleapis.com/auth/cloud-platform']


def get_impersonated_id_token(_target_principal: str, _target_scopes: list, _audience: str | None = None) -> str:
    """Use Service Account Impersonation to generate a token for authorized requests.
    Caller must have the “Service Account Token Creator” role on the target service account.
    Args:
        _target_principal: The Service Account email address to impersonate.
        _target_scopes: List of auth scopes for the Service Account.
        _audience: the URI of the Google Cloud resource to access with impersonation.
    Returns: Open ID Connect ID Token-based service account credentials bearer token
    that can be used in HTTP headers to make authenticated requests.
    refs:
    https://cloud.google.com/docs/authentication/get-id-token#impersonation
    https://cloud.google.com/iam/docs/create-short-lived-credentials-direct#user-credentials_1
    https://stackoverflow.com/questions/74411491/python-equivalent-for-gcloud-auth-print-identity-token-command
    https://googleapis.dev/python/google-auth/latest/reference/google.auth.impersonated_credentials.html
    The get_impersonated_id_token method is equivalent to the following gcloud commands:
    https://cloud.google.com/run/docs/configuring/custom-audiences#verifying
    """
    # Get ADC for the caller (a Google user account).
    creds, project = google.auth.default()

    # Create impersonated credentials.
    target_creds = impersonated_credentials.Credentials(
        source_credentials=creds,
        target_principal=_target_principal,
        target_scopes=_target_scopes
    )

    # Use impersonated creds to fetch and refresh an access token.
    request = google.auth.transport.requests.Request()
    id_creds = impersonated_credentials.IDTokenCredentials(
        target_credentials=target_creds,
        target_audience=_audience,
        include_email=True
    )
    id_creds.refresh(request)

    return id_creds.token


def get_token(_audience: str):
    if not api_domain: return None
    return get_impersonated_id_token(
        _target_principal=target_principal,
        _target_scopes=target_scopes,
        _audience=_audience,
    )


def main():
    """This is main function that serves as an example how to use the respond API method"""
    url = f"{audience}/respond/"

    data = {
        "question": "I injured my back. Is massage therapy covered?",
        "member_context_full": {"set_number": "001acis", "member_id": "1234"},
    }

    token = get_token(audience)
    if token:
        response = requests.post(url, json=data, headers={'Authorization': f'Bearer {token}'}, timeout=3600)
    else:
        response = requests.post(url,  json=data, timeout=3600)

    if response.status_code == 200:
        print("Success!")
        print(response.json())  # This will print the response data
    else:
        print("Error:", response.status_code)
        print(response.text)  # This will print the error message, if any


if __name__ == "__main__":
    main()
