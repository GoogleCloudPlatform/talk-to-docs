# README step Bootstrap 6b:
# Set the target project ID and the service account email you created with gcloud.
project_id                = "my-project-id" # example only
terraform_service_account = "terraform-service-account@my-project-id.iam.gserviceaccount.com" # example only
region                    = "us-central1"

services = [
  "aiplatform.googleapis.com",
  "bigquery.googleapis.com",
  "cloudbuild.googleapis.com",
  "cloudresourcemanager.googleapis.com",
  "compute.googleapis.com",
  "discoveryengine.googleapis.com",
  "dns.googleapis.com",
  "iam.googleapis.com",
  "iamcredentials.googleapis.com",
  "iap.googleapis.com",
  "logging.googleapis.com",
  "monitoring.googleapis.com",
  "redis.googleapis.com",
  "run.googleapis.com",
  "serviceusage.googleapis.com",
  "vpcaccess.googleapis.com",
  "workflows.googleapis.com",
]

# README step 6b - the optional Data Mover service account email.
# This service account must already exist and have permission to read the source documents.
# You can omit this if you don't intend to use a service account to migrate document extractions to the staging bucket.
# (I.e. you will manually upload the documents to the staging bucket or otherwise use your user account to migrate the documents.)
# data_mover_service_account = "staging-data-mover@???.iam.gserviceaccount.com" # example only
cloudbuild_iam_roles  = ["roles/cloudbuild.builds.builder"]
cloudbuild_sa_name    = "t2x-cloudbuild"
staging_bucket_prefix = "t2x-staging"
