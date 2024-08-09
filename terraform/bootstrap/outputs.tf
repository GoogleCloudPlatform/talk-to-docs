output "enabled_services" {
  description = "The names of enabled services"
  value       = [for service in google_project_service.project_service : service.service]
}

output "cloudbuild_service_account" {
  description = "The Cloud Build service account for the T2X service."
  value       = google_service_account.cloudbuild_service_account.email
}

output "data_mover_service_account" {
  description = "The service account to impersonate for data migration."
  value       = var.data_mover_service_account
}

output "staging_bucket" {
  description = "The source code and data staging bucket."
  value       = google_storage_bucket.staging_bucket.name
}

output "artifact_registry_tag_namesapce" {
  description = "The talk-to-docs Docker repository."
  value       = "us-central1-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.talk_to_docs.name}"
}
