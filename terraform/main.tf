locals {
  config = yamldecode(file("../gen_ai/llm.yaml"))
  table_schemas = {
    "ground_truth" = {
      fields = [
        { name = "question_id", type = "STRING", mode = "REQUIRED" },
        { name = "question", type = "STRING", mode = "REQUIRED" },
        { name = "gt_answer", type = "STRING", mode = "REQUIRED" },
        { name = "gt_document_names", type = "STRING", mode = "REPEATED" },
      ]
    },
    "prediction" = {
      fields = [
        { name = "user_id", type = "STRING", mode = "REQUIRED" },
        { name = "prediction_id", type = "STRING", mode = "REQUIRED" },
        { name = "timestamp", type = "TIMESTAMP", mode = "REQUIRED" },
        { name = "system_state_id", type = "STRING", mode = "REQUIRED" },
        { name = "session_id", type = "STRING", mode = "REQUIRED" },
        { name = "question_id", type = "STRING", mode = "REQUIRED" },
        { name = "question", type = "STRING", mode = "REQUIRED" },
        { name = "react_round_number", type = "STRING", mode = "REQUIRED" },
        { name = "response", type = "STRING", mode = "REQUIRED" },
        { name = "retrieved_documents_so_far", type = "STRING", mode = "REQUIRED" },
        { name = "post_filtered_documents_so_far", type = "STRING", mode = "REQUIRED" },
        { name = "retrieved_documents_so_far_content", type = "STRING", mode = "REQUIRED" },
        { name = "post_filtered_documents_so_far_content", type = "STRING", mode = "REQUIRED" },
        { name = "post_filtered_documents_so_far_all_metadata", type = "STRING", mode = "REQUIRED" },
        { name = "confidence_score", type = "INTEGER", mode = "REQUIRED" },
        { name = "response_type", type = "STRING", mode = "REQUIRED" },
        { name = "run_type", type = "STRING", mode = "REQUIRED" },
        { name = "time_taken_total", type = "FLOAT", mode = "REQUIRED" },
        { name = "time_taken_retrieval", type = "FLOAT", mode = "REQUIRED" },
        { name = "time_taken_llm", type = "FLOAT", mode = "REQUIRED" },
        { name = "tokens_used", type = "INTEGER", mode = "REQUIRED" },
        { name = "summaries", type = "STRING", mode = "REQUIRED" },
        { name = "relevance_score", type = "STRING", mode = "REQUIRED" },
        { name = "additional_question", type = "STRING", mode = "NULLABLE" },
        { name = "plan_and_summaries", type = "STRING", mode = "REQUIRED" },
      ]
    },
    "experiment" = {
      fields = [
        { name = "system_state_id", type = "STRING", mode = "REQUIRED" },
        { name = "session_id", type = "STRING", mode = "REQUIRED" },
        { name = "github_hash", type = "STRING", mode = "REQUIRED" },
        { name = "gcs_bucket_path", type = "STRING", mode = "REQUIRED" },
        { name = "pipeline_parameters", type = "STRING", mode = "REQUIRED" },
        { name = "comments", type = "STRING", mode = "NULLABLE" },
      ]
    },
    "query_evaluation" = {
      fields = [
        { name = "prediction_id", type = "STRING", mode = "REQUIRED" },
        { name = "timestamp", type = "TIMESTAMP", mode = "REQUIRED" },
        { name = "system_state_id", type = "STRING", mode = "REQUIRED" },
        { name = "session_id", type = "STRING", mode = "REQUIRED" },
        { name = "question_id", type = "STRING", mode = "REQUIRED" },
        { name = "react_round_number", type = "STRING", mode = "REQUIRED" },
        { name = "metric_type", type = "STRING", mode = "REQUIRED" },
        { name = "metric_level", type = "STRING", mode = "REQUIRED" },
        { name = "metric_name", type = "STRING", mode = "REQUIRED" },
        { name = "metric_value", type = "FLOAT64", mode = "REQUIRED" },
        { name = "metric_confidence", type = "FLOAT64", mode = "NULLABLE" },
        { name = "metric_explanation", type = "STRING", mode = "NULLABLE" },
        { name = "run_type", type = "STRING", mode = "REQUIRED" },
        { name = "response_type", type = "STRING", mode = "REQUIRED" },
      ]
    },
    "questions" = {
      fields = [
        { name = "question_id", type = "STRING", mode = "REQUIRED" },
        { name = "question", type = "STRING", mode = "REQUIRED" },
        { name = "parent_question_id", type = "STRING", mode = "NULLABLE" },
      ]
    }
  }
}

variable "project_id" {
  description = "The Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "The region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The zone"
  type        = string
  default     = "us-central1-a"
}

provider "google" {
  credentials = file(local.config.terraform_credentials)
  project     = var.project_id
  region      = var.region
  zone        = var.zone
}

resource "google_compute_network" "platform_gen_ai_network" {
  name                    = "platform-gen-ai-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "platform_gen_ai_subnet" {
  name                     = "platform-gen-ai-subnet"
  ip_cidr_range            = "10.100.0.0/24"
  network                  = google_compute_network.platform_gen_ai_network.id
  private_ip_google_access = true
  stack_type               = "IPV4_ONLY"
}

resource "google_compute_instance" "default" {
  name         = local.config.terraform_instance_name
  machine_type = "e2-medium"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
    }
  }

  network_interface {
    network = google_compute_network.platform_gen_ai_network.id
    access_config {
    }
  }
}

resource "google_redis_instance" "default" {
  name               = local.config.terraform_redis_name
  tier               = "BASIC" # STANDARD_HA for highly available
  memory_size_gb     = 1
  authorized_network = google_compute_network.platform_gen_ai_network.id

  redis_configs = {
    maxmemory-policy = "allkeys-lru"
  }
}

resource "google_dns_managed_zone" "redis_private_zone" {
  name        = "redis-private-zone"
  dns_name    = "t2xservice.internal."
  description = "Private DNS zone to allow hostname connections to the T2X Redis instance."
  visibility  = "private"
  private_visibility_config {
    networks {
      network_url = google_compute_network.platform_gen_ai_network.id
    }
  }
}

resource "google_dns_record_set" "redis" {
  name         = "redis.t2xservice.internal."
  type         = "A"
  ttl          = 300
  managed_zone = google_dns_managed_zone.redis_private_zone.name
  rrdatas      = [google_redis_instance.default.host]
  depends_on   = [google_redis_instance.default]
}

resource "google_bigquery_dataset" "dataset" {
  dataset_id    = local.config.dataset_name
  location      = "us-central1" # Change to your desired region
  friendly_name = "AI Experiment Data"
}


resource "google_bigquery_table" "tables" {
  for_each            = local.table_schemas
  dataset_id          = google_bigquery_dataset.dataset.dataset_id
  table_id            = each.key
  schema              = jsonencode(each.value.fields)
  deletion_protection = false
}
