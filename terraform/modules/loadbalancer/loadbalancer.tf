# Global IP address.
resource "google_compute_global_address" "t2x_lb_global_address" {
  name         = "t2x-lb-global-address"
  address_type = "EXTERNAL"
}

# HTTPS resources.
resource "google_compute_global_forwarding_rule" "https_redirect" {
  name                  = "t2x-global-forwarding-rule-https"
  target                = google_compute_target_https_proxy.https_redirect.id
  port_range            = "443-443"
  ip_address            = google_compute_global_address.t2x_lb_global_address.address
  ip_protocol           = "TCP"
  load_balancing_scheme = "EXTERNAL_MANAGED"
}

resource "google_compute_target_https_proxy" "https_redirect" {
  name             = "t2x-lb-target-https-proxy"
  url_map          = google_compute_url_map.t2x_lb_url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.cert.id]
}

locals {
  t2x_lb_domain = coalesce(var.global_lb_domain, "${google_compute_global_address.t2x_lb_global_address.address}.nip.io")
}

resource "random_id" "certificate" {
  byte_length = 4
  prefix      = "t2x-lb-cert-"

  keepers = {
    # Ensure a new id is generated when the domain changes
    # ref: https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_managed_ssl_certificate#example-usage---managed-ssl-certificate-recreation
    # ref: https://github.com/hashicorp/terraform-provider-google/issues/5356
    domain = local.t2x_lb_domain
  }
}

resource "google_compute_managed_ssl_certificate" "cert" {
  name = random_id.certificate.hex

  lifecycle {
    create_before_destroy = true
  }

  managed {
    domains = [local.t2x_lb_domain]
  }
}

# Default Backend Service.
resource "google_storage_bucket" "default_backend_bucket" {
  location                    = "US"
  name                        = "${var.project_id}-default-backend-bucket"
  uniform_bucket_level_access = true


  website {
    main_page_suffix = "site/index.html"
    not_found_page   = "site/404.html"
  }

  force_destroy = true
}

resource "google_storage_managed_folder" "folder" {
  bucket = google_storage_bucket.default_backend_bucket.name
  name   = "site/"
}

# # Create a public viewer IAM member for the default backend bucket managed folder.
# # Projects enforcing policy constraints allowing IAM members only from
# # authorized domains will fail to create this resource.
# # Omit this resource to prevent the constraint from failing the deployment.
# # Requests reaching the default bucket will return a 403 error in that case.
# resource "google_storage_managed_folder_iam_member" "public_viewer" {
#   bucket         = google_storage_managed_folder.folder.bucket
#   managed_folder = google_storage_managed_folder.folder.name
#   role           = "roles/storage.objectViewer"
#   member         = "allUsers"
# }

resource "google_storage_bucket_object" "index" {
  name         = "${google_storage_managed_folder.folder.name}index.html"
  content      = "<html><body><h1>Hello, World!</h1></body></html>"
  content_type = "text/html"
  bucket       = google_storage_bucket.default_backend_bucket.name
}

resource "google_storage_bucket_object" "not_found" {
  name         = "${google_storage_managed_folder.folder.name}404.html"
  content      = "<html><body><h1>Uh oh</h1></body></html>"
  content_type = "text/html"
  bucket       = google_storage_bucket.default_backend_bucket.name
}

resource "google_compute_backend_bucket" "default_backend_bucket" {
  name        = "default-backend-bucket"
  bucket_name = google_storage_bucket.default_backend_bucket.name
}

# URL Map.
resource "google_compute_url_map" "t2x_lb_url_map" {
  name            = "t2x-lb-url-map"
  default_service = google_compute_backend_bucket.default_backend_bucket.id

  host_rule {
    hosts        = [local.t2x_lb_domain]
    path_matcher = "t2x-path-matcher"
  }

  path_matcher {
    default_service = google_compute_backend_bucket.default_backend_bucket.id
    name            = "t2x-path-matcher"

    dynamic "path_rule" {
      for_each = var.backend_services

      content {
        paths   = path_rule.value.paths
        service = path_rule.value.service
        route_action {
          url_rewrite {
            path_prefix_rewrite = path_rule.value.path_prefix_rewrite
          }
        }
      }
    }
  }
}
