region           = "us-central1"
zone             = "us-central1-a"
vpc_network_name = "t2x-network"
vpc_subnet_name  = "t2x-subnet"
vpc_subnet_cidr  = "10.0.0.0/24"
nat_router_name  = "t2x-nat-router"
nat_gateway_name = "t2x-nat-gateway"

# README step Automate Deployments with Cloud Build - step 2:
# Set the domain to a name under control, or leave unset to default to using nip.io with the load balancer IP.
# If set, update the DNS records for the domain to point to the load balancer IP after creating it.

# global_lb_domain = "demoapp.example.com" # example only
