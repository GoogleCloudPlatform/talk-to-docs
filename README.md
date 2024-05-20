# Solution Accelerators for GenAI
This repository contains platform code for accelerating development of GenAI solutions in Applied AI Engineering team

![alt text](resources/image.png)

# Structure

- **docs**: This directory contains documentation, user guides, and any other resources that help you understand and use the GenAI solution accelerators effectively.

- **src**: The source code for the GenAI solution accelerators is located here. This is where you'll find the core codebase for the tools and frameworks provided in this repository.

- **data**: This directory may contain sample data or data-related resources that can be used for testing and development.

- **tests**: Test cases and resources related to testing the GenAI solution accelerators are stored in this directory.

- **scripts**: Utility scripts or automation scripts that can assist in various tasks related to GenAI development and deployment.

- **examples**: This directory may contain example projects, code snippets, or reference implementations that showcase how to use the provided solution accelerators effectively.

## Getting Started

To get started with the GenAI solution accelerators, follow the instructions in the documentation located in the `docs` directory. It will provide you with step-by-step guidance on how to set up your development environment and use the tools and frameworks provided in this repository.

## Contribution Guidelines

We welcome contributions from the GenAI community! If you'd like to contribute to this repository, please follow our [Contribution Guidelines](CONTRIBUTING.md) to ensure a smooth collaboration process.

## License

This repository is licensed under the [Apaache License](LICENSE). See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or need assistance, feel free to reach out to the project maintainers or create an issue in this repository.

Happy GenAI development!


## Setting up
To begin development you can use 2 different approaches: using Python Environment or using Docker. Below are instructions for each approach.

### Setting up Python Environment
Make sure to install miniconda environment:
```
cd ~/
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
```
After that just install the package in editable mode:

```
pip install -e .
```

### Setting up Docker
If this is your first time, you probably don't have Docker installed on VM. Execute the following commands:
```
sudo apt update && sudo apt upgrade
sudo apt install make
sudo apt install docker.io
sudo groupadd docker
sudo usermod -aG docker $USER
sudo chmod 777 /var/run/docker.sock
```

### Setting up environment

```
make build && make container
```

If you want to remove it, execute:

```
make clean
```


### Copying resources

Make sure `gcs_source_bucket` field in `llm.yaml` is up to date with the latest extraction in use. Then run the copying python script:
```
python gen_ai/copy_resources.py
```


### Deploying terraform resources
You must declare the project ID where you want to deploy Compute Engine, BigQuery, and Memorystore Redis resources. Pass the value to the `project_id` root module [input variable](https://developer.hashicorp.com/terraform/language/values/variables#assigning-values-to-root-module-variables) using a `.tfvars` file, the `-var` command line argument, or the `TF_VAR_project_id` environment variable.

The variable sets the default project in the `google` provider block. The provider also uses the `region` and `zone` variables to set defaults for resources.

The `credentials` provider argument supplies credentials to deploy resources. It reads the path to a local service account key file from the `llm.yaml` file.


### Updating BigQuery table

All the runs are logged into a dataset located in the project discovered from [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials). Change the dataset name by updating the `dataset_name` variable in the `llm.yaml` file.
