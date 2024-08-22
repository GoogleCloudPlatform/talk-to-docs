# Extraction Pipeline

## Overview
Extraction pipeline is a comprehensive suite of tools custom created to extract valuable data from various document formats, including PDF, DOCX, XML, and JSON.
The extraction pipeline supports both *Batch* processing for a one-time extraction and *Continuous* processing to monitor a Google Cloud Storage bucket for new files and extract data at specified intervals.
In Batch mode all the files in given directory or gs bucket are processed at once. 
In Continuous mode the pipeline processes files in given bucket and then checks the bucket every set time if new files were added. If so, the files are processed and copied into given output bucket or datastore.

## Usage
There are three parameters you need to pass to run the pipeline:
- **mode** - it can be either `batch` or `continuous` (Required argument)
- **input** - input directory or gs bucket directory where input files are located (Required argument)
- **output** - output directory in local system, GCS bucket path or Datastore ID. Default is `output_data`.

```sh
python gen_ai/extraction_pipeline/processor.py <mode> -i <input> -o <output>
```
## Input
Expected input is a folder containing `.json` files retrieved from original html, pdf, and docx files.

Sample `json` file from html:

```json
{
  "name": "KM1613153",
  "metadata": {
    "id": "KM1613153",
    "structData": {
      "name": "Name of the document",
      "type": "vkm:ArticleContent",
      "url": "https:/path-to-original-url",
      "doc_identifier": "KM1613153",
      "date_modified": "2024-04-04T14:22:10.440Z",
      "date_published": "2024-04-04T14:22:10.510Z",
      "annotations": "some_annotations_listed_with_space",
      "cob": "",
      "policy_number": "",
      "doc_cat": "kc"
    },
    "content": {
      "mimeType": "text/html",
      "uri": "gs://../KM1613153.html"
    }
  },
  "article": "Parsed HTML Text here"
}


```

Sample `json` file from pdf/docs (content is not parsed, so a postprocessing step is required):

```json
{
  "name": "KM1707152",
  "metadata": {
    "id": "KM1707152",
    "structData": {
      "name": "Preventive Care Services",
      "type": "vkm:UploadedContent",
      "url": "https://link_to_original",
      "doc_identifier": "KM1707152",
      "date_modified": "2024-04-05T18:13:18.207Z",
      "date_published": "2024-04-05T18:13:18.215Z",
      "annotations": "some_annotations_listed_with_space",
      "cob": "",
      "policy_number": "",
      "doc_cat": "kc"
    },
    "content": {
      "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "uri": "gs://../KM1707152.docx"
    }
  },
  "article": "preventive care services.docx"
}
```

Right now, for the post-processing step for `docx` and `pdf` files you need to have them locally in the `raw_files` directory.
This directory **must be** in the current directory from where you are executing the script.

## Batch mode
Batch mode can process both local directories and GCS buckets. And the result can be uploaded to the GCS bucket automatically or datastore can be updated through BQ table if necessary. Arguments you need to pass are `mode` and `input`. If no `output` is provided, default value is "output_dir" which is created automatically. Input directory can be local directory or GCS bucket directory.

## Continuous mode
Continuous mode processes GCS buckets only, it first processes all the files in the input directory. Afterwards it checks the bucket every 10 minutes if new files were added. If so, it processes new files and transfer them to the destination GCS bucket or updates datastore through BQ table. Arguments you need to pass are `mode`, `input` (bucket address) and `output` where processed files will be uploaded.

## Config file

The configuration file of which type of extraction to use for each file type is in `config.yaml`, inside *'extraction_pipeline'* directory. For each type of file there are two parameters: `Extraction` and `Chunking`. If no value is given in config file, "default" is used as the value.

## Examples
*Batch processing of a local directory*
```sh
python gen_ai/extraction_pipeline/processor.py batch -i /mnt/resources/dataset/main_folder -o output_dir
```

*Batch processing of a GCS Bucket and GCS Bucket output*
```sh
python gen_ai/extraction_pipeline/processor.py batch -i gs://dataset_raw_data/extractions -o gs://dataset_clean_data
```

*Continuous processing of a GCS bucket and Datastore output*
```sh
python gen_ai/extraction_pipeline/processor.py continuous -i gs://dataset_raw_data/20240417_docx -o datastore:datastore_id
```
## Installation
1. Update the package list and Install the dependencies
```sh
sudo apt-get update && sudo apt-get install libgl1 poppler-utils tesseract-ocr
```
2. Install the python requirements  
Update the line 13 in `setup.py` file to: `install_requires=get_requirements("gen_ai/extraction_pipeline/requirements.txt"),`  
Then run pip install:
```sh
pip install -e .
```


## Important Notes
- Ensure that the `config.yaml` file is correctly configured with the required parameters.
- For GCS operations, make sure you have the necessary permissions to access the buckets.
- In continuous mode, the script will run indefinitely, monitoring the GCS bucket for new files.


