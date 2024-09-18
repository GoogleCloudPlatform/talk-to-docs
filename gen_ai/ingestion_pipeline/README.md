# Knowledge Management Policy Specific Document Processing Pipeline
This Python script orchestrates a pipeline for processing and modifying knowledge management (KM) documents, particularly handling the removal or modification of policy specific content from documents based on context and configuration.

# Functionality
**Configuration Loading:** Reads configuration settings from a YAML file (gen_ai/ingestion_pipeline/ingestion_config.yaml).  
**Vector Indices:** Utilizes vector indices for efficient similarity search and retrieval of relevant documents.  
**Document Loading:** Loads KM documents into a hashmap and initializes vector indices for search.  
**Document Separation:** Identifies 'specific' documents requiring modifications.  
**Section Extraction:** Extracts sections from 'specific' documents, creating new documents if needed.  
**LLM Processing:** Uses an LLM to process sections and identify text for removal or modification.  
**Text Rewriting:** Modifies document content based on LLM output.  
**Output Saving:** Saves processed documents and associated metadata to the specified output directory.  


# Configuration
The script relies on a YAML configuration file (gen_ai/ingestion_pipeline/ingestion_config.yaml) to specify various parameters, such as:
- input and output dirs: Directory containing processed text files and metadata and directory where modified documents will be saved.
- embeddings_model_name: name of embedding model.
- model_name: Name of the LLM model to use.
- temperature: Temperature setting for LLM output.
- max_output_tokens: Maximum number of tokens in LLM output.
- Prompts and templates for LLM interaction.

# Execution
- Ensure all dependencies are installed (`pip install -e .`).  
- Prepare your KM documents and metadata in the `processed_files_dir`.  
- Configure the ingestion_config.yaml file.  
- Run the script: `python gen_ai/ingestion_pipeline/ingestion.py`  

# Output
The script generates modified KM documents and their metadata in the specified output_dir. It also creates a log file (removal_{timestamp}.json) detailing the removals made during processing.
