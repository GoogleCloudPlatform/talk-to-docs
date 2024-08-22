import os
import json
import json5
import re
from langchain_community.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.embeddings import Embeddings
from gen_ai.common.common import load_yaml
from gen_ai.common.embeddings_provider import EmbeddingsProvider
from gen_ai.common.storage import DefaultStorage
from gen_ai.common.vector_provider import VectorStrategy, VectorStrategyProvider

import datetime

LLM_YAML_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ingestion_config.yaml")

def replace_consecutive_whitespace(text):
    """Replaces consecutive whitespace characters of the same type with a single instance."""
    return re.sub(r'(\s)\1+', r'\1', text)

def parse_into_hashmap(processed_files_dir: str) -> dict[str, dict]:
    """
    Parses documents text files and associated metadata into a structured hashmap.

    This function iterates through text files in the specified directory,
    extracting their content and loading corresponding metadata from JSON files.
    It constructs a hashmap where keys are policy numbers (or 'generic' if unavailable)
    and values are nested dictionaries. In these nested dictionaries, keys are 
    combinations of the original file path and section name, and values are the 
    metadata dictionaries enriched with content and filename.

    Args:
        processed_files_dir (str): The directory containing processed text files 
                                   and their metadata.

    Returns:
        dict: A hashmap organizing data by policy number with nested dictionaries
              containing metadata and content for each section.
    """
    data_dict = {}
    for filename in os.listdir(processed_files_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(processed_files_dir, filename)
            metadata_filepath = os.path.join(processed_files_dir, filename.replace(".txt", "_metadata.json"))
            try:
                with open(filepath, "r") as f:
                    data = f.read()
                with open(metadata_filepath, "r") as f:
                    metadata = json.load(f)
                metadata["content"] = replace_consecutive_whitespace(data)
                metadata["filename"] = filename
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error processing {filename}: {e}")

            section_identifier = f'{metadata["original_filepath"]} - {metadata["section_name"]}'
            policy_number = metadata.get("policy_number")
            policy_number = policy_number or "generic"
            if policy_number not in data_dict:
                data_dict[policy_number] = {}
            if section_identifier in data_dict[policy_number]:
                print(f"Duplicate key: {section_identifier}")
                return
            data_dict[policy_number][section_identifier] = metadata
    return data_dict


def provide_vector_indices(regenerate: bool = False):
    """
    Provides or regenerates vector indices for embeddings using a specified vector strategy.

    This function initializes or updates vector indices based on the configuration specified in LLM_YAML_FILE.
    It manages embeddings and vector strategies to create a Chroma vector store instance suitable for semantic
    operations.

    Args:
        regenerate (bool, optional): If true, existing vector indices are regenerated; otherwise, the current indices
        are used. Defaults to False.

    Returns:
        Chroma: An instance of Chroma vector store populated with the appropriate vector indices for the configured
        embeddings and vector strategy.
    """
    config = load_yaml(LLM_YAML_FILE)
    embeddings_name = config.get("embeddings_name")
    embeddings_model_name = config.get("embeddings_model_name")
    vector_name = config.get("vector_name")
    dataset_name = config.get("dataset_name")
    processed_files_dir = config.get("processed_files_dir").format(dataset_name=dataset_name)
    vectore_store_path = config.get("vector_store_path")

    embeddings_provider = EmbeddingsProvider(embeddings_name, embeddings_model_name)
    embeddings: Embeddings = embeddings_provider()

    vector_strategy_provider = VectorStrategyProvider(vector_name)
    vector_strategy: VectorStrategy = vector_strategy_provider(
        storage_interface=DefaultStorage(), config=config, vectore_store_path=vectore_store_path
    )

    local_vector_indices = {}
    return vector_strategy.get_vector_indices(regenerate, embeddings, local_vector_indices, processed_files_dir)


def clean_line(line: str) -> str:
    """
    Cleans a line of text by removing unwanted characters and extra spaces.

    Args:
        line (str): The line of text to be cleaned.

    Returns:
        str: The cleaned line of text.
    """
    line = re.sub(r'\s+', ' ', line).strip()
    cleaned_line = re.sub(r'[^a-zA-Z0-9\s]', '', line)
    return cleaned_line


def parse_text_to_dict(text: str, config: dict[str, str]) -> dict[str, str]:
    """
    Parses policy specific document text into a dictionary with titles and the corresponding text sections.

    Args:
        text (str): The text to be parsed.
        config (dict): A dictionary containing configuration options.

    Returns:
        dict: A dictionary where keys are titles extracted from the text and values are the associated text sections.
    """
    model_name = config["model_name"]
    temperature = config["temperature"]
    max_output_tokens = config["max_output_tokens"]
    find_title_prompt = config["find_title_prompt"]
    
    llm = None

    result = {}
    text_sections = re.split(r"\n---\n", text)
    title_pattern = r"\*\*(.*?)\*\*$"
    if len(text_sections) == 1:
        if not llm:
            llm = VertexAI(model_name=model_name, temperature=temperature, max_output_tokens=max_output_tokens)
            title_template = PromptTemplate(input_variables=["text"], template=find_title_prompt)
            title_chain = LLMChain(
                llm=llm,
                prompt=title_template,
                output_key="text",
                verbose=False,
            )
        try:
            title = title_chain.run(text)
        except Exception as e: # pylint: disable=W0718
            print(f"Exception occured while creating title: {e}")
            title = "Section " + text.split('\n')[0]
        result[title] = text
    else:
        for section in text_sections:
            lines = section.splitlines()
            for i in range(len(lines)):
                line = lines[i]
                stripped_line = line.strip()
                title_found = re.search(title_pattern, stripped_line)
                if title_found:
                    title = clean_line(line)
                    text = "\n".join(lines[i:])
                    result[title] = text
                    break

    return result


def find_docs_and_ask_llm(sections_to_clean: dict[str, dict], 
                          vector_indices, 
                          documents_hashmap: dict[str, dict], 
                          config: dict[str, str]
) -> dict[str, dict]:
    """
    Identifies relevant documents based on topics within sections to be cleaned, 
    uses an LLM to generate modified content based on provided changes, 
    and updates the documents hashmap with the new content.

    Args:
        sections_to_clean (dict): A dictionary mapping policy numbers to data about sections to clean,
            including the document key, topic, and text to be removed or modified.
        vector_indices: An index for performing similarity searches on documents.
        documents_hashmap (dict): A dictionary mapping policy numbers and document keys 
            to the content of documents.
        config (dict): A dictionary containing configuration settings.

    Returns:
        dict: A dictionary containing information about the removals made, including the 
            original document, the removed text, and the new content.
    """

    model_name = config["model_name"]
    temperature = config["temperature"]
    max_output_tokens = config["max_output_tokens"]
    template = config["test_content"]
    corrector_prompt = config["json_corrector_prompt"]
    similar_documents_count = config["similar_documents_count"]

    llm = VertexAI(
        model_name=model_name, 
        temperature=temperature, 
        max_output_tokens=max_output_tokens
    )
    answer_template = PromptTemplate(
        input_variables=["document", "changes"], 
        template=template
    )
    chain = LLMChain(
        llm=llm,
        prompt=answer_template,
        output_key="text",
        verbose=False,
        llm_kwargs={"response_mime_type": "application/json"},
    )
    json_corrector_template = PromptTemplate(
        input_variables=["json"], 
        template=corrector_prompt
    )
    corrector_chain = LLMChain(
        llm=llm,
        prompt=json_corrector_template,
        output_key="text",
        verbose=False,
        llm_kwargs={"response_mime_type": "application/json"},
    )

    removals = {}
    for policy_number, policy_data in sections_to_clean.items():
        documents_hashmap[policy_number] = documents_hashmap["generic"].copy()
        for doc_key, sections_data in policy_data.items():
            print(f"Processing {doc_key}, {len(sections_data)} items")
            for topic, text in sections_data.items():
                documents = vector_indices.similarity_search(topic, k=similar_documents_count) 
                for document in documents:
                    key = f'{document.metadata["original_filepath"]} - {document.metadata["section_name"]}'
                    if key == doc_key:
                        continue
                    if key not in documents_hashmap[policy_number]:
                        continue
                    content = documents_hashmap[policy_number][key]["content"]
                    llm_output = chain.run(document=content, changes=text)
                    modified_output = llm_output.replace("```json", "").replace("```", "")
                    try:
                        the_output = json5.loads(modified_output)
                        modified_text = the_output["new_content"]
                        removed = the_output["removed"]
                        removals[key] = {
                            "item": [doc_key, topic, text], 
                            "removed": removed, 
                            "document": [key, content], 
                            "new_content": modified_text
                        }
                        documents_hashmap[policy_number][key]["content"] = modified_text
                        print(f"\n\n{'-'*50}\nWorking on {key}")
                        print(doc_key, topic)

                        print(f"REMOVED: {removed}\n")
                        print(f"TO REMOVE: {text}")
                        print(f"NEW CONTENT: {modified_text}")
                        print("-"*50)
                    except Exception as e:
                        try:
                            modified_output = corrector_chain.run(json=modified_output)
                            modified_output = modified_output.replace("```json", "").replace("```", "")
                            the_output = json5.loads(modified_output)
                            modified_text = the_output["new_content"]
                            removed = the_output["removed"]
                            removals[key] = {
                                "item": [doc_key, topic, text], 
                                "removed": removed, 
                                "document": [key, content], 
                                "new_content": modified_text
                            }
                            documents_hashmap[policy_number][key]["content"] = modified_text
                            print(f"\n\n{'-'*50}\nWorking on {key}")
                            print(doc_key, topic)
                            print(f"REMOVED: {removed}\n")
                            print(f"TO REMOVE: {text}")
                            print(f"NEW CONTENT: {modified_text}")
                            print("-"*50)
                        except Exception as _:  # pylint: disable=W0718
                            print("ERROR OCCURED at parsing the llm output")
                            print(llm_output)
                            print(e)
    removals_filepath = f"removal_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    with open(removals_filepath, "w") as f:
        json.dump(removals, f)
    return removals


def save_files_to_disk(documents_hashmap: dict[str, dict], output_dir: str):
    """
    Saves files and their metadata to the specified output directory.

    Args:
        documents_hashmap (dict): A dictionary where keys are policy numbers 
                                  and values are dictionaries containing file data.
        output_dir (str): The directory where files and metadata will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files_created = 0
    for policy_number, policy_data in documents_hashmap.items():
        for _, data in policy_data.items():
            filename_parts = [policy_number] if policy_number != "generic" else []
            filename_parts.append(data["filename"])
            filename = "_".join(filename_parts)
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, "w") as f:
                    f.write(data["content"])
                files_created += 1 
                metadata = data.copy()
                metadata.pop("filename")
                metadata.pop("content")
                metadata["policy_number"] = policy_number
                metadata_filepath = os.path.join(output_dir, filename.replace(".txt", "_metadata.json"))
                with open(metadata_filepath, "w") as f:
                    json.dump(metadata, f)
                files_created += 1 
            except IOError as e:
                print(f"Error writing file: {e}")
            except json.JSONDecodeError as e:
                print(f"Error serializing metadata: {e}")
    return files_created


def extract_sections_and_create_new_docs(specific_docs: dict[str, dict], config: dict[str, str]) -> tuple[dict, dict]:
    """
    Splits policy specific documents into sections based on their content and configuration. 

    This function iterates through specific documents, parses their content into sections
    using the provided configuration, and reorganizes the data into two structures:

    1. `sections_to_clean`: A nested dictionary organizing sections by policy number, document key, and title.
    2. `new_documents_hashmap`: A nested dictionary containing new documents created by splitting the original ones
       based on the identified sections. Each new document includes relevant metadata and the extracted section content.

    Args:
        specific_docs (dict): A dictionary containing documents to process, organized by policy number and document key.
        config (dict): A configuration dictionary used for parsing document content into sections.

    Returns:
        tuple: A tuple containing two dictionaries:
            - sections_to_clean (dict): The reorganized sections.
            - new_documents_hashmap (dict): The newly created documents.
    """

    sections_to_clean = {}
    new_documents_hashmap = {}
    for policy_number, policy_data in specific_docs.items():
        for doc_key, doc in policy_data.items():
            for title, body in parse_text_to_dict(doc["content"], config).items():
                if policy_number not in sections_to_clean:
                    sections_to_clean[policy_number] = {}
                if doc_key not in sections_to_clean[policy_number]:
                    sections_to_clean[policy_number][doc_key] = {}
                sections_to_clean[policy_number][doc_key][title] = body
                
                new_doc_key = f"{doc_key} {title}"
                new_doc = doc.copy()
                new_doc["section_name"] = f"{doc['section_name']} {title}"
                new_doc["filename"] = f"{doc['filename'].replace('.txt', '')}_{title.lower().replace(' ', '_')}.txt"
                new_doc["content"] = body
                if policy_number not in new_documents_hashmap:
                    new_documents_hashmap[policy_number] = {}
                new_documents_hashmap[policy_number][new_doc_key] = new_doc
    return sections_to_clean, new_documents_hashmap


def run_the_pipeline():
    """
    Executes a pipeline for processing and modifying knowledge management (KM) documents.

    This pipeline involves the following steps:

    1. Loads configuration from a YAML file.
    2. Creates a data store for KM files and loads documents into a hashmap.
    3. Separates 'specific' documents from 'generic' ones.
    4. Extracts sections from 'specific' documents and creates new documents if necessary.
    5. Uses an LLM to process sections and identify text for removal.
    6. Rewrites the modified text in the dictionary.
    7. Saves the processed documents to disk.

    Key features:

    - Utilizes vector indices for document search and retrieval.
    - Leverages an LLM for text modification based on document context.
    - Handles 'specific' and 'generic' document categories.
    - Organizes documents using a hashmap structure.
    - Saves output to a specified directory.
    """
    config = load_yaml(LLM_YAML_FILE)

    # 1. create a data store that stores only KM-kc files (copy in folder only kc files)
    # 1.5. create dictionary where load all of the documents in the directory
    vector_indices = provide_vector_indices()

    documents_hashmap = parse_into_hashmap(config["processed_files_dir"])
    print(f"Done loading files into hashmap. hashmap length: {len(documents_hashmap)}")

    # 2. go thru files in the directory, if has "Specific .." in section_name call the process
    specific_docs = {pn: docs for pn, docs in documents_hashmap.items() if pn != "generic"}
    documents_hashmap = {"generic":documents_hashmap["generic"]}
    print(f"Found Specific Docs: {sum(map(len, specific_docs.values()))}")

    # 3. GO thru each section inside the processed KM file and isolate the data in it. Call another function that takes this data as argument (title and text)
    sections_to_clean, new_documents_hashmap = extract_sections_and_create_new_docs(specific_docs, config)
    print(f"Found documents with items to clean: {len(sections_to_clean)}")
    
    # 4. The function that receives title and text searches data inside the data store and gets all documents related to it. Once it gets it, it will run call to llm that needs to return modified text
    removals = find_docs_and_ask_llm(sections_to_clean, vector_indices, documents_hashmap, config)

    # # 5. rewrite the modified text in dictionary with files contents
    count_items_to_remove = sum(map(len, sections_to_clean.values()))
    print(f"Items to remove were {count_items_to_remove}")
    print(f"Touched docs: {len(removals)}")
    real_removal = 0
    for item in removals.values():
        if item is not None or item != "":
            real_removal += 1
    print(f"Real removals in : {real_removal}")

    
    # 6. once done with the processing, save files
    for policy_number, policy_data in new_documents_hashmap.items():
        documents_hashmap[policy_number].update(policy_data)
    files_created = save_files_to_disk(documents_hashmap, config["output_dir"])
    print(f"Created {files_created} files in {len(documents_hashmap)} policy numbers.")


if __name__ == "__main__":
    run_the_pipeline()