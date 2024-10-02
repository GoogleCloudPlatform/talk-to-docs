import yaml
import uuid
from google.cloud import bigquery


def insert_prompts_into_bigquery(yaml_file_path):

    client = bigquery.Client()
    project_id = "chertushkin-genai-sa"
    dataset_name = "uhg5"
    table_id = f"{project_id}.{dataset_name}.default_prompts"

    with open(yaml_file_path, "r") as file:
        prompts_data = yaml.safe_load(file)

    vertical_id = "20e264fd-7c30-4d99-8292-f02f5e92461b"

    rows_to_insert = []
    map_to_names = {
        "golden_answer_scoring_prompt": "evaluation",
        "enhanced_prompt": "personalization",
        "previous_conversation_scoring_prompt": "previous conversation scoring",
        "aspect_based_summary_prompt": "context summarization",
        "retriever_scoring_prompt": "context relevance scoring",
        "answer_scoring_prompt": "answer scoring",
        "similar_questions_prompt": "similar question generation",
        "react_chain_prompt": "answer generation",
    }
    # Loop through each prompt in the YAML file and prepare rows for insertion
    for k in prompts_data:
        prompt_name = k
        if k not in map_to_names:
            continue
        prompt_display_name = map_to_names[k].capitalize()
        prompt_value = prompts_data[k]

        prompt_id = str(uuid.uuid4())

        row = {
            "id": prompt_id,
            "vertical_id": vertical_id,
            "prompt_name": prompt_name,
            "prompt_display_name": prompt_display_name,
            "prompt_value": prompt_value,
        }

        rows_to_insert.append(row)

    errors = client.insert_rows_json(table_id, rows_to_insert)

    if errors == []:
        print("Rows inserted successfully.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))


# Usage example
insert_prompts_into_bigquery("/home/chertushkin/platform-gen-ai/gen_ai/prompts.yaml")  # Adjust path if needed
