#!/usr/bin/env python3
"""
This module checks the efficiency of a pipeline in processing questions using a language model.
It provides functionality to load questions from a CSV file and process each question through
the model to evaluate response times and effectiveness under various scenarios. This script
supports running in batch or step mode to accommodate different testing requirements.

Classes:
    None

Functions:
    get_input_df(csv_path)
    run_single_prediction(question, member_context_full=None)
    get_default_personalized_info(row)
    run_pipeline(mode, csv_path=None)

Exceptions:
    None
"""
import uuid
from timeit import default_timer
from typing import Literal
import click
import llm
import pandas as pd

from gen_ai.common.ioc_container import Container

from gen_ai.check_recall import (
    prepare_recall_calculation,
    prepare_scoring_calculation,
    prepare_semantic_score_calculation,
)


def get_input_df(csv_path: str) -> pd.DataFrame:
    """Loads a CSV file and returns it as a pandas DataFrame.

    Args:
        csv_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(csv_path)
    return df


def run_single_prediction(question: str, member_context_full: dict | None = None) -> str:
    """Processes a single question through the language model API and returns the response.

    Args:
        question (str): The question to be processed by the language model.
        member_context_full (dict | None): Optional dictionary containing additional context to personalize
        the language model's response.

    Returns:
        str: The answer generated by the language model, or an error message if the process fails.

    Raises:
        Exception: An exception is raised and caught internally if the language model API call fails.
        The specific error message is printed.
    """
    try:
        conversation = llm.respond_api(question, member_context_full)
        return conversation.exchanges[-1].answer
    except Exception as e:  # pylint: disable=W0718
        Container.logger().info(msg=e)
        return "I apologize, but no answer is available at this time."


def get_default_personalized_info(row: dict) -> dict | None:
    """Extracts and returns the default personalization information from a row if available.

    Args:
        row (dict): The row from which personalization information is to be extracted.

    Returns:
        dict | None: A dictionary containing personalization information if 'set_number' exists in the row;
        otherwise, None.

    Side Effects:
        Prints a fallback message if 'set_number' is not found in the row.
    """
    if "set_number" in row:
        return {"set_number": row["set_number"].lower(), "policy_number": str(row["policy_number"]).lower()}
    Container.logger().info(msg="Personalization info does not have set_number, falling back to None")
    return None


def prepend_question_with_member_info(row: dict, question: str) -> str:
    if "Context" not in row:
        Container.logger().info(msg="There is no Member Info In the GT Doc. Asking simple question")
        return question
    return llm.enhance_question(question, row["Context"])


def compute_gt_scores(session_id: str, df: pd.DataFrame):
    df_joined = prepare_recall_calculation(session_id, df, True)
    df_joined = prepare_scoring_calculation(df_joined)
    df_joined = prepare_semantic_score_calculation(df_joined)
    return df_joined


def run_pipeline(
    mode: Literal["batch", "step"] = "step",
    csv_path: str | None = None,
    comments: str | None = None,
    is_gt: bool = False,
    n_calls: int = 1,
    output_path: str = ".",
) -> None:
    """Executes the pipeline check based on the specified mode.

    This function orchestrates the loading and processing of questions to evaluate the language model's response
    efficiency.
    In 'batch' mode, it processes questions from a CSV file; in 'step' mode, it runs a set of predefined questions
    to measure performance iteratively.

    Args:
        mode (Literal["batch", "step"]): The mode of operation. 'batch' processes questions from a CSV file,
                                         'step' processes a predefined list of questions.
        csv_path (str, optional): The path to the CSV file containing questions. Required if mode is 'batch'.
        comments: (str, Optional): Comments about the run
        is_gt: (bool): Flag specifying whether to use Ground Truth files or not, defaults to False
        n_calls: (int): Number of calls that pipeline will go through the input csv file
        output_path: (str): Path where to save the results

    Raises:
        ValueError: If the specified mode is not implemented.

    Side Effects:
        Prints session details, questions, and responses to the console.
        Measures and displays execution time in 'step' mode.
    """

    for _ in range(n_calls):
        session_id = str(uuid.uuid4())
        Container.logger().info(msg=f"Session id is: {session_id}")
        Container.comments = comments
        if mode == "batch":
            df = get_input_df(csv_path)
            df = df.sort_values(by=["Multi-turn or Single-turn", "Scenario/Question #"])
            for i, row in df.iterrows():
                if row["Multi-turn or Single-turn"] == "Single-turn":
                    print("SINGLE-TUUUUUUUUUURN")
                    print(row["Scenario/Question #"])
                    member_id = str(uuid.uuid4())
                else:
                    print("MULTI-TUUUUUUUUUURN")
                    print(row["Scenario/Question #"])
                    if "Q1" in row["Scenario/Question #"]:
                        member_id = str(uuid.uuid4())
                    if "Q2" in row["Scenario/Question #"]:
                        pass

                Container.logger().info(msg=f"Asking question {i} in document ")
                question = row["question"]
                Container.logger().info(msg=f"Question: {question}")

                if Container.config.get("personalization"):
                    personalized_data = get_default_personalized_info(row)
                    personalized_data["session_id"] = session_id
                    personalized_data["member_id"] = member_id
                Container.original_question = question
                question = prepend_question_with_member_info(row, question)
                answer = run_single_prediction(question, personalized_data)
                Container.logger().info(msg=f"Answer: {answer}")

            if is_gt:
                Container.logger().info(msg="Computing GT Scores")
                scores_df = compute_gt_scores(session_id, df)
                import os

                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                scores_df.to_csv(f"{output_path}/{session_id}_run.csv", index=None)
        elif mode == "step":
            start = default_timer()
            question = "I would like to know the answer to a question from the following member. The member is a subscriber, 59 years old female without any OI (other insurance coverage). The member is traveling to Europe on vacation. Do they have coverage if they need to seek medical treatment?"
            # question = "I would like to know the answer to a question from the following member. The member is a subscriber, 54 years old female without any OI (other insurance coverage). Can her partner be covered under her plan?"
            # question = "My doctor said he'd be billing using G0105. Is that a valid code?"
            question = "I would like to know the answer to a question from the following member. The member is a subscriber, 59 years old female without any OI (other insurance coverage). Their doctor said he'd be billing using G0105. Is that a valid code?"
            question = "I would like to know the answer to a question from the following member. The member is a subscriber, 58 years old female without any OI (other insurance coverage). Does her insurance cover post-mastectomy items like bras?"
            question = "I would like to know the answer to a question from the following member. The member is a subscriber, 57 years old female without any OI (other insurance coverage). The member got laid off, is she eligible for COBRA?"
            question = "I would like to know the answer to a question from the following member. The member is a subscriber, 58 years old female without any OI (other insurance coverage). Now that she has lost her employment, will her dependents be covered under COBRA?"
            question = "I would like to know the answer to a question from the following member. The member is a subscriber, a 59 year old female without any OI (other insurance coverage). She was just diagnosed with ESRD and is now eligible for Medicare. Which is her primary plan?"
            # question = 'Find providers in a 85216 ZIP code for a allara healthcare provider'
            acis = "001acis"
            for idx, input_query in enumerate([question]):
                Container.original_question = question
                Container.logger().info(msg=f"Asking question {idx} in document ")
                Container.logger().info(msg=f"Question: {input_query}")
                answer = run_single_prediction(
                    input_query,
                    {"set_number": acis, "member_id": "q1e23", "session_id": session_id, "policy_number": "905531"},
                )
                Container.logger().info(msg=f"Answer: {answer}")
            end = default_timer()
            print(f"Total flow took {end - start} seconds")
        else:
            raise ValueError("Not implemented mode")


@click.command()
@click.argument("mode", required=True)
@click.argument("csv_path", required=False, type=click.Path(exists=True))
@click.argument("comments", required=False)
@click.option("--is_gt", is_flag=True, help="Trigger the computation of ground truth scores.")
@click.option("--n_calls", default=1, help="Number of times to run the pipeline.")
@click.argument("output_path", required=False)
def run_cli(mode, csv_path=None, comments=None, is_gt=False, n_calls=1, output_path="."):
    run_pipeline(mode, csv_path, comments, is_gt, n_calls, output_path)


if __name__ == "__main__":
    run_cli()
