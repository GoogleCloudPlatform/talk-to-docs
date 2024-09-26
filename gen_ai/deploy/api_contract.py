import datetime

login_response = {"Status": bool, "userData": {"user_id": str, "user_name": str, "profile_picture": "url"}}

# dashboard response for better understanding
all_projects_response = {
    "recent_projects": [{"project_id": str, "project_name": str, "created_on": datetime, "updated_on": datetime}, ...],
    "all_projects": [...],
}

uploaded_documents_response = {"documents": [{"document_url": str, "created_on": datetime, "document_name": str}]}

create_project_input = {
    "project_name": str,
    "user_id": str,
    "local_documents": [{"file_object": bytearray, "file_title": str}, ...],
    "external_documents": [{"document_url": str, "created_on": datetime, "document_name": str}, ...],
}

create_project_response = {"project_id": str}

delete_project_response = {"status": bool}

project_detail_response = {
    "project_name": str,
    "created_on": datetime,
    "updated_on": datetime,
    "documents": [{"document_url": str, "created_on": datetime, "document_name": str}],
}

delete_document_input = {"document_url": str, "user_id": str}

previous_chat_input = {"project_id": str, "user_id": str}

previous_chat_response = {
    "chat_list": [
        {
            "is_ai": bool,
            "messsage": str,
            "like_status": "Enum[like, dislike, nan]",
            "icon": "image_user_url",
            "prediction_id": str,
        },
        ...,
    ],
    "document_list": [{"document_url": str, "created_on": datetime, "document_name": str}, ...],
    "query_suggestions": [{"question": str}],
}

debug_response = {
    "answer": str,
    "plan_and_summaries": str,
    "context_used": str,
    "additional_information_to_retrieve": str,
    "confidence_score": str,
    "documents": [{"document_section_name": str, "document_url": "str", "document_relevancy_score": float}, ...],
    "total_latency": datetime.timedelta,
}

chat_input = {"message": str, "user_id": str, "project_id": str}

mid_upload_response = {"document_url": str, "created_on": datetime, "document_name": str}

like_input = {"like_status": "Enum[like, dislike, nan]", "user_id": str}

like_response = {"status": bool}

feedback_like_input = {"rating": int, "prediction_id": str, "user_id": str}

feedback_dislike_input = {"selected_response": str, "prediction_id": str, "user_id": str, "feedback_message": str}

get_prompt_names_input = {"user_id": str, "project_id": str}
get_prompt_names_response = {"prompts": [{"prompt_name": str, "prompt_value": str}]}

change_prompt_input = {"user_id": str, "project_id": str, "prompt_name": str, "prompt_value": str}
change_prompt_response = {"status": bool}

restore_prompt_response = {"status": bool}
