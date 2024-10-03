import threading
from contextvars import ContextVar

current_client_project_id: ContextVar[str] = ContextVar('current_client_project_id')

class UserContext:
    def __init__(self, client_project_id: str):
        self.client_project_id = client_project_id
        self.token = None

    def __enter__(self):
        self.token = current_client_project_id.set(self.client_project_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_client_project_id.reset(self.token)

def get_current_client_project_id() -> str:
    try:
        return current_client_project_id.get()
    except LookupError:
        return None