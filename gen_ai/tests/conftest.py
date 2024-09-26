# import os
# import pytest
# from gen_ai import constants


# @pytest.fixture(autouse=True, scope="session")
# def override_processed_files_dir():
#     """Overrides the `PROCESSED_FILES_DIR` constant for tests.

#     This fixture temporarily changes the directory where processed files are stored
#     during test execution. The original directory is restored after all tests are
#     completed.

#     Yields:
#         None

#     Example usage:
#         This fixture is automatically applied to all tests in the session due to 
#         `autouse=True`. There is no need to explicitly use it in your test functions.
#     """
#     original_dir = constants.PROCESSED_FILES_DIR
#     constants.PROCESSED_FILES_DIR = f'{os.getenv("HOME")}/resources/dataset_name/main_folder'
#     yield
#     constants.PROCESSED_FILES_DIR = original_dir