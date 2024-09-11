import os


def get_version():
    version_file_path = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_file_path, "r") as version_file:
        return version_file.read().strip()


__version__ = get_version()