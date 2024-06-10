"""Provides the HtmlExtractor class for extracting textual data from HTML files and organizing the extracted data into separate files for structured document processing."""

import re

from gen_ai.extraction_pipeline.document_extractors.base_extractor import BaseExtractor
import markdownify


class HtmlExtractor(BaseExtractor):
    """Extractor class of textual data from HTML files and chunks sections into separate files.

    This class inherits from the `BaseExtractor` and provides specialized
    functionality for extracting text content from HTML documents.

    Args:
        filepath (str): The path to the .docx file.
        config_file_parameters (dict[str, str]): Configuration settings for the
          extraction process.

    Attributes:
        filepath (str): Stores the path to the input file.
        config_file_parameters (dict[str, str]): Stores the configuration
          parameters.
        html_extraction (str): Configuration parameter fot the extraction
          method. Defaults to 'default'.
        html_chunking (str):  Configuration parameter fot the chunking method.
          Defaults to 'default'.
    """

    def __init__(self, filepath: str, config_file_parameters: dict[str, str]):
        super().__init__(filepath, config_file_parameters)
        self.html_extraction = config_file_parameters.get(
            "html_extraction", "default"
        )
        self.html_chunking = config_file_parameters.get(
            "html_chunking", "default"
        )

    def process(self, output_dir: str):
        pass


class DefaultHtmlExtractor:
    """Default extractor class that provides methods for extracting content from html files.

    Args:
        filepath (str): The path to the html file.

    Attributes:
        filepath (str): The path to the html file.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    @staticmethod
    def extract_text_from_html(html_text: str) -> str | None:
        """Extracts and cleans up text from an HTML tags.

        Args:
            html_text: The HTML string to process.

        Returns:
            str: The extracted and cleaned text.
        """
        tags = (
            "<br>",
            "<p>",
            "<li>",
            "<ul>",
            "</li>",
            "<hr>",
            "<div>",
            "</ul>",
            "<td>",
            "<tr>",
            "<h1>",
            "<h2>",
            "<h3>",
            "<h4>",
            "<h5>",
            "<h6>",
        )
        text = re.sub(
            r"<[^>]+>",
            lambda match: ("\n" if match.group(0) in tags else " "),
            html_text,
        )
        text = re.sub(r"&nbsp;", " ", text)

        while True:
            new_text = re.sub(r"\n{3}", "\n\n", text)
            new_text = re.sub(r"^\s+", "", new_text)
            if new_text == text:
                break
            text = new_text
        if new_text:
            return new_text
        return None


class CustomHtmlExtractor:
    """Custom Extractor class processes HTML files, specifically handling ordered lists, tables and other HTML tags.

    It offers methods to extract content from HTML and convert it cleanly into
    Markdown formatting.

    Args:
        filepath (str): The path to the html file.

    Attributes:
        filepath (str): The path to the html file.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    @staticmethod
    def extract_text_from_html(html_text: str) -> str | None:
        """Converts an HTML string to Markdown format.

        Args:
            html_text (str): The HTML string to convert.

        Returns:
            str: The converted Markdown text.
        """

        try:
            markdown_text = markdownify.markdownify(html_text)
            return markdown_text
        except TypeError as e:
            print(f"An error occurred during conversion: {e}")
            return None
