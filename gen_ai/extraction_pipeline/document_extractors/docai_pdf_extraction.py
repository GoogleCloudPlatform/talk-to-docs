from io import BytesIO
import traceback
from typing import Optional, Sequence

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter

GCP_PROJECT_ID = "dauren-genai-bb" #os.environ["GCP_PROJECT_ID"]
DOCOCR_PROCESSOR_ID = "5d106d45b46ec280" #os.environ["DOCOCR_PROCESSOR_ID"]
GCP_PROJECT_LOCATION = "us"  # Format is "us" or "eu"
APP_MIME_TYPE = "application/pdf"
DOCOCR_PROCESSOR_VERSION = "pretrained-ocr-v2.0-2023-06-02"


def extract_pdf_chunk(pdf_reader: PdfReader, start_page: int, end_page: int) -> bytes:
    """This method takes in a PdfReader object and get the contents of the BytesIO object written from the specified page number range.

    Arguments:
        pdf_reader {PyPDF2.PdfReader} -- The PdfReader object to get the chunk from.
        start_page {int} -- The page number where the PDF chunk starts.
        end_page {int} -- The page number for the last page of the PDF chunk.

    Returns:
        bytes -- PDF chunk content in bytes.
    """

    try:
        writer = PdfWriter()

        for page_number in range(start_page - 1, end_page):
            writer.add_page(pdf_reader.pages[page_number])

        chunk_pdf_content = BytesIO()
        writer.write(chunk_pdf_content)
        chunk_pdf_content.seek(0)
    except Exception as e:
        # Handle the exception here
        traceback.print_exc()
        print("An error occurred:", str(e))

    return chunk_pdf_content.getvalue()


def list_blocks(blocks: Sequence[documentai.Document.Page.Block], text: str, page_number: int):
    # print(f"{len(blocks)} blocks detected:")
    margin = 0.5
    block_list = [[], []]
    for block in blocks:
        # print(block.layout)
        block_text = layout_to_text(block.layout, text)
        if all(x.x < margin for x in block.layout.bounding_poly.normalized_vertices):
            block_list[0].append(
                [
                    block_text,
                    min(y.y for y in block.layout.bounding_poly.normalized_vertices),
                    page_number,
                ]
            )
        else:
            block_list[1].append(
                [
                    block_text,
                    min(y.y for y in block.layout.bounding_poly.normalized_vertices),
                    page_number,
                ]
            )
    return block_list


def process_document(
    pdf_reader: PdfReader,
    start_page: int,
    end_page: int,
    process_options: Optional[documentai.ProcessOptions] = None,
) -> documentai.Document:
    """Process a range of pages from a PDF document using Document AI and return the Document object
        with all its metadata.

    Arguments:
        pdf_reader {PdfReader} -- A PDF reader object.
        start_page {int} -- The starting page number to process.
        end_page {int} -- The ending page number to process.
        process_options {documentai.ProcessOptions} -- An optional field to set config for DocOCR.
    Returns:
        documentai.Document: The processed Document AI document, or None if an error occurs.
    """

    project_id = GCP_PROJECT_ID  # The Google Cloud Platform (GCP) project ID.
    location = GCP_PROJECT_LOCATION  # The GCP location where the processor is deployed.
    mime_type = APP_MIME_TYPE  # The MIME type of the document.
    processor_id = DOCOCR_PROCESSOR_ID  # The ID of the processor from DocAI
    processor_version = (
        DOCOCR_PROCESSOR_VERSION
    )  # version of the processor to be used. Default is 'rc'.

    try:
        # You must set the `api_endpoint` if you use a location other than "us".
        client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        )

        # The full resource name of the processor version, e.g.:
        # `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
        # You must create a processor before running this sample.
        name = client.processor_version_path(project_id, location, processor_id, processor_version)

        # Extract the relevant pages for the current chunk
        chunk_pdf_content = extract_pdf_chunk(pdf_reader, start_page, end_page)

        # Load Binary Data into Document AI RawDocument Object
        raw_document = documentai.RawDocument(content=chunk_pdf_content, mime_type=mime_type)

        print(f"About to process {start_page} - {end_page}")
        # Configure the process request
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document,
            # Only supported for Document OCR processor
            process_options=process_options,
        )

        result = client.process_document(request=request)
    except Exception as e:
        # Handle the exception here
        traceback.print_exc()
        print(f"An error occurred while processing pages {start_page} to {end_page}:", str(e))
        return None
    return result.document


def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document's text. This function converts
    offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response


def extract_text_data(page: documentai.Document.Page, text: str) -> (list[list[str]], list[int]):
    # Create a dictionary of tables in both cell and row format, using which we can compare
    # text lines to see whether that line of text belongs to a table, and figure out which table
    # it belongs to.
    all_table_cells = {}
    all_table_rows = {}
    table_info = []
    sorted_table_info = []
    table_dimensions = []
    sorted_tables = []

    for i, table in enumerate(page.tables):
        # getting min and max  x,y coordinates for the table
        min_x, min_y, max_x, max_y = table_coordinates(table)
        dimensions = (min_x, min_y, max_x, max_y)

        table_cells = []
        table_rows = []
        for table_row in table.header_rows:
            process_table_row(text, table_cells, table_rows, table_row)
        for table_row in table.body_rows:
            process_table_row(text, table_cells, table_rows, table_row)
        all_table_cells[i] = table_cells
        all_table_rows[i] = table_rows

        table_info.append((table_rows, dimensions, i))

    # sort the tables based on the min_y coordinate
    sorted_table_info = sorted(table_info, key=lambda x: x[1][1])
    for stl in sorted_table_info:
        sorted_tables.append((stl[0], stl[1][0], stl[1][1], stl[2]))
    # list of coordinates for all tables
    table_dimensions = [element[1] for element in sorted_table_info]

    text_lines_only = []
    sorted_text_lines_only = []
    for l, line in enumerate(page.lines):
        line_in_text_block = False
        for td in table_dimensions:
            # check if line is inside a table
            if is_line_inside_current_text_block(line, td):
                line_in_text_block = True
                break
        # append the line once you confirm its not part of a table
        if line_in_text_block == False:
            text_lines_only.append(
                (
                    layout_to_text(line.layout, text),
                    line.layout.bounding_poly.vertices[0].x,
                    line.layout.bounding_poly.vertices[0].y,
                    l,
                )
            )

    sorted_text_lines_only = sorted(text_lines_only, key=lambda x: (50 * x[2] + x[1]))

    return sorted_tables, sorted_text_lines_only


def is_line_inside_current_text_block(line, td):
    return (
        line.layout.bounding_poly.vertices[0].y >= td[1]
        and line.layout.bounding_poly.vertices[0].y <= td[3]
    )


def table_coordinates(table):
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    table_x = [table.layout.bounding_poly.vertices[x].x for x in range(4)]
    table_y = [table.layout.bounding_poly.vertices[y].y for y in range(4)]
    for j in range(4):
        min_x = min(min_x, table_x[j])
        min_y = min(min_y, table_y[j])
        max_x = max(max_x, table_x[j])
        max_y = max(max_y, table_y[j])
    return min_x, min_y, max_x, max_y


def process_table_row(
    text: str,
    table_cells: list[str],
    table_rows: list[str],
    table_row: documentai.Document.Page.Table.TableRow,
):
    row_text = ""
    for cell in table_row.cells:
        cell_text = layout_to_text(cell.layout, text)
        table_cells.append(cell_text)
        row_text += cell_text
    table_rows.append(row_text.strip())


def convert_table_to_dataframe(table, text):
    rows = []

    for table_row in table.header_rows:
        row = [layout_to_text(cell.layout, text).strip() for cell in table_row.cells]
        header = row
        rows.append(row)

    for table_row in table.body_rows:
        row = [layout_to_text(cell.layout, text).strip() for cell in table_row.cells]
        rows.append(row)

    # Convert rows into dataframe
    dataframe = pd.DataFrame(rows, columns=header)
    dataframe = dataframe.drop(0)
    latex_table = dataframe.style.to_latex()

    return latex_table


def process_chunk(chunk_index: int, chunk_size: int, document: documentai.Document) -> list[str]:
    # For a full list of Document object attributes, please reference this page:
    # https://cloud.google.com/python/docs/reference/documentai/latest/google.cloud.documentai_v1.types.Document
    """Read the contents of the document from DocAI's Document object and sort it according to
        their y-coordinate values to get it line-by-line.

    Args:
        chunk_index {int} -- Integer index of the current chunk.
        chunk-size {int} -- The size limit of a chunk in pages. For DocAI, it is 15.
        document {documentai.Document} -- DocAI's document object.

    Returns:
        list[str] -- A list of strings after processing and sorting.
    """
    output = []
    try:
        text = document.text
        for page in document.pages:
            # page_number = page.page_number + chunk_index * chunk_size
            # page_text = f"Page Number {page_number}:\n"
            page_text = ""
            # Get the text data in the page in the form of a list of text blocks.
            tables, lines = extract_text_data(page, text)

            m = len(tables)
            n = len(lines)

            while m > 0 or n > 0:
                if m > 0 and n > 0:
                    if lines[0][2] < tables[0][2]:
                        n, page_text = transfer_line_to_output(page_text, lines, n)
                    else:
                        m, page_text = transfer_table_to_output(text, page_text, page, tables, m)
                elif m == 0:
                    while n > 0:
                        n, page_text = transfer_line_to_output(page_text, lines, n)
                else:
                    while m > 0:
                        m, page_text = transfer_table_to_output(text, page_text, page, tables, m)
            output.append(page_text.replace("\\n", "\n"))

    except Exception as e:
        # Handle the exception here
        traceback.print_exc()
        print("An error occurred:", str(e))
    return output


def process_blocks(chunk_index: int, chunk_size: int, document: documentai.Document):
    output = [[], []]
    try:
        text = document.text

        for page in document.pages:
            page_number = page.page_number + chunk_index * chunk_size
            blocks = list_blocks(page.blocks, text, page_number)
            output[0].extend(blocks[0])
            output[1].extend(blocks[1])
    except Exception as e:
        # Handle the exception here
        traceback.print_exc()
        print("An error occurred:", str(e))
    # for x in output:
    #     print(x)
    return output


def transfer_table_to_output(text, output, page, tables, m):
    curr_table = tables.pop(0)
    dataframe = convert_table_to_dataframe(page.tables[curr_table[3]], text)
    output += f" \nTable: {dataframe}\n"
    m -= 1
    return m, output


def transfer_line_to_output(output, lines, n):
    curr_line = lines.pop(0)
    output += curr_line[0]
    n -= 1
    return n, output


def gs_bucket_bypass(
    pdf_reader: PdfReader,
    process_options: documentai.ProcessOptions,
    output: list,
    blocks: list[list],
    start_page: int,
    end_page: int,
    chunk_index: int,
    chunk_size: int,
) -> list[str]:
    """This is a method designed for testing purpose. This avoids the storage-and-retrieval part
        from the normal pipeline but does the same processing on each chunk.

    Args:
        pdf_reader {PyPDF2.PdfReader} -- PdfReader object to get the raw text from.
        process_options {documentai.ProcessOptions} -- An optional field to set config for DocOCR.
        output {list} -- List of strings containing extraction text till the previous chunk.
        blocks {list{list}} -- Blocks
        start_page {int} -- Starting page index of the chunk to be processed.
        end_page {int} -- Last page index to be considered in the chunk.
        chunk_index {int} -- Integer index of the current chunk.
        chunk_size {int} -- The size limit of a chunk in pages. For DocAI, it is 15.

    Returns:
        list[str] -- An updated `output` list with the text output of the current chunk appeneded at
                        the end.
    """

    # Online processing request to Document AI
    document = process_document(
        pdf_reader,
        start_page,
        end_page,
        process_options=process_options,
    )

    if document is not None:
        # Process the current chunk and print the results
        curr_chunk_output = process_chunk(chunk_index, chunk_size, document)
        output += curr_chunk_output
        terms_boxes = process_blocks(chunk_index, chunk_size, document)
        blocks[0].extend(terms_boxes[0])
        blocks[1].extend(terms_boxes[1])

    return output, blocks


def process_document_in_chunks(file_path: str) -> list[str]:
    """This method is the starting point of Argonaut's DocAI-based PDF processing pipeline. Here,
        the document gets split into 15-page chunks and each chunk gets processed one after another.

    Arguments:
        file_path {str} -- Path of the file to be processed.

    Returns:
        list[str] -- A list of strings for each chunk of the document.
    """
    # Process options for DocAI
    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            # compute_style_info=True,
            enable_native_pdf_parsing=True,
            enable_image_quality_scores=False,
            enable_symbol=False,
        )
    )

    try:
        # Read the PDF and get the total number of pages
        print("Processing:", file_path.split("/")[-1])
        extracted_text = []
        extracted_blocks = [[], []]
        pdf_reader = None
        with open(file_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()
            pdf_reader = PdfReader(BytesIO(pdf_content))
            num_pages = len(pdf_reader.pages)


        # Split the PDF into chunks of 15 pages
        chunk_size = 15
        num_chunks = (num_pages - 1) // chunk_size + 1

        for chunk_index in range(num_chunks):
            start_page = chunk_index * chunk_size + 1
            end_page = min((chunk_index + 1) * chunk_size, num_pages)
            extracted_text, extracted_blocks = gs_bucket_bypass(
                pdf_reader,
                process_options,
                extracted_text,
                extracted_blocks,
                start_page,
                end_page,
                chunk_index,
                chunk_size,
            )


    except Exception as e:
        # Handle the exception here
        traceback.print_exc()
        print("An error occurred:", str(e))
        # You can perform additional error handling or logging as needed

    return extracted_text, extracted_blocks



