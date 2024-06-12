"""
`commands.ingest` module: Controller for the `ingest` CLI command.
"""

from collections import namedtuple
import os
import glob
import traceback
import io
from shutil import rmtree

import click
import chromadb
from bs4 import BeautifulSoup
from bs4 import Comment as HTMLComment
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from warcio.archiveiterator import ArchiveIterator
from libzim.reader import Archive
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from flask import current_app
from time import perf_counter
from statistics import mean

from warc_gpt import WARC_RECORD_DATA

ChromaItem = namedtuple('ChromaItem', ['documents', 'ids', 'metadatas', 'embeddings'])

class Ingester:
    def __init__(self, multi_chunk_mode, batch_size):
        # Init embedding model
        self.embedding_model = SentenceTransformer(
            os.environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL"],
            device=os.environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"],
        )
    
        # Init text splitter function
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=os.environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL"],
            chunk_overlap=int(os.environ["VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP"]),
            tokens_per_chunk=self.embedding_model[0].max_seq_length,
        )  # Note: The text splitter adjusts its cut-off based on the models' max_seq_length

        # Init vector store
        chroma_client = chromadb.PersistentClient(
            path=os.environ["VECTOR_SEARCH_PATH"],
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.chroma_collection = chroma_client.create_collection(
            name=os.environ["VECTOR_SEARCH_COLLECTION_NAME"],
            metadata={"hnsw:space": os.environ["VECTOR_SEARCH_DISTANCE_FUNCTION"]},
        )

        self.multi_chunk_mode = multi_chunk_mode
        self.batch_size = batch_size
        self.encoding_timings = []
        self.total_records = 0
        self.total_embeddings = 0

    def _extract_html(self, record_data, get_record_html_fn):
        if not record_data["warc_record_content_type"].startswith("text/html"):
            return

        try:
            response_as_text = get_record_html_fn()

            soup = BeautifulSoup(response_as_text, "html.parser")

            # Skip documents with no body tag
            if not soup.body or len(soup.body) < 1:
                return

            all_text = soup.body.findAll(string=True)

            for text in all_text:
                if text.parent.name in ["script", "style"]:  # No <script> or <style>
                    continue

                if isinstance(text, HTMLComment):  # No HTML comments
                    continue

                record_data["warc_record_text"] += f"{text} "

            record_data["warc_record_text"] = record_data["warc_record_text"].strip()
        except Exception:
            click.echo(
                f"- Could not extract text from {record_data['warc_record_target_uri']}"
            )
            click.echo(traceback.format_exc())

    def _extract_pdf(self, record_data, get_record_raw_fn=None):
        if not record_data["warc_record_content_type"].startswith("application/pdf") or get_record_content_fn is None:
            return

        raw = io.BytesIO(get_record_raw_fn())
        pdf = PdfReader(raw)

        for page in pdf.pages:
            record_data["warc_record_text"] += page.extract_text()

    def _process_record(self, record_data, get_record_html_fn, get_record_raw_fn=None):
        chunk_prefix = os.environ["VECTOR_SEARCH_CHUNK_PREFIX"]

        #
        # Extract text from text/html
        #
        self._extract_html(record_data, get_record_html_fn)

        #
        # Extract text from PDF
        #
        self._extract_pdf(record_data, get_record_raw_fn)

        #
        # Stop here if we don't have text, or text contains less than 5 words
        #
        if not record_data["warc_record_text"]:
            return

        if len(record_data["warc_record_text"].split()) < 5:
            return

        record_data["warc_record_text"] = record_data["warc_record_text"].strip()
        self.total_records += 1

        # Split text into chunks
        text_chunks = self.text_splitter.split_text(record_data["warc_record_text"])
        click.echo(f"{record_data['warc_record_target_uri']} = {len(text_chunks)} chunks.")

        if not text_chunks:
            return

        # Add VECTOR_SEARCH_CHUNK_PREFIX to every chunk
        text_chunks = [chunk_prefix + chunk for chunk in text_chunks]

        # Generate embeddings and metadata for each chunk
        item = self.chunk_objects(record_data, text_chunks)
        self.total_embeddings += len(item.embeddings)

        # Store embeddings and metadata
        self.chroma_collection.add(**item._asdict())

    def process_warc_record(self, record, warc_file):
        if record.rec_type != "response":
            return

        # Extract metadata
        rec_headers = record.rec_headers
        http_headers = record.http_headers

        if not rec_headers or not http_headers:
            return

        record_data = dict(WARC_RECORD_DATA)
        record_data["warc_filename"] = os.path.basename(warc_file)
        record_data["warc_record_id"] = rec_headers.get_header("WARC-Record-ID")
        record_data["warc_record_date"] = rec_headers.get_header("WARC-Date")
        record_data["warc_record_target_uri"] = rec_headers.get_header("WARC-Target-URI")
        record_data["warc_record_content_type"] = http_headers.get_header("Content-Type")
        record_data["warc_record_text"] = ""

        # Skip incomplete records
        if (
            not record_data["warc_record_id"]
            or not record_data["warc_record_date"]
            or not record_data["warc_record_target_uri"]
            or not record_data["warc_record_content_type"]
        ):
            return

        # Skip records that are not HTTP 2XX
        if not http_headers.get_statuscode().startswith("2"):
            return

        self._process_record(record_data, lambda: record.content_stream().read().decode("utf-8"), lambda: record.content_stream().read())

    def process_zim_entry(self, entry, zim_file, zim_date, zim_uuid):
        record_data = dict(WARC_RECORD_DATA)
        record_data["warc_filename"] = os.path.basename(zim_file)
        record_data["warc_record_id"] = f'{zim_uuid}/{entry._index}'
        record_data["warc_record_date"] = zim_date
        record_data["warc_record_target_uri"] = '/'.join(entry.path.split('/')[1:])
        record_data["warc_record_content_type"] = entry.get_item().mimetype
        record_data["warc_record_text"] = ""

        self._process_record(record_data, lambda: bytes(entry.get_item().content).decode("UTF-8"))

    def chunk_objects(
        self,
        record_data: dict,
        text_chunks: list[str]
    ):
        """
        Return one document, metadata, id, and embedding object per chunk; also return
        control variables multi_chunk_mode and encoding_timings

        """
        normalize_embeddings = os.environ["VECTOR_SEARCH_NORMALIZE_EMBEDDINGS"] == "true"
        chunk_prefix = os.environ["VECTOR_SEARCH_CHUNK_PREFIX"]

        chunk_range = range(len(text_chunks))

        documents = [record_data["warc_filename"] for _ in chunk_range]

        ids = [f"{record_data['warc_record_id']}-{i+1}" for i in chunk_range]

        metadatas = [
            dict(record_data, **{"warc_record_text": text_chunks[i][len(chunk_prefix):]})
            for i in chunk_range
        ]

        # In some contexts, passing all the text chunks to embedding_model.encode() at once
        # takes advantage of parallelization, so we default to that, but stop doing it if
        # it is too slow.
        if self.multi_chunk_mode:
            start = perf_counter()
            embeddings = self.embedding_model.encode(
                text_chunks,
                batch_size=self.batch_size,
                normalize_embeddings=normalize_embeddings,
            ).tolist()
            encoding_time = perf_counter() - start

            if len(text_chunks) == 1:
                self.encoding_timings.append(encoding_time)
            else:
                if len(self.encoding_timings) == 0:
                    pass
                elif encoding_time > len(text_chunks) * mean(self.encoding_timings):
                    self.multi_chunk_mode = False
                    click.echo("Leaving multi-chunk mode")
        else:
            # we've left multi-chunk mode, and there's no need to capture timings anymore
            embeddings = [
                self.embedding_model.encode(
                    [chunk],
                    batch_size=1,
                    normalize_embeddings=normalize_embeddings,
                ).tolist()[0]
                for chunk in text_chunks
            ]

        return ChromaItem(documents, ids, metadatas, embeddings)


@current_app.cli.command("ingest")
@click.option(
    "--batch-size",
    default=2,
    type=int,
    help="Batch size for encoding",
    show_default=True,
)
def ingest(batch_size) -> None:
    """
    Generates sentence embeddings and metadata for a set of WARCs and saves them in a vector store.

    See: options in .env.example
    """
    warc_files = []
    zim_files = []

    if batch_size == 1:
        multi_chunk_mode = False
    elif batch_size > 1:
        multi_chunk_mode = True
    else:
        raise click.UsageError("Batch size must be a positive integer, preferably a power of 2")
    encoding_timings = []

    # Cleanup
    rmtree(os.environ["VECTOR_SEARCH_PATH"], ignore_errors=True)
    os.makedirs(os.environ["VECTOR_SEARCH_PATH"], exist_ok=True)

    # List WARC files to process
    for ext in ('.warc', '.warc.gz'):
        warc_files += glob.glob(os.environ["WARC_FOLDER_PATH"] + f'/*{ext}', recursive=True)
    warc_files.sort()

    # List ZIM files to process
    zim_files += glob.glob(os.environ["ZIM_FOLDER_PATH"] + "/*.zim", recursive=True)
    zim_files.sort()

    if not warc_files:
        click.echo("No WARC files to ingest.")

    if not zim_files:
        click.echo("No ZIM files to ingest.")

    if not warc_files and not zim_files:
        exit(1)

    click.echo(f"{len(warc_files)} WARC files to ingest.")
    click.echo(f"{len(zim_files)} ZIM files to ingest.")

    ingester = Ingester(multi_chunk_mode, batch_size)

    #
    # For each WARC:
    # - Extract text from text/html and application/pdf records
    # - Split and generate embeddings for said text
    # - Save in vector store
    #
    for warc_file in warc_files:
        click.echo(f"🗜️ Ingesting HTML and PDF records from {warc_file}")
        with open(warc_file, "rb") as stream:
            for record in ArchiveIterator(stream):
                ingester.process_warc_record(record, warc_file)

    #
    # For each ZIM:
    # - Extract text from HTML records
    # - Split and generate embeddings for said text
    # - Save in vector store
    for zim_file in zim_files:
        click.echo(f"🗜️ Ingesting HTML records from {zim_file}")
        zim = Archive(zim_file)
        if 'Date' in zim.metadata_keys:
            zim_date = zim.get_metadata('Date').decode('utf-8')
        else:
            zim_date = '1970-01-01'

        for i in range(0, zim.all_entry_count):
            entry = zim._get_entry_by_id(i)
            return_tuple = ingester.process_zim_entry(entry, zim_file, zim_date, zim.uuid)

    click.echo(f"Total: {ingester.total_embeddings} embeddings from {ingester.total_records} HTML/PDF records.")
