import hashlib
import logging
import math
import re
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FINANCIAL VECTOR STORE
# --------------------------------------------------

class FinancialVectorStore:

    def __init__(
        self,
        db_path="./chroma_db",
        collection_name="corporate_statements",
        model_name="all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB and the sentence embedding model.

        Parameters
        ----------
        db_path : str
            Directory where the vector database is stored.

        collection_name : str
            Name of the ChromaDB collection.

        model_name : str
            SentenceTransformer model used to generate embeddings.
        """

        self.db_path = str(db_path)
        self.collection_name = collection_name

        # Create the persistent vector database.
        self.chroma_client = chromadb.PersistentClient(
            path=self.db_path
        )

        # Load the embedding model.
        self.embedding_model = SentenceTransformer(
            model_name
        )

        # Create or reuse the collection.
        self.collection = (
            self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Corporate financial statements"
                }
            )
        )

        logger.info(
            "Vector store initialized. Collection: %s",
            self.collection_name
        )

    # --------------------------------------------------
    # VALIDATE AND PREPARE CHUNKS
    # --------------------------------------------------

    @staticmethod
    def _prepare_chunk(chunk, index):
        """
        Validate one text chunk and prepare its metadata.
        """

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {index} must be a dictionary."
            )

        text = chunk.get("text")
        metadata = chunk.get("metadata")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk {index} has missing or invalid text."
            )

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Chunk {index} must contain dictionary metadata."
            )

        # Standardize the company name.
        company = str(
            metadata.get("company", "")
        ).strip().lower()

        if not company:
            raise ValueError(
                f"Chunk {index} has no valid company name."
            )

        # Validate the year.
        try:
            year_value = float(metadata["year"])

            if not math.isfinite(year_value):
                raise ValueError

            if not year_value.is_integer():
                raise ValueError

            year = int(year_value)

        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError(
                f"Chunk {index} has an invalid or missing year."
            ) from None

        # Validate the row index.
        try:
            row_value = float(
                metadata.get("row_index", index)
            )

            if not math.isfinite(row_value):
                raise ValueError

            if not row_value.is_integer():
                raise ValueError

            row_index = int(row_value)

            if row_index < 0:
                raise ValueError

        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"Chunk {index} has an invalid row_index."
            ) from None

        # Keep ChromaDB metadata values to supported scalar types.
        clean_metadata = {
            "company": company,
            "year": year,
            "row_index": row_index
        }

        # Preserve additional metadata where possible.
        for key, value in metadata.items():

            if key in {"company", "year", "row_index"}:
                continue

            if not isinstance(key, str) or not key.strip():
                continue

            # Convert NumPy scalar values to Python scalar values.
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except (ValueError, TypeError):
                    continue

            if value is None or pd.isna(value):
                continue

            if isinstance(value, (str, int, float, bool)):

                # ChromaDB metadata must not contain infinite floats.
                if isinstance(value, float) and not math.isfinite(value):
                    continue

                clean_metadata[key] = value

        # Stable ID for each company/year/row combination.
        # Hashing the company prevents problematic characters in IDs.
        company_hash = hashlib.sha256(
            company.encode("utf-8")
        ).hexdigest()[:16]

        chunk_id = (
            f"{company_hash}_{year}_{row_index}"
        )

        return chunk_id, text.strip(), clean_metadata

    # --------------------------------------------------
    # INDEX FINANCIAL CHUNKS
    # --------------------------------------------------

    def index_financial_chunks(
        self,
        structured_chunks: list,
        batch_size: int = 128
    ):
        """
        Generate embeddings and store financial chunks in ChromaDB.

        Existing IDs are updated using upsert.

        Parameters
        ----------
        structured_chunks : list
            Chunks generated by FinancialDataPipeline.

        batch_size : int
            Number of chunks processed per batch.
        """

        if not structured_chunks:
            logger.warning(
                "No financial chunks available to index."
            )
            return

        if not isinstance(structured_chunks, list):
            raise TypeError(
                "structured_chunks must be a list."
            )

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError(
                "batch_size must be a positive integer."
            )

        ids = []
        documents = []
        metadatas = []

        # Validate every chunk before writing to the database.
        for index, chunk in enumerate(structured_chunks):

            chunk_id, text, metadata = self._prepare_chunk(
                chunk,
                index
            )

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(metadata)

        # Duplicate IDs in a single upsert can cause problems.
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate chunk IDs found. Each company/year/row_index "
                "combination must be unique."
            )

        total_chunks = len(documents)

        # Process records in batches to control memory usage.
        for start in range(0, total_chunks, batch_size):

            end = min(
                start + batch_size,
                total_chunks
            )

            batch_documents = documents[start:end]
            batch_ids = ids[start:end]
            batch_metadatas = metadatas[start:end]

            # Generate vector embeddings.
            embeddings = self.embedding_model.encode(
                batch_documents,
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()

            # Insert new records or update existing records.
            self.collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

            logger.info(
                "Indexed chunks %d to %d of %d",
                start + 1,
                end,
                total_chunks
            )

        logger.info(
            "Successfully indexed %d financial chunks.",
            total_chunks
        )

        logger.info(
            "Total documents in collection: %d",
            self.collection.count()
        )

    # --------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------

    def query_semantic_context(
        self,
        user_query: str,
        target_company: str,
        n_results: int = 2
    ) -> list:
        """
        Retrieve the most relevant financial documents
        for a particular company.

        Returns
        -------
        list[str]
            Matching financial text documents.
        """

        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError(
                "user_query must be a non-empty string."
            )

        target_company = str(
            target_company
        ).strip().lower()

        if not target_company:
            raise ValueError(
                "target_company must not be empty."
            )

        if (
            isinstance(n_results, bool)
            or not isinstance(n_results, int)
            or n_results < 1
        ):
            raise ValueError(
                "n_results must be a positive integer."
            )

        # Check whether the database contains any documents.
        total_documents = self.collection.count()

        if total_documents == 0:
            logger.warning(
                "The vector database contains no documents."
            )
            return []

        # Convert the question into an embedding.
        query_embedding = self.embedding_model.encode(
            [user_query.strip()],
            convert_to_numpy=True
        ).tolist()

        # Restrict results to the requested company.
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, total_documents),
            where={
                "company": target_company
            },
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results.get("documents") or []

        if not documents or not documents[0]:
            logger.info(
                "No matching documents found for company: %s",
                target_company
            )
            return []

        return documents[0]

    # --------------------------------------------------
    # SEMANTIC SEARCH WITH METADATA
    # --------------------------------------------------

    def query_semantic_results(
        self,
        user_query: str,
        target_company: str,
        n_results: int = 2
    ) -> list:
        """
        Retrieve relevant financial documents with metadata
        and similarity distances.

        Useful for displaying results in Streamlit.
        """

        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError(
                "user_query must be a non-empty string."
            )

        target_company = str(
            target_company
        ).strip().lower()

        if not target_company:
            raise ValueError(
                "target_company must not be empty."
            )

        if (
            isinstance(n_results, bool)
            or not isinstance(n_results, int)
            or n_results < 1
        ):
            raise ValueError(
                "n_results must be a positive integer."
            )

        total_documents = self.collection.count()

        if total_documents == 0:
            return []

        query_embedding = self.embedding_model.encode(
            [user_query.strip()],
            convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, total_documents),
            where={
                "company": target_company
            },
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        if not documents or not documents[0]:
            return []

        output = []

        for index, document in enumerate(documents[0]):

            metadata = (
                metadatas[0][index]
                if metadatas and metadatas[0]
                else {}
            )

            distance = (
                distances[0][index]
                if distances and distances[0]
                else None
            )

            output.append({
                "text": document,
                "metadata": metadata,
                "distance": distance
            })

        return output

    # --------------------------------------------------
    # GET COLLECTION STATISTICS
    # --------------------------------------------------

    def get_collection_stats(self):
        """Return basic information about the vector database."""

        return {
            "collection_name": self.collection_name,
            "database_path": self.db_path,
            "total_documents": self.collection.count()
        }


# --------------------------------------------------
# RUN THIS FILE DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    from financial_data_pipepline import (
        FinancialDataPipeline,
        download_kaggle_dataset
    )

    try:
        # Prefer a local CSV when available.
        local_csv = Path("financial_statements.csv")

        if local_csv.exists():
            csv_path = local_csv
        else:
            # Otherwise, download the Kaggle dataset.
            csv_path = download_kaggle_dataset()

        # Step 1: Initialize the financial data pipeline.
        pipeline = FinancialDataPipeline(
            csv_filepath=csv_path
        )

        # Step 2: Load and clean the data FIRST.
        cleaned_df = pipeline.load_and_clean()

        print("\n--- Cleaned Financial Data ---")
        print(cleaned_df.head())

        # Step 3: Generate text chunks.
        chunks = pipeline.generate_textual_chunks()

        print("\nTotal chunks generated:", len(chunks))

        # Step 4: Initialize the vector store.
        vector_store = FinancialVectorStore(
            db_path="./chroma_db"
        )

        # Step 5: Index the chunks.
        vector_store.index_financial_chunks(
            chunks
        )

        # Step 6: Search the indexed data.
        if not cleaned_df.empty:

            company_name = cleaned_df["company"].iloc[0]

            matched = vector_store.query_semantic_context(
                user_query="What was the revenue and net income?",
                target_company=company_name,
                n_results=3
            )

            print(
                f"\nRetrieved financial context for {company_name}:"
            )

            for document in matched:
                print("\n", document)

        # Step 7: Display database statistics.
        print("\n--- Vector Database Statistics ---")
        print(vector_store.get_collection_stats())

    except Exception as error:
        logger.exception(
            "Financial vector indexing failed: %s",
            error
        )