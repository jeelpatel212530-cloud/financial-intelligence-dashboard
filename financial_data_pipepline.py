import logging
import re
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------
import kagglehub

from pathlib import Path


def download_kaggle_dataset():
    """Download the dataset and return the path to a CSV file."""

    dataset_path = kagglehub.dataset_download(
        "rish59/financial-statements-of-major-companies2009-2023"
    )

    dataset_directory = Path(dataset_path)

    csv_files = list(dataset_directory.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {dataset_directory}"
        )

    print("CSV files found:")

    for file in csv_files:
        print(file)

    # Select the first CSV file.
    # Verify the filename if multiple CSV files are listed.
    csv_path = csv_files[0]

    print("\nUsing CSV:", csv_path)

    return csv_path
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FINANCIAL DATA PIPELINE
# --------------------------------------------------

class FinancialDataPipeline:

    def __init__(self, csv_filepath=None, dataframe=None):
        """
        Initialize the financial data pipeline.

        Parameters
        ----------
        csv_filepath : str or Path, optional
            Path to a local CSV file.

        dataframe : pandas.DataFrame, optional
            An already loaded DataFrame.
        """

        self.file_path = (
            Path(csv_filepath) if csv_filepath is not None else None
        )

        self.df = dataframe.copy() if dataframe is not None else None

        # Store generated text chunks.
        self.chunks = []

    # --------------------------------------------------
    # LOAD CSV DATA
    # --------------------------------------------------

    def _read_csv(self):
        """Read the CSV file using common text encodings."""

        if self.file_path is None:
            raise ValueError(
                "No CSV file path provided. "
                "Initialize with FinancialDataPipeline('file.csv')."
            )

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise ValueError(
                f"The provided path is not a file: {self.file_path}"
            )

        encodings = ["utf-8-sig", "utf-8", "latin1"]

        last_error = None

        for encoding in encodings:
            try:
                df = pd.read_csv(
                    self.file_path,
                    encoding=encoding,
                    low_memory=False
                )

                if df.empty:
                    raise ValueError(
                        f"The CSV file is empty: {self.file_path}"
                    )

                logger.info("Loaded CSV: %s", self.file_path)

                return df

            except UnicodeDecodeError as error:
                last_error = error

        raise ValueError(
            f"Could not read the CSV file: {self.file_path}"
        ) from last_error

    # --------------------------------------------------
    # CLEAN COLUMN NAMES
    # --------------------------------------------------

    @staticmethod
    def _clean_column_names(df):
        """Standardize column names."""

        df = df.copy()

        cleaned_columns = []

        for column in df.columns:
            column = str(column).strip().lower()

            # Replace spaces and special characters with underscores.
            column = re.sub(r"[^a-z0-9]+", "_", column)

            column = column.strip("_")

            cleaned_columns.append(column)

        df.columns = cleaned_columns

        # Prevent duplicate column names.
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()

            raise ValueError(
                f"Duplicate column names found after cleaning: {duplicates}"
            )

        # Support common alternatives for required columns.
        aliases = {
            "company_name": "company",
            "companyname": "company",
            "company_id": "company",
            "fiscal_year": "year",
            "financial_year": "year",
            "fiscalyear": "year"
        }

        df = df.rename(
            columns={
                old_name: new_name
                for old_name, new_name in aliases.items()
                if old_name in df.columns and new_name not in df.columns
            }
        )

        return df

    # --------------------------------------------------
    # CLEAN NUMERIC VALUES
    # --------------------------------------------------

    @staticmethod
    def _convert_numeric_column(series):
        """
        Convert a column to numeric only when its values
        appear predominantly numeric.

        Handles values such as:
        $1,200
        1,200
        25%
        (500)
        """

        if pd.api.types.is_numeric_dtype(series):
            return series

        cleaned = (
            series.astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
            .str.replace("%", "", regex=False)
        )

        # Convert accounting-style negative values: (500) -> -500.
        cleaned = cleaned.str.replace(
            r"^\((.*)\)$",
            r"-\1",
            regex=True
        )

        cleaned = cleaned.replace(
            {
                "": pd.NA,
                "NA": pd.NA,
                "N/A": pd.NA,
                "null": pd.NA,
                "None": pd.NA,
                "nan": pd.NA,
                "-": pd.NA
            }
        )

        numeric = pd.to_numeric(cleaned, errors="coerce")

        # Avoid accidentally converting categorical columns into
        # columns full of missing values.
        non_missing_count = series.notna().sum()

        if non_missing_count == 0:
            return series

        numeric_count = numeric.notna().sum()

        numeric_ratio = numeric_count / non_missing_count

        if numeric_ratio >= 0.70:
            return numeric

        return series

    # --------------------------------------------------
    # LOAD AND CLEAN DATA
    # --------------------------------------------------

    def load_and_clean(self):
        """
        Load and clean financial data.

        Returns
        -------
        pandas.DataFrame
            Cleaned financial data.
        """

        # Load data from a CSV if a DataFrame was not supplied.
        if self.df is None:
            self.df = self._read_csv()

        df = self.df.copy()

        if df.empty:
            raise ValueError("The input DataFrame contains no records.")

        # Remove completely empty rows.
        df = df.dropna(how="all")

        if df.empty:
            raise ValueError(
                "No records remain after removing empty rows."
            )

        # Clean column names.
        df = self._clean_column_names(df)

        # Required columns.
        required_columns = {"company", "year"}

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Clean company names.
        df["company"] = (
            df["company"]
            .astype("string")
            .str.strip()
        )

        # Convert year to numeric.
        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce"
        )

        # Remove records without a valid company or year.
        df = df.dropna(
            subset=["company", "year"]
        ).copy()

        df = df[
            df["company"].str.len() > 0
        ].copy()

        # Reject fractional years.
        valid_years = df["year"] % 1 == 0

        df = df[valid_years].copy()

        if df.empty:
            raise ValueError(
                "No valid records remain after cleaning company names "
                "and years."
            )

        df["year"] = df["year"].astype(int)

        # Convert financial metric columns when appropriate.
        for column in df.columns:

            if column in ["company", "year"]:
                continue

            df[column] = self._convert_numeric_column(df[column])

        # Remove exact duplicate records.
        df = df.drop_duplicates()

        # Sort records by company and year.
        df = df.sort_values(
            by=["company", "year"]
        ).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "The cleaned DataFrame contains no records."
            )

        self.df = df

        # Reset previously generated chunks after reloading.
        self.chunks = []

        logger.info(
            "Data cleaning completed. Records: %d, Columns: %d",
            len(self.df),
            len(self.df.columns)
        )

        return self.df

    # --------------------------------------------------
    # GENERATE TEXTUAL CHUNKS
    # --------------------------------------------------

    def generate_textual_chunks(self):
        """
        Convert financial records into text chunks for
        vector indexing and retrieval-augmented generation (RAG).

        Returns
        -------
        list
            A list of dictionaries containing text and metadata.
        """

        if self.df is None:
            raise ValueError(
                "DataFrame is not loaded. "
                "Call load_and_clean() first."
            )

        chunks = []

        for index, row in self.df.iterrows():

            company = str(row["company"])
            year = int(row["year"])

            metrics = []

            for column, value in row.items():

                if column in ["company", "year"]:
                    continue

                # Ignore missing metric values.
                if pd.isna(value):
                    continue

                # Make column names human-readable.
                metric_name = column.replace("_", " ").title()

                # Format numeric values consistently.
                if isinstance(value, float):
                    metric_value = f"{value:,.2f}"
                else:
                    metric_value = str(value)

                metrics.append(
                    f"{metric_name}: {metric_value}"
                )

            if metrics:
                metrics_summary = "; ".join(metrics)
            else:
                metrics_summary = "No financial metrics available"

            chunk_text = (
                f"Financial statement for {company} "
                f"for the year {year}. "
                f"{metrics_summary}."
            )

            metadata = {
                "company": company,
                "year": year,
                "row_index": int(index)
            }

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata
                }
            )

        self.chunks = chunks

        logger.info(
            "Generated %d textual chunks.",
            len(self.chunks)
        )

        return self.chunks

    # --------------------------------------------------
    # GET CLEANED DATA
    # --------------------------------------------------

    def get_dataframe(self):
        """Return the cleaned DataFrame."""

        if self.df is None:
            raise ValueError(
                "DataFrame is not loaded. "
                "Call load_and_clean() first."
            )

        return self.df.copy()

    # --------------------------------------------------
    # GET CHUNKS
    # --------------------------------------------------

    def get_chunks(self):
        """Return generated text chunks."""

        if not self.chunks:
            raise ValueError(
                "No text chunks available. "
                "Call generate_textual_chunks() first."
            )

        return self.chunks.copy()


# --------------------------------------------------
# KAGGLE DATASET DOWNLOAD
# --------------------------------------------------

def download_kaggle_dataset():
    """
    Download the Kaggle financial statements dataset.

    Requires:
        pip install kagglehub

    Returns
    -------
    pathlib.Path
        Path to the selected CSV file.
    """

    try:
        import kagglehub

    except ImportError as error:
        raise ImportError(
            "KaggleHub is not installed. Run: pip install kagglehub"
        ) from error

    dataset_name = (
        "rish59/financial-statements-of-major-companies2009-2023"
    )

    logger.info("Downloading Kaggle dataset: %s", dataset_name)

    dataset_path = kagglehub.dataset_download(dataset_name)

    dataset_directory = Path(dataset_path)

    csv_files = sorted(
        dataset_directory.rglob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in the downloaded dataset: "
            f"{dataset_directory}"
        )

    # Prefer filenames that clearly describe the financial data.
    preferred_files = [
        path for path in csv_files
        if any(
            keyword in path.name.lower()
            for keyword in ["financial", "statement"]
        )
    ]

    candidates = preferred_files if preferred_files else csv_files

    if len(candidates) == 1:
        selected_file = candidates[0]

    else:
        # Select the largest candidate as a practical default.
        # Verify this choice if the dataset contains multiple CSVs.
        selected_file = max(
            candidates,
            key=lambda path: path.stat().st_size
        )

        logger.warning(
            "Multiple CSV files were found. Using the largest candidate: %s",
            selected_file
        )

        logger.info(
            "Available CSV files: %s",
            [str(path) for path in csv_files]
        )

    logger.info("Selected dataset file: %s", selected_file)

    return selected_file


# --------------------------------------------------
# RUN THIS FILE DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    try:
        # Download the Kaggle dataset.
        csv_path = download_kaggle_dataset()

        # Initialize the pipeline.
        pipeline = FinancialDataPipeline(
            csv_filepath=csv_path
        )

        # Load and clean the data.
        cleaned_df = pipeline.load_and_clean()

        # Generate text chunks for vector indexing.
        chunks = pipeline.generate_textual_chunks()

        # Display results.
        print("\n--- Cleaned Financial Data ---")
        print(cleaned_df.head())

        print("\n--- Dataset Information ---")
        print("Total records:", len(cleaned_df))
        print("Total columns:", len(cleaned_df.columns))
        print("Companies:", cleaned_df["company"].nunique())

        print("\n--- Generated Text Chunks ---")
        print("Total chunks:", len(chunks))

        if chunks:
            print("\nFirst text chunk:")
            print(chunks[0]["text"])

            print("\nFirst chunk metadata:")
            print(chunks[0]["metadata"])

    except Exception as error:
        logger.exception("Pipeline failed: %s", error)