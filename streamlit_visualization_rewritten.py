"""
Financial Intelligence Dashboard
Run: streamlit run streamlit_visualization_improved.py
"""

import hashlib
import logging
import os
from io import BytesIO

import pandas as pd
import streamlit as st

from financial_data_pipepline import FinancialDataPipeline
from vector_indexing import FinancialVectorStore
from analytics import FinancialForecaster
from rag import FinancialRAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Financial Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Financial Intelligence Dashboard")
st.write(
    "Explore corporate financial statements with semantic search, "
    "financial forecasting, and grounded AI insights."
)


@st.cache_data(show_spinner=False)
def load_and_prepare_csv(csv_bytes: bytes):
    """Use FinancialDataPipeline for all cleaning and chunk creation."""
    if not csv_bytes:
        raise ValueError("The uploaded CSV file is empty.")

    raw_df = None
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            raw_df = pd.read_csv(BytesIO(csv_bytes), encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")

    if raw_df is None:
        raise ValueError("Could not parse the CSV. " + " | ".join(errors))
    if raw_df.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    pipeline = FinancialDataPipeline(dataframe=raw_df)
    cleaned_df = pipeline.load_and_clean()
    if cleaned_df is None or cleaned_df.empty:
        raise ValueError(
            "No valid records remain after cleaning. Check that the CSV has "
            "company and year columns and valid rows."
        )

    chunks = pipeline.generate_textual_chunks()
    if not chunks:
        raise ValueError("No searchable financial text chunks were created.")
    return cleaned_df, chunks


@st.cache_resource(show_spinner=False)
def get_vector_store(dataset_id: str):
    """Use a separate persistent Chroma collection for each uploaded dataset."""
    return FinancialVectorStore(
        db_path="./chroma_db",
        collection_name=f"financial_{dataset_id[:20]}",
    )


with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader(
        "Upload Corporate Statements (CSV)",
        type=["csv"],
        help="CSV should include company and year columns plus financial metrics.",
        key="financial_csv_uploader",
    )
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Optional if OPENAI_API_KEY is set in your environment.",
    )
    st.caption("Your API key is not displayed.")
    st.divider()
    st.caption("Upload a CSV to begin.")

# This guard must remain immediately before getvalue().
if uploaded_file is None:
    st.info("Please upload your financial statements CSV using the sidebar.")
    st.stop()

csv_bytes = uploaded_file.getvalue()
if not csv_bytes:
    st.error("The uploaded CSV file is empty.")
    st.stop()

dataset_id = hashlib.sha256(csv_bytes).hexdigest()

try:
    with st.spinner("Loading and cleaning financial data..."):
        cleaned_df, chunks = load_and_prepare_csv(csv_bytes)
except Exception as exc:
    st.error(f"Could not prepare the uploaded CSV: {exc}")
    st.stop()

required = {"company", "year"}
missing = required.difference(cleaned_df.columns)
if missing:
    st.error("Missing required cleaned column(s): " + ", ".join(sorted(missing)))
    st.stop()

companies_series = cleaned_df["company"].dropna().astype(str).str.strip()
companies = sorted(companies_series[companies_series != ""].unique().tolist())
if not companies:
    st.error("No valid company names were found in the uploaded CSV.")
    st.stop()

metric_columns = [
    col for col in cleaned_df.columns
    if col not in {"company", "year", "row_index"}
    and pd.api.types.is_numeric_dtype(cleaned_df[col])
]
for col in cleaned_df.columns:
    if col in {"company", "year", "row_index"} or col in metric_columns:
        continue
    converted = pd.to_numeric(cleaned_df[col], errors="coerce")
    if converted.notna().any():
        cleaned_df[col] = converted
        metric_columns.append(col)

st.subheader("Financial Data Overview")
a, b, c, d = st.columns(4)
a.metric("Records", f"{len(cleaned_df):,}")
b.metric("Companies", f"{len(companies):,}")
c.metric("Financial chunks", f"{len(chunks):,}")
d.metric("Numeric metrics", f"{len(metric_columns):,}")

with st.expander("Preview cleaned financial data"):
    st.dataframe(cleaned_df, use_container_width=True, hide_index=True)

st.subheader("Analysis Configuration")
left, right = st.columns(2)
with left:
    selected_company = st.selectbox("Select Company", companies)
with right:
    selected_metric = (
        st.selectbox("Select Financial Metric", metric_columns)
        if metric_columns else None
    )

user_query = st.text_area(
    "Ask a financial question",
    placeholder="Example: Summarize revenue trends, profitability, and financial outlook.",
    height=100,
)
periods_ahead = st.slider(
    "Forecast horizon (years)", min_value=1, max_value=10, value=2
)
run_analysis = st.button(
    "Run Financial Analysis", type="primary", use_container_width=True
)

if run_analysis:
    if not user_query.strip():
        st.warning("Please enter a financial question.")
        st.stop()

    effective_api_key = (api_key_input or os.getenv("OPENAI_API_KEY", "")).strip()
    if not effective_api_key:
        st.warning(
            "OpenAI API key is missing. Enter it in the sidebar or set "
            "OPENAI_API_KEY in your environment."
        )
        st.stop()

    st.subheader("Financial Analysis Results")
    semantic_context = []
    forecast_value = None

    # 1. Semantic indexing and retrieval.
    try:
        vector_store = get_vector_store(dataset_id)
        with st.spinner("Indexing financial statements..."):
            vector_store.index_financial_chunks(chunks)

        with st.spinner("Retrieving relevant financial context..."):
            semantic_context = vector_store.query_semantic_context(
                user_query=user_query.strip(),
                target_company=selected_company,
                n_results=5,
            )

        if semantic_context:
            st.success(
                f"Retrieved {len(semantic_context)} relevant record(s) "
                f"for {selected_company}."
            )
            with st.expander("Retrieved Financial Context"):
                for i, document in enumerate(semantic_context, start=1):
                    st.markdown(f"**Record {i}**")
                    st.write(document)
        else:
            st.warning(
                "No relevant context was returned for this company. "
                "Check company names and vector-store metadata."
            )
    except Exception as exc:
        logger.exception("Semantic indexing or retrieval failed")
        st.error(f"Semantic search failed: {exc}")
        st.stop()

    # 2. Forecast selected metric, if possible.
    if selected_metric:
        try:
            company_rows = cleaned_df[
                cleaned_df["company"].astype(str).str.strip() == selected_company
            ]
            valid_count = pd.to_numeric(
                company_rows[selected_metric], errors="coerce"
            ).notna().sum()

            if valid_count < 2:
                st.warning(
                    f"Forecast skipped: {selected_company} has fewer than "
                    f"two valid observations for {selected_metric}."
                )
            else:
                with st.spinner(f"Forecasting {selected_metric}..."):
                    forecaster = FinancialForecaster(cleaned_df)
                    forecast_value, forecast_figure = forecaster.forecast_metric(
                        company_name=selected_company,
                        target_metric=selected_metric,
                        periods_ahead=periods_ahead,
                    )

                if forecast_value is not None:
                    st.metric(
                        f"Forecasted {selected_metric.replace('_', ' ').title()}",
                        f"{forecast_value:,.2f}",
                    )
                else:
                    st.warning("The forecasting module did not return a prediction.")

                if forecast_figure is not None:
                    st.plotly_chart(forecast_figure, use_container_width=True)
        except Exception as exc:
            logger.exception("Forecasting failed")
            st.warning(f"Forecasting failed; continuing with available context. {exc}")
            forecast_value = None
    else:
        st.info("Forecasting is unavailable because no numeric metric was detected.")

    # 3. Grounded AI insight.
    try:
        with st.spinner("Generating AI-powered financial insights..."):
            rag_engine = FinancialRAGEngine(api_key=effective_api_key)
            insight = rag_engine.generate_grounded_insight(
                query=user_query.strip(),
                context_documents=semantic_context,
                forecast_value=forecast_value,
            )
        st.subheader("AI Financial Insights")
        st.markdown(insight)
    except Exception as exc:
        logger.exception("AI insight generation failed")
        st.error(f"Could not generate financial insights: {exc}")

st.divider()
st.caption(
    "Financial Intelligence Dashboard | Semantic Retrieval • Forecasting • RAG"
)
