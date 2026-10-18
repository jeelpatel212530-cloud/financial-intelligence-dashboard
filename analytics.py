import logging
import re

import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FINANCIAL FORECASTER
# --------------------------------------------------

class FinancialForecaster:

    def __init__(self, cleaned_dataframe: pd.DataFrame):
        """
        Initialize the forecaster with cleaned financial data.
        """

        if not isinstance(cleaned_dataframe, pd.DataFrame):
            raise TypeError(
                "cleaned_dataframe must be a pandas DataFrame."
            )

        if cleaned_dataframe.empty:
            raise ValueError(
                "The financial DataFrame is empty."
            )

        self.df = cleaned_dataframe.copy()

        # Use column-name normalization compatible with the pipeline.
        self.df.columns = [
            re.sub(
                r"[^a-z0-9]+",
                "_",
                str(col).strip().lower()
            ).strip("_")
            for col in self.df.columns
        ]

        # Check for duplicate column names.
        if self.df.columns.duplicated().any():
            raise ValueError(
                "Duplicate column names found after normalization."
            )

        # Required columns.
        required_columns = {"company", "year"}

        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}. "
                f"Available columns: {self.df.columns.tolist()}"
            )

        # Normalize company names.
        self.df["company"] = (
            self.df["company"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # Convert years to numeric.
        self.df["year"] = pd.to_numeric(
            self.df["year"],
            errors="coerce"
        )

        # Remove rows without a valid company or year.
        self.df = self.df.dropna(
            subset=["company", "year"]
        ).copy()

        # Keep only whole-number years.
        self.df = self.df[
            self.df["year"] % 1 == 0
        ].copy()

        self.df["year"] = self.df["year"].astype(int)

        # Remove empty company names.
        self.df = self.df[
            self.df["company"].str.len() > 0
        ].copy()

        if self.df.empty:
            raise ValueError(
                "No valid company/year records remain."
            )

        logger.info(
            "FinancialForecaster initialized with %d records.",
            len(self.df)
        )

    # --------------------------------------------------
    # FORECAST A FINANCIAL METRIC
    # --------------------------------------------------

    def forecast_metric(
        self,
        company_name: str,
        target_metric: str,
        periods_ahead: int = 2
    ) -> tuple:
        """
        Forecast a financial metric for one company.

        Returns
        -------
        tuple
            (latest_prediction, plotly_figure)

            Returns (None, None) when insufficient valid
            historical observations are available.
        """

        # Validate company name.
        if not isinstance(company_name, str) or not company_name.strip():
            raise ValueError(
                "company_name must be a non-empty string."
            )

        company_name = company_name.strip().lower()

        # Validate the forecast horizon.
        if (
            isinstance(periods_ahead, bool)
            or not isinstance(periods_ahead, int)
            or periods_ahead < 1
        ):
            raise ValueError(
                "periods_ahead must be a positive integer."
            )

        # Normalize the metric name consistently.
        target_metric = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(target_metric).strip().lower()
        ).strip("_")

        if not target_metric:
            raise ValueError(
                "target_metric must not be empty."
            )

        if target_metric not in self.df.columns:
            raise ValueError(
                f"Metric '{target_metric}' not found. "
                f"Available columns: {self.df.columns.tolist()}"
            )

        # Select records for the requested company.
        comp_df = self.df[
            self.df["company"] == company_name
        ].copy()

        if comp_df.empty:
            logger.warning(
                "No records found for company: %s",
                company_name
            )
            return None, None

        # Convert the selected financial metric to numeric.
        comp_df["y"] = pd.to_numeric(
            comp_df[target_metric],
            errors="coerce"
        )

        # Remove missing and infinite metric values.
        comp_df = comp_df[
            comp_df["y"].notna()
            & comp_df["y"].map(
                lambda value: pd.notna(value)
                and abs(float(value)) != float("inf")
            )
        ].copy()

        # Create annual dates.
        comp_df["ds"] = pd.to_datetime(
            comp_df["year"].astype(str) + "-12-31",
            errors="coerce"
        )

        comp_df = comp_df.dropna(
            subset=["ds", "y"]
        )

        # If multiple records exist for a company/year,
        # aggregate them into one annual observation.
        comp_df = (
            comp_df.groupby("ds", as_index=False)["y"]
            .mean()
            .sort_values("ds")
            .reset_index(drop=True)
        )

        if len(comp_df) < 2:
            logger.warning(
                "Insufficient valid history for %s / %s.",
                company_name,
                target_metric
            )
            return None, None

        if len(comp_df) < 5:
            logger.warning(
                "Only %d annual observations are available for %s. "
                "The forecast may be unreliable.",
                len(comp_df),
                company_name
            )

        # Prepare Prophet input.
        prophet_df = comp_df[["ds", "y"]].copy()

        # Initialize the forecasting model.
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=1000
        )

        # Train the model.
        model.fit(prophet_df)

        # Generate future year-end dates.
        future = model.make_future_dataframe(
            periods=periods_ahead,
            freq="YE"
        )

        # Generate forecasts.
        forecast = model.predict(future)

        # Keep only dates after the final historical observation.
        last_historical_date = comp_df["ds"].max()

        future_forecast = forecast[
            forecast["ds"] > last_historical_date
        ].copy()

        if future_forecast.empty:
            logger.warning(
                "The model did not produce future predictions."
            )
            return None, None

        # Last forecasted value.
        latest_row = future_forecast.iloc[-1]

        latest_prediction = float(
            latest_row["yhat"]
        )

        forecast_year = int(
            latest_row["ds"].year
        )

        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------

        fig = go.Figure()

        # Historical actual values.
        fig.add_trace(
            go.Scatter(
                x=comp_df["ds"],
                y=comp_df["y"],
                name="Historical Actual",
                mode="lines+markers"
            )
        )

        # Future forecast line.
        fig.add_trace(
            go.Scatter(
                x=future_forecast["ds"],
                y=future_forecast["yhat"],
                name="Forecast",
                mode="lines+markers",
                line=dict(dash="dash")
            )
        )

        # Forecast uncertainty interval.
        fig.add_trace(
            go.Scatter(
                x=future_forecast["ds"],
                y=future_forecast["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future_forecast["ds"],
                y=future_forecast["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="Forecast Uncertainty",
                hoverinfo="skip"
            )
        )

        fig.update_layout(
            title=(
                f"{company_name.title()} - "
                f"{target_metric.replace('_', ' ').title()} Forecast"
            ),
            xaxis_title="Year",
            yaxis_title=target_metric.replace("_", " ").title(),
            template="plotly_white",
            hovermode="x unified"
        )

        # Add a vertical marker between history and forecast.
        fig.add_vline(
            x=last_historical_date.timestamp() * 1000,
            line_dash="dot",
            annotation_text="Forecast Starts"
        )

        logger.info(
            "Forecast generated for %s / %s through %d.",
            company_name,
            target_metric,
            forecast_year
        )

        return latest_prediction, fig


# --------------------------------------------------
# RUN THIS FILE DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path
    from financial_data_pipepline import (
        FinancialDataPipeline,
        download_kaggle_dataset
    )

    try:
        # Prefer the local CSV if it exists.
        csv_path = Path("financial_statements.csv")

        if not csv_path.exists():
            csv_path = download_kaggle_dataset()

        # Load and clean financial data.
        pipeline = FinancialDataPipeline(
            csv_filepath=csv_path
        )

        cleaned_df = pipeline.load_and_clean()

        print("\n--- Cleaned Financial Data ---")
        print(cleaned_df.head())

        # Initialize forecaster.
        forecaster = FinancialForecaster(
            cleaned_df
        )

        # Select a company that actually exists in the data.
        company_name = cleaned_df["company"].iloc[0]

        # Forecast revenue if that metric exists.
        if "revenue" not in cleaned_df.columns:
            raise ValueError(
                "The dataset does not contain a 'revenue' column. "
                f"Available columns: {cleaned_df.columns.tolist()}"
            )

        prediction, chart = forecaster.forecast_metric(
            company_name=company_name,
            target_metric="revenue",
            periods_ahead=2
        )

        if prediction is None:
            print(
                "Insufficient valid financial history for forecasting."
            )
        else:
            print(
                f"\nForecasted revenue for {company_name.title()}: "
                f"{prediction:,.2f}"
            )

            chart.show()

    except Exception as error:
        logger.exception(
            "Financial forecasting failed: %s",
            error
        )