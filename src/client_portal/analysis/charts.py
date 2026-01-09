from __future__ import annotations

import pandas as pd
import plotly.express as px


def value_over_time(df: pd.DataFrame):
    """Simple line chart: value by date."""
    if not {"date", "value"}.issubset(df.columns):
        raise ValueError("Expected columns: date, value")
    df2 = df.dropna(subset=["date", "value"]).sort_values("date")
    fig = px.line(df2, x="date", y="value", title="Value over time")
    return fig
