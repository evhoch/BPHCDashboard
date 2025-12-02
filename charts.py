# charts.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def build_forecast_figure(
    plot_df: pd.DataFrame,
    shelter_name: str,
    target_col: str,  # we still accept it, even if we don't need it inside
) -> go.Figure:
    hist_plot = plot_df[plot_df["type"] == "Historical"]
    fc_plot   = plot_df[plot_df["type"] == "Forecast"]

    fig = go.Figure()

    # Historical: will be men or women depending on target_col used upstream
    fig.add_trace(
        go.Scatter(
            x=hist_plot["date"],
            y=hist_plot["Shelter Guests"],
            mode="lines+markers",
            name="Historical",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Series: Historical<br>"
                "Shelter Guests: %{y:.1f}<extra></extra>"
            ),
        )
    )

    # Forecast with error bars
    fig.add_trace(
        go.Scatter(
            x=fc_plot["date"],
            y=fc_plot["Shelter Guests"],
            mode="lines+markers",
            name="Forecast",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(fc_plot["upper"] - fc_plot["Shelter Guests"]),
                arrayminus=(fc_plot["Shelter Guests"] - fc_plot["lower"]),
                visible=True,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Series: Forecast<br>"
                "Shelter Guests: %{y:.1f}<br>"
                "Lower: %{customdata[0]:.1f}<br>"
                "Upper: %{customdata[1]:.1f}<extra></extra>"
            ),
            customdata=np.stack(
                [fc_plot["lower"].values, fc_plot["upper"].values],
                axis=-1,
            ),
        )
    )

    fig.update_layout(
        title=f"Forecast & Historical Shelter Guests — {shelter_name}",
        height=400,
        xaxis_title="Date",
        yaxis_title="Shelter Guests",
        yaxis=dict(rangemode="tozero"),
        legend_title_text="Series",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig
