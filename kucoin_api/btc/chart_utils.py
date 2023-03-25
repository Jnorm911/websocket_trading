from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource


def plot_candlestick_chart_bokeh(df):
    inc = df.color == "green"
    dec = df.color == "red"

    source_inc = ColumnDataSource(df.loc[inc])
    source_dec = ColumnDataSource(df.loc[dec])

    p = figure(
        x_axis_type="datetime", width=1000, height=300, sizing_mode="scale_width"
    )
    p.xaxis.major_label_orientation = 3.14 / 4
    p.grid.grid_line_alpha = 0.3

    w = (
        0.9 * 60 * 1000
    )  # width of a candle in milliseconds, adjusted to 90% of 1 minute

    p.segment(
        "start_time",
        "ha_high",
        "start_time",
        "ha_low",
        color="black",
        source=source_inc,
    )
    p.segment(
        "start_time",
        "ha_high",
        "start_time",
        "ha_low",
        color="black",
        source=source_dec,
    )
    p.vbar(
        "start_time",
        w,
        "ha_open",
        "ha_close",
        fill_color="green",
        line_color="black",
        source=source_dec,
    )

    show(p)
