#!/usr/bin/python

# This script creates an interactive plot of
# sales forecasts by company
#
# 2019 - Jaime Lopez <jailop AT gmail DOT com>

import pandas as pd
from bokeh.models import ColumnDataSource, DataRange1d, Select
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column, row
from bokeh.io import curdoc
from datetime import datetime

def dataset(source, company):
    """
    This functions filters the dataset by company
    and returns a ColumnDataSource object
    """
    df = source[source.company == company].copy()
    return ColumnDataSource(df)

def make_plot(source, title):
    """
    Main function to plot data
    """
    plot = figure(x_axis_type="datetime", plot_width=800)
    plot.title.text = title
    plot.scatter(x='ds', y='sales', color='blue', source=source)
    plot.line(x='ds', y='sales', color='green', line_width=0.3, source=source)
    plot.yaxis.axis_label = "Predicted Sales (USD)"
    return plot

def update_plot(attr, old, new):
    plot.title.text = new
    company = company_select.value
    src = dataset(df, company)
    source.data.update(src.data)

# Data preparation
df = pd.read_csv('salesvalues.csv')
df['ds'] = df.ds.apply(datetime.fromisoformat)
df['company'] = df.company.apply(str)
# Model preparation
company = df.company.values[0]
company_select = Select(value=company, title='company',
        options=list(df.company.sort_values().unique()))
source = dataset(df, company)
# Plotting data
plot = make_plot(source, company)
# Setting interaction
company_select.on_change('value', update_plot)
# Display output
controls = column(company_select)
curdoc().add_root(row(plot, controls))
curdoc().title = 'Forecast'
