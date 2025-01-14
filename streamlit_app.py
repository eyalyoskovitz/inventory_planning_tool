import streamlit as st
st.title("Inventory Planning Tool")
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import linprog
import plotly.express as px

def create_sample_data():
    """Create sample data for the app."""
    inventory_data = {
        "Material": ["Steel Rods", "Aluminum Sheets", "Brass Blocks", "Cutting Fluids"],
        "SKU": ["ST1001", "AL2002", "BR3003", "CF4004"],
        "Current Stock Quantity": [500, 200, 120, 50],
        "Reorder Level": [300, 100, 60, 20]
    }
    demand_data = {
        "Month": ["Month1", "Month2", "Month3"],
        "Steel Rods": [450, 480, 520],
        "Aluminum Sheets": [150, 170, 200],
        "Brass Blocks": [100, 110, 120],
        "Cutting Fluids": [40, 45, 50]
    }
    supplier_data = {
        "Material": ["Steel Rods", "Aluminum Sheets", "Brass Blocks", "Cutting Fluids"],
        "Supplier": ["SupplierA", "SupplierB", "SupplierC", "SupplierD"],
        "Lead Time Days": [10, 14, 7, 5],
        "Price Per Unit": [20, 15, 30, 10]
    }
    production_data = {
        "Job ID": ["J001", "J002", "J003"],
        "Material": ["Steel Rods", "Aluminum Sheets", "Brass Blocks"],
        "Required Quantity": [300, 100, 50],
        "Deadline": ["2025-01-15", "2025-01-20", "2025-01-25"]
    }

    return pd.DataFrame(inventory_data), pd.DataFrame(demand_data), pd.DataFrame(supplier_data), pd.DataFrame(production_data)

def main():
    st.title("Inventory Planning and Optimization Tool")

    # Step 1: Load Data
    st.sidebar.header("Step 1: Data Input")
    inventory_df, demand_df, supplier_df, production_df = create_sample_data()

    st.sidebar.subheader("Sample Data Loaded")
    if st.sidebar.button("Show Inventory Data"):
        st.write("### Inventory Data", inventory_df)
    if st.sidebar.button("Show Demand Data"):
        st.write("### Demand Data", demand_df)
    if st.sidebar.button("Show Supplier Data"):
        st.write("### Supplier Data", supplier_df)

    # Step 2: Set Priorities
    st.sidebar.header("Step 2: Set Priorities")
    priorities = {
        "Pricing": st.sidebar.slider("Pricing", 0.0, 1.0, 0.3),
        "Lead Time": st.sidebar.slider("Lead Time", 0.0, 1.0, 0.3),
        "Safety Stock": st.sidebar.slider("Safety Stock", 0.0, 1.0, 0.2),
        "Local Supplier": st.sidebar.slider("Local Supplier", 0.0, 1.0, 0.1),
        "Global Supplier": st.sidebar.slider("Global Supplier", 0.0, 1.0, 0.1),
    }

    # Forecasting Demand
    st.header("Demand Forecasting")
    demand_for_forecasting = demand_df.set_index("Month").T
    forecast_results = {}
    forecast_period = 3
    for material in demand_for_forecasting.index:
        model = ExponentialSmoothing(
            demand_for_forecasting.loc[material], trend="add", seasonal=None, initialization_method="estimated"
        ).fit()
        forecast = model.forecast(forecast_period)
        forecast_results[material] = forecast.values
    forecast_df = pd.DataFrame(forecast_results, index=["Month4", "Month5", "Month6"]).reset_index()
    st.write("### Forecasted Demand", forecast_df)

    # Consolidate Data
    consolidated_data = pd.merge(inventory_df, supplier_df, on="Material")
    consolidated_data["Total Forecasted Demand"] = consolidated_data["Material"].apply(
        lambda mat: sum(forecast_results.get(mat, []))
    )
    consolidated_data["Safety Stock"] = consolidated_data["Lead Time Days"] * consolidated_data["Total Forecasted Demand"] / 90

    # Optimization using SciPy
    st.header("Optimization Results")
    c = [
        (priorities["Pricing"] * row["Price Per Unit"] +
         priorities["Lead Time"] / row["Lead Time Days"] +
         priorities["Safety Stock"] * row["Safety Stock"])
        for _, row in consolidated_data.iterrows()
    ]
    A_eq = [[1 if i == j else 0 for j in range(len(consolidated_data))] for i in range(len(consolidated_data))]
    b_eq = consolidated_data["Total Forecasted Demand"] + consolidated_data["Safety Stock"]
    bounds = [(0, None) for _ in range(len(consolidated_data))]
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        consolidated_data["Optimal Order Quantity"] = result.x
        st.write("### Consolidated Data with Optimal Order Quantity", consolidated_data)
    else:
        st.error("Optimization failed. Please check your input data.")

    # Visualization
    fig = px.bar(consolidated_data, x="Material", y="Optimal Order Quantity", title="Optimal Order Quantities")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()

