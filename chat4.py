import os
import io
import base64
import json
import pyttsx3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import streamlit as st
from fpdf import FPDF
import tempfile

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase

api_key= st.secrets["api"]["key"]

# Load environment and API key
load_dotenv()
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20241022",
    api_key=api_key
)

# Connect to database
sqlite_uri = 'sqlite:///database/mystorep.db'
db = SQLDatabase.from_uri(sqlite_uri)

# Get DB schema
def get_schema(_):
    return db.get_table_info()

# SQL generation chain
sql_template = """Based on the table schema below, write a SQL query without any explanation that would answer the user's question:
{schema}

Question: {question}
SQL Query:
"""
sql_prompt = ChatPromptTemplate.from_template(sql_template)
sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | sql_prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# Run SQL query
from sqlalchemy import text

def run_query(query):
    try:
        result = db.run(query)

        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                df = pd.DataFrame(result)
            else:
                # List of tuples — manually fetch column names
                with db._engine.connect() as conn:
                    cursor = conn.execute(text(query))
                    column_names = cursor.keys()
                df = pd.DataFrame(result, columns=column_names)

        elif isinstance(result, str):
            import ast
            parsed = ast.literal_eval(result)

            # Fetch column names from live cursor
            with db._engine.connect() as conn:
                cursor = conn.execute(text(query))
                column_names = cursor.keys()
            df = pd.DataFrame(parsed, columns=column_names)

        else:
            return None

        return df

    except Exception as e:
        st.error(f"Query failed: {e}")
        return None


# Generate explanation
def generate_text_summary(df, user_question):
    if df is None or df.empty:
        return "I couldn't generate a valid response."

    summary_template = """Here is the SQL query result:

{data}

Based on this result, provide a clear and concise explanation to answer the user's question: "{question}" in simple terms.
"""
    summary_prompt = ChatPromptTemplate.from_template(summary_template)
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({
        "data": df.to_string(),
        "question": user_question
    })

# Smart chart suggestion
def suggest_best_chart(df):
    import numpy as np
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()

    if datetime_cols and numeric_cols:
        return "Line Chart", datetime_cols[0], numeric_cols[0]
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.max().max()
        if max_corr > 0.7:
            pair = corr.unstack().sort_values(ascending=False).index[0]
            return "Scatter Plot", pair[0], pair[1]
    if numeric_cols and categorical_cols:
        return "Bar Chart", categorical_cols[0], numeric_cols[0]
    if len(numeric_cols) == 1:
        return "Histogram", numeric_cols[0], None
    if categorical_cols:
        return "Count Plot", categorical_cols[0], None
    return "Table", None, None

# Chart type explanations
CHART_EXPLAINERS = {
    "Table": "Displays the raw data in tabular form.",
    "Bar Chart": "Compares values across categories using bars.",
    "Line Chart": "Shows trends over time or ordered data using lines.",
    "Scatter Plot": "Visualizes relationships between two numeric variables.",
    "Histogram": "Shows the distribution of a single numeric variable.",
    "Count Plot": "Displays the frequency of categories.",
    "Pie Chart": "Displays proportions as slices of a circle (for categories).",
    "Pairplot": "Shows pairwise relationships between all numeric variables.",
    "Correlation Heatmap": "Visualizes correlation coefficients between numeric columns."
}

# AI-generated insight
def get_chart_insight(df, x_axis, y_axis, chart_type, llm):
    if df is None or df.empty or x_axis not in df.columns or (y_axis and y_axis not in df.columns):
        return "No insight could be generated."

    insight_template = """
Given the following chart type: "{chart_type}", and the chart data:

{data}

Generate a clear, concise insight or observation that a user might gain from this chart.
"""
    prompt = ChatPromptTemplate.from_template(insight_template)
    summary_chain = prompt | llm | StrOutputParser()
    return summary_chain.invoke({
        "chart_type": chart_type,
        "data": df[[x_axis, y_axis]].to_string(index=False) if y_axis else df[[x_axis]].to_string(index=False)
    })

# Export chart and insight to PDF

def export_pdf(fig, explanation):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        fig.savefig(tmp_img.name)
        img_path = tmp_img.name

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Chart Insight Report", ln=True, align="C")
        pdf.image(img_path, x=10, y=30, w=180)
        pdf.ln(120)
        pdf.multi_cell(0, 10, f"AI Explanation:\n{explanation}")
        pdf.output(tmp_pdf.name)
        return tmp_pdf.name


# Main visualization function
def visualize_data(df, llm):
    if df is None or df.empty:
        st.warning("No data available for visualization.")
        return

    suggested_chart, x_axis, y_axis = suggest_best_chart(df)
    chart_types = ["Suggested", "Table", "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Count Plot", "Pie Chart", "Pairplot", "Correlation Heatmap"]

    st.subheader("📊 Smart Visualization")
    viz_type = st.selectbox("Choose chart type:", chart_types)
    if viz_type == "Suggested":
        viz_type = suggested_chart

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    st.markdown("**X-axis** ❓ _This is the variable plotted horizontally (e.g., time, categories)_: ")
    x_axis = st.selectbox("X-axis", all_cols, index=all_cols.index(x_axis) if x_axis in all_cols else 0)

    if viz_type not in ["Histogram", "Count Plot", "Pie Chart", "Table", "Pairplot", "Correlation Heatmap"]:
        st.markdown("**Y-axis** ❓ _This is the variable you want to compare or measure_: ")
        y_axis = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(y_axis) if y_axis in numeric_cols else 0)
    else:
        y_axis = None

    hue = st.selectbox("Optional: Group by (Hue):", [None] + cat_cols) if viz_type in ["Bar Chart", "Line Chart", "Scatter Plot"] else None

    st.info(f"ℹ️ **{viz_type}**: {CHART_EXPLAINERS.get(viz_type, 'No description available.')}")

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if viz_type == "Table":
            st.dataframe(df)
            return
        elif viz_type == "Bar Chart":
            sns.barplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
        elif viz_type == "Line Chart":
            sns.lineplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
        elif viz_type == "Scatter Plot":
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
        elif viz_type == "Histogram":
            sns.histplot(data=df, x=x_axis, kde=True, ax=ax)
        elif viz_type == "Count Plot":
            sns.countplot(data=df, x=x_axis, ax=ax)
        elif viz_type == "Pie Chart":
            pie_data = df[x_axis].value_counts()
            ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
        elif viz_type == "Pairplot":
            st.pyplot(sns.pairplot(df.select_dtypes(include="number")))
            return
        elif viz_type == "Correlation Heatmap":
            corr = df.select_dtypes(include="number").corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        else:
            st.warning("Unsupported chart type.")
            return

        ax.set_title(f"{viz_type} of {y_axis if y_axis else ''} vs {x_axis}")
        ax.set_xlabel(x_axis)
        if y_axis:
            ax.set_ylabel(y_axis)

        st.pyplot(fig)

        # AI Insight
        with st.expander("🧠 AI Insight", expanded=True):
            with st.spinner("Generating chart insight..."):
                insight = get_chart_insight(df, x_axis, y_axis, viz_type, llm)
                st.success(insight)

        # PDF Export
        pdf_path = export_pdf(fig, insight)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Insight PDF", f, file_name="chart_report.pdf")

    except Exception as e:
        st.error(f"Plotting failed: {e}")

# Full Data Summary Tab
def summarize_dataframe(df, llm):
    if df.empty:
        return "No data available to summarize."
    summary_prompt = ChatPromptTemplate.from_template("""
You are a data analyst. Based on this dataset:

{data}

Give a brief summary of patterns, trends, or interesting insights across all variables.
""")
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({"data": df.to_string(index=False)})

# Streamlit App UI
st.title("🧠 AI Database Assistant")
st.write("Ask a question about your database below:")

user_question = st.text_input("Enter your question:")

if "df" not in st.session_state:
    st.session_state.df = None

if st.button("Get Answer"):
    if user_question:
        query = sql_chain.invoke({"question": user_question})
        df = run_query(query)
        st.session_state.df = df  # 🔐 persist it
        st.session_state.question = user_question
    else:
        st.warning("Please enter a question.")

df = st.session_state.get("df")
user_question = st.session_state.get("question")

if df is not None and not df.empty:
    explanation = generate_text_summary(df, user_question)

    tab1, tab2, tab3 = st.tabs(["💡 Explanation", "📊 Visualization", "📈 Insights Summary"])

    with tab1:
        st.subheader("💡 AI Explanation")
        st.write(explanation)

    with tab2:
        visualize_data(df, llm)

    with tab3:
        st.subheader("📈 AI Summary of Dataset")
        with st.spinner("Analyzing..."):
            summary = summarize_dataframe(df, llm)
            st.success(summary)
