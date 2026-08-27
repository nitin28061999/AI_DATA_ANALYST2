import pandas as pd
import plotly.express as px
import streamlit as st

from analyst import analyze_data # pyright: ignore[reportAttributeAccessIssue]
from data_profile import profile_data

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")
st.markdown("Upload a CSV file and ask questions about your dataset in natural language.")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns.")

        # Interface Navigation Tabs
        tab_analysis, tab_profile, tab_preview = st.tabs(["💬 Ask AI Analyst", "📈 Data Profile", "📋 Raw Data"])

        with tab_preview:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(50), use_container_width=True)

        with tab_profile:
            st.subheader("Dataset Metadata & Profile")
            profile = profile_data(df)
            st.json(profile)

        with tab_analysis:
            st.subheader("Query Your Data")
            question = st.text_input("Enter your question:", placeholder="e.g., Show top 5 products by revenue")

            if st.button("Run Analysis", type="primary"):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Analyzing dataset with AI..."):
                        try:
                            output = analyze_data(df, question)

                            st.markdown("### Answer")
                            st.write(output["explanation"])

                            # Render Chart if result is tabular data
                            result = output.get("result")
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                st.markdown("### Visualization")
                                cols = result.columns.tolist()
                                
                                if len(cols) >= 2:
                                    x_col, y_col = cols[0], cols[1]
                                    
                                    # Choose chart type based on column semantics
                                    if x_col.lower() in ["month", "date"]:
                                        fig = px.line(result, x=x_col, y=y_col, title=f"{y_col} over {x_col}", markers=True)
                                    else:
                                        fig = px.bar(result, x=x_col, y=y_col, title=f"{y_col} by {x_col}", text_auto=True)
                                        
                                    st.plotly_chart(fig, use_container_width=True)

                            with st.expander("Show Execution Details & Plan"):
                                st.markdown("**Generated Execution Plan:**")
                                st.json(output["plan"])
                                st.markdown("**Raw Query Output:**")
                                st.write(result)

                        except Exception as e:
                            st.error(f"Analysis Error: {str(e)}")

    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
else:
    st.info("Please upload a CSV file to start analyzing.")