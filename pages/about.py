import streamlit as st
import pandas as pd

def main():
    st.title("About the Data 💾")

    # Load sample data
    try:
        df = pd.read_csv("data/alzheimers.csv")
        st.success("Data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.markdown(
        """
        This dataset documents global Alzheimer's risk factor data collected from 20 countries.
        It includes features like Age, BMI, Cognitive Score, Stress Level, Education, and more.
        """
    )

    st.write("### Sample Data Preview")
    st.dataframe(df)

if __name__ == "__main__":
    main()
