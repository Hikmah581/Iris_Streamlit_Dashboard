# =====================================================
# IRIS DATASET ANALYSIS DASHBOARD
# Using Seaborn + Streamlit
# =====================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Iris Dataset Dashboard",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dataset Overview",
        "Statistics",
        "Visualizations",
        "Insights"
    ]
)

# -----------------------------------------------------
# DATASET OVERVIEW
# -----------------------------------------------------
if page == "Dataset Overview":

    st.title("🌸 Iris Dataset Dashboard")

    st.markdown("""
    This dashboard presents an exploratory analysis of the famous Iris Dataset.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Species", df["species"].nunique())

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes.astype(str)
    })

    st.dataframe(info_df)

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum().reset_index()
                 .rename(columns={"index":"Column",0:"Missing Values"}))

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Dataset",
        data=csv,
        file_name="iris_dataset.csv",
        mime="text/csv"
    )

# -----------------------------------------------------
# STATISTICS PAGE
# -----------------------------------------------------
elif page == "Statistics":

    st.title("📊 Statistical Summary")

    st.dataframe(df.describe())

    st.subheader("Species Distribution")

    st.dataframe(df["species"].value_counts())

# -----------------------------------------------------
# VISUALIZATION PAGE
# -----------------------------------------------------
elif page == "Visualizations":

    st.title("📈 Iris Dataset Visualizations")

    chart = st.selectbox(
        "Choose Visualization",
        [
            "Count Plot",
            "Histogram",
            "Boxplot",
            "Pairplot",
            "Correlation Heatmap",
            "Scatter Plot",
            "Violin Plot"
        ]
    )

    # COUNT PLOT
    if chart == "Count Plot":

        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(data=df, x="species", ax=ax)
        ax.set_title("Species Distribution")
        st.pyplot(fig)

    # HISTOGRAM
    elif chart == "Histogram":

        feature = st.selectbox(
            "Select Feature",
            df.columns[:-1]
        )

        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df[feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

    # BOXPLOT
    elif chart == "Boxplot":

        feature = st.selectbox(
            "Select Feature",
            df.columns[:-1]
        )

        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(
            data=df,
            x="species",
            y=feature,
            ax=ax
        )
        ax.set_title(f"{feature} by Species")
        st.pyplot(fig)

    # PAIRPLOT
    elif chart == "Pairplot":

        st.write("Generating Pairplot...")

        pair_fig = sns.pairplot(
            df,
            hue="species"
        )

        st.pyplot(pair_fig.figure)

    # HEATMAP
    elif chart == "Correlation Heatmap":

        fig, ax = plt.subplots(figsize=(8,6))

        corr = df.drop("species", axis=1).corr()

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title("Correlation Heatmap")

        st.pyplot(fig)

    # SCATTER
    elif chart == "Scatter Plot":

        x_feature = st.selectbox(
            "X Axis",
            df.columns[:-1]
        )

        y_feature = st.selectbox(
            "Y Axis",
            df.columns[:-1],
            index=1
        )

        fig, ax = plt.subplots(figsize=(8,5))

        sns.scatterplot(
            data=df,
            x=x_feature,
            y=y_feature,
            hue="species",
            ax=ax
        )

        ax.set_title(
            f"{x_feature} vs {y_feature}"
        )

        st.pyplot(fig)

    # VIOLIN PLOT
    elif chart == "Violin Plot":

        feature = st.selectbox(
            "Feature",
            df.columns[:-1]
        )

        fig, ax = plt.subplots(figsize=(8,5))

        sns.violinplot(
            data=df,
            x="species",
            y=feature,
            ax=ax
        )

        ax.set_title(f"{feature} by Species")

        st.pyplot(fig)

# -----------------------------------------------------
# INSIGHTS PAGE
# -----------------------------------------------------
elif page == "Insights":

    st.title("💡 Key Insights")

    st.markdown("""
    ### Insights from the Iris Dataset

    1. The dataset contains 150 observations.

    2. There are three flower species:
       - Setosa
       - Versicolor
       - Virginica

    3. Petal length and petal width are highly correlated.

    4. Setosa is clearly separated from the other species.

    5. Virginica generally has the largest petal dimensions.

    6. Sepal measurements show more overlap among species.

    7. Petal measurements are the strongest predictors of species classification.

    ### Conclusion

    The Iris dataset demonstrates clear species differentiation,
    especially through petal length and petal width.
    """)
