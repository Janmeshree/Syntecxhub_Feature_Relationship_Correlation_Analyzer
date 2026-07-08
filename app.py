import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Feature Relationship & Correlation Analyzer",
    layout="wide"
)

st.title("Feature Relationship & Correlation Analyzer")

st.markdown("""
Upload any **CSV dataset** to explore relationships between numerical features using **Pearson Correlation**.

The application generates an interactive correlation heatmap, pairplot, identifies the strongest positive and negative feature relationships, and provides a downloadable analysis summary.
""")

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    overview_tab, correlation_tab, pairplot_tab, summary_tab = st.tabs(
        [
            "Overview",
            "Correlation Analysis",
            "Pairwise Relationships",
            "Correlation Summary"
        ]
    ) 

    with overview_tab:

        st.header("Dataset Overview")

        st.dataframe(df.head(), use_container_width=True)

        st.write("")

        # Dashboard Cards
        total_rows = df.shape[0]
        total_columns = df.shape[1]
        numeric_columns = len(df.select_dtypes(include=np.number).columns)
        missing_values = df.isnull().sum().sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Rows", total_rows)
        c2.metric("Total Columns", total_columns)
        c3.metric("Numeric Features", numeric_columns)
        c4.metric("Missing Values", missing_values)

        st.divider()

        st.subheader("Summary Statistics")

        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True
        )

        st.divider()

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            st.error("The uploaded dataset must contain at least two numerical columns.")
            st.stop()
    
    with correlation_tab:
        # PEARSON CORRELATION
        st.subheader("Pearson Correlation Heatmap")

        st.markdown("Heatmap Controls")

        col1, col2 = st.columns(2)

        with col1:
            correlation_method = st.selectbox(
                "Correlation Method",
                ["Pearson", "Spearman"]
            )

        with col2:
            color_palette = st.selectbox(
                "Color Palette",
                [
                    "coolwarm",
                    "viridis",
                    "RdBu_r",
                    "Spectral"
                ]
            )

        # correlation_matrix = numeric_df.corr(method="pearson")

        correlation_matrix = numeric_df.corr(
        method=correlation_method.lower()
        )

        # Creating upper triangle mask
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap=color_palette,
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(
            f"{correlation_method} Correlation Heatmap",
            fontsize=16,
            fontweight="bold",
            pad=20
        )

        # st.pyplot(fig, width=700)

        left, center, right = st.columns([1, 4, 1])

        with center:
            st.pyplot(fig)

        st.caption(
        f"The heatmap displays {correlation_method.title()} correlation values for all numerical features in the dataset."
        )

        # Save heatmap
        heatmap_path = f"{correlation_method.lower()}_correlation_heatmap.png"
        fig.savefig(
            heatmap_path,
            dpi=300,
            bbox_inches="tight"
        )

        with open(heatmap_path, "rb") as file:
            st.download_button(
                label="Download Heatmap (PNG)",
                data=file,
                file_name=f"{correlation_method.lower()}_correlation_heatmap.png",
                mime="image/png"
            )

        st.divider()

    with pairplot_tab:
        # PAIRPLOT
        st.subheader("Pairwise Relationships")

        available_columns = numeric_df.columns.tolist()

        selected_columns = st.multiselect(
            "Select 2 to 5 numerical features for Pairplot",
            available_columns,
            default=available_columns[:min(4, len(available_columns))]
        )

        if len(selected_columns) >= 2:
            with st.spinner("Generating pairplot..."):
                pairplot = sns.pairplot(
                    numeric_df[selected_columns],
                    corner=True,
                    diag_kind="hist",
                    height=1.8
                )

            # st.pyplot(pairplot.figure, width=700)

            left, center, right = st.columns([1, 4, 1])

            with center:
                st.pyplot(pairplot.figure)

            pairplot_path = "pairplot.png"

            pairplot.figure.savefig(
                pairplot_path,
                dpi=300,
                bbox_inches="tight"
            )

            with open(pairplot_path, "rb") as file:
                st.download_button(
                    "Download Pairplot (PNG)",
                    data=file,
                    file_name="pairplot.png",
                    mime="image/png"
                )

        else:
            st.warning("Please select at least two numerical columns.")

        st.divider()

        st.subheader("Feature Relationship Ranking")

        corr_pairs = (
            correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            .stack()
            .reset_index()
        )

        corr_pairs.columns = [
            "Feature 1",
            "Feature 2",
            "Correlation"
        ]

        positive_corr = corr_pairs.sort_values(
            by="Correlation",
            ascending=False
        ).head(5)

        negative_corr = corr_pairs.sort_values(
            by="Correlation"
        ).head(5)

        strongest_positive = positive_corr.iloc[0]
        strongest_negative = negative_corr.iloc[0]

        st.success(
            f"**Strongest Positive Relationship:** "
            f"{strongest_positive['Feature 1']} ↔ {strongest_positive['Feature 2']} "
            f"({strongest_positive['Correlation']:.2f})"
        )

        st.error(
            f"**Strongest Negative Relationship:** "
            f"{strongest_negative['Feature 1']} ↔ {strongest_negative['Feature 2']} "
            f"({strongest_negative['Correlation']:.2f})"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Positive Correlations")
            st.dataframe(
                positive_corr,
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.subheader("Top Negative Correlations")
            st.dataframe(
                negative_corr,
                use_container_width=True,
                hide_index=True
            )

        st.divider()
    
    with summary_tab:
        # CORRELATION SUMMARY
        st.header("Correlation Summary")

        # Determine strength of the strongest positive correlation
        positive_value = strongest_positive["Correlation"]

        if positive_value >= 0.7:
            relationship_strength = "a strong"
        elif positive_value >= 0.4:
            relationship_strength = "a moderate"
        else:
            relationship_strength = "a weak"

        summary = f"""CORRELATION SUMMARY REPORT

Dataset Information

Dataset Name : {uploaded_file.name}
Rows : {total_rows}
Columns : {total_columns}
Numeric Features : {numeric_columns}
Correlation Method : Pearson


Strongest Positive Relationship

{strongest_positive['Feature 1']} ↔ {strongest_positive['Feature 2']}
Correlation Coefficient : {strongest_positive['Correlation']:.2f}


Strongest Negative Relationship

{strongest_negative['Feature 1']} ↔ {strongest_negative['Feature 2']}
Correlation Coefficient : {strongest_negative['Correlation']:.2f}


Top 5 Positive Correlations
        """

        for i, row in enumerate(positive_corr.itertuples(index=False), start=1):
            summary += (
                f"\n{i}. {row[0]} ↔ {row[1]} "
                f"({row[2]:.2f})"
            )

        summary += "\n\nTop 5 Negative Correlations\n"

        for i, row in enumerate(negative_corr.itertuples(index=False), start=1):
            summary += (
                f"\n{i}. {row[0]} ↔ {row[1]} "
                f"({row[2]:.2f})"
            )

        summary += f"""

Conclusion

Pearson correlation analysis identified the strongest positive and negative relationships among the numerical features.
The heatmap and pairplot provide additional visual support for interpreting these relationships.
        """

        st.code(summary)

        st.download_button(
            label="Download Correlation Summary Report",
            data=summary,
            file_name="correlation_summary_report.txt",
            mime="text/plain"
        )

        st.divider()

        st.sidebar.markdown("---")
        st.sidebar.subheader("About")

        st.sidebar.info("""
        This application analyzes relationships between numerical features using Pearson Correlation.
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Application Features")

        st.sidebar.write("""

        Dataset Overview 
                        
        Summary Statistics
                        
        Pearson Correlation Heatmap
                        
        Pairplot for selected features
                        
        Strongest Positive & Negative Relationships
                        
        Download Plots
                        
        Download Correlation Summary Report
        """)

else:
    st.info("Upload a CSV dataset from the sidebar to begin the analysis.")