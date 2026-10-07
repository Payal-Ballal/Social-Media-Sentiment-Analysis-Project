
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sentiment import process_data

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Social Sentiment Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Sentiment Analytics")
st.caption("Analyze public opinion through text sentiment analysis")

# ---------------- LOAD DATA ----------------

#@st.cache_data

@st.cache_data
def load_data():
    from sentiment import clean_text, analyze_sentiment

    df = pd.read_csv("data/train.csv")
    df = df[["text", "sentiment"]].copy()

    df["text"] = df["text"].fillna("").astype(str)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip() != ""].copy()

    original_labels = (
        df["sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    label_map = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral"
    }

    df["sentiment"] = original_labels.map(label_map)

    # Record the source of each label
    missing = df["sentiment"].isna()

    df["classification_source"] = "Dataset label"
    df.loc[missing, "classification_source"] = "VADER prediction"

    # Predict sentiment only where the original label is missing
    if missing.any():
        predictions = df.loc[missing, "text"].apply(analyze_sentiment)

        df.loc[missing, "sentiment"] = predictions.apply(
            lambda result: result[0]
        )

    # Calculate VADER scores for analysis
    scores = df["text"].apply(analyze_sentiment)
    df["compound_score"] = scores.apply(
        lambda result: result[1]
    )

    # No actual timestamps are supplied in this dataset
    df["date"] = pd.NaT

    return df.reset_index(drop=True)



try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Dashboard Filters")

sentiment_options = ["Positive", "Negative", "Neutral"]
selected_sentiments = st.sidebar.multiselect(
    "Select sentiment",
    sentiment_options,
    default=sentiment_options
)




keyword = st.sidebar.text_input(
    "Search posts by keyword"
)

# ---------------- APPLY FILTERS ----------------
filtered_df = df[df["sentiment"].isin(selected_sentiments)]


if keyword.strip():
    filtered_df = filtered_df[
        filtered_df["text"].str.contains(
            keyword.strip(), case=False, na=False
        )
    ]

# ---------------- KPI CARDS ----------------
st.subheader("Overview")

total_posts = len(filtered_df)

positive_count = (filtered_df["sentiment"] == "Positive").sum()
negative_count = (filtered_df["sentiment"] == "Negative").sum()
neutral_count = (filtered_df["sentiment"] == "Neutral").sum()

positive_pct = positive_count / total_posts * 100 if total_posts else 0
negative_pct = negative_count / total_posts * 100 if total_posts else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Posts", total_posts)
c2.metric("Positive Posts", int(positive_count))
c3.metric("Negative Posts", int(negative_count))
c4.metric("Positive Share", f"{positive_pct:.1f}%")


# ---------------- SENTIMENT INSIGHTS ----------------
st.subheader("Key Sentiment Insights")

if total_posts > 0:
    neutral_count = (filtered_df["sentiment"] == "Neutral").sum()
    neutral_pct = neutral_count / total_posts * 100

    avg_score = filtered_df["compound_score"].mean()

    i1, i2, i3 = st.columns(3)

    i1.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    i2.metric("Neutral Sentiment", f"{neutral_pct:.1f}%")
    i3.metric("Average Sentiment Score", f"{avg_score:.2f}")

    if positive_pct > negative_pct:
        st.success(
            "Overall, positive posts are more common "
            "than negative posts in the selected data."
        )
    elif negative_pct > positive_pct:
        st.warning(
            "Negative posts are more common than positive "
            "posts in the selected data."
        )
    else:
        st.info(
            "Positive and negative post counts are equal "
            "in the selected data."
        )
else:
    st.info("Select filters that include at least one post.")

st.divider()

# ---------------- LABEL SOURCE INSIGHT ----------------
st.subheader("Where Do Sentiment Labels Come From?")

source_counts = filtered_df["classification_source"].value_counts()

st.bar_chart(source_counts)

st.caption(
    "Dataset labels are provided in the original data. "
    "VADER predictions are used only when a label is missing."
)

st.divider()


# ---------------- SENTIMENT DISTRIBUTION ----------------
st.subheader("Sentiment Distribution")

left, right = st.columns(2)

sentiment_order = ["Positive", "Neutral", "Negative"]
sentiment_counts = (
    filtered_df["sentiment"]
    .value_counts()
    .reindex(sentiment_order, fill_value=0)
)

with left:
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_ylabel("Number of Posts")
    ax.set_title("Posts by Sentiment")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

with right:
    fig, ax = plt.subplots()
    nonzero_counts = sentiment_counts[sentiment_counts > 0]

    if len(nonzero_counts) > 0:
        ax.pie(
            nonzero_counts.values,
            labels=nonzero_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Sentiment Percentage")
    else:
        ax.text(0.5, 0.5, "No posts to display", ha="center")
        ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)

st.divider()



# ---------------- SENTIMENT DISTRIBUTION BY SOURCE ----------------
st.subheader("Sentiment Label Source")

source_counts = (
    filtered_df["classification_source"]
    .value_counts()
)

if not source_counts.empty:
    st.bar_chart(source_counts)
else:
    st.info("No posts available for the selected filters.")

st.caption(
    "The dataset has no actual timestamps. "
    "This chart shows label sources, not sentiment trends over time."
)

st.divider()


# ---------------- KEYWORD ANALYSIS ----------------
st.subheader("Most Common Words")

all_text = " ".join(filtered_df["text"].astype(str)).lower()

stop_words = {
    "the", "is", "a", "an", "and", "or", "to", "it",
    "this", "i", "my", "was", "with", "have", "had",
    "of", "for", "in", "on", "very", "are", "be", "not"
}

words = [
    word.strip(".,!?;:()[]{}\"'")
    for word in all_text.split()
]

words = [
    word for word in words
    if len(word) > 2 and word not in stop_words
]

common_words = Counter(words).most_common(10)

if common_words:
    word_df = pd.DataFrame(
        common_words,
        columns=["Word", "Frequency"]
    ).set_index("Word")

    st.bar_chart(word_df)
else:
    st.info("No words available for the selected filters.")

st.divider()

# ---------------- SENTIMENT SCORE DISTRIBUTION ----------------
st.subheader("Sentiment Score Distribution")

if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        filtered_df["compound_score"].dropna(),
        bins=20,
        edgecolor="black"
    )

    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Number of Posts")
    ax.set_title("Distribution of Sentiment Scores")
    ax.set_xlim(-1, 1)
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No posts available for the selected filters.")

st.caption(
    "Scores closer to -1 indicate more negative language; "
    "scores closer to +1 indicate more positive language."
)

st.divider()

# ---------------- POSTS TABLE ----------------
st.subheader("Analyzed Posts")


display_df = filtered_df[
    [
        "text",
        "sentiment",
        "compound_score",
        "classification_source"
    ]
].copy()


if "date" in display_df.columns:
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ---------------- DOWNLOAD ANALYZED DATA ----------------
st.subheader("Export Your Analysis")

csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Analyzed Data (CSV)",
    data=csv_data,
    file_name="sentiment_analysis_results.csv",
    mime="text/csv"
)


st.caption(
    "Sentiment labels are model predictions. "
    "This dashboard currently uses sample data, not live X posts."
)
