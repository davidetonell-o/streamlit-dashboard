#Import libraries
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

#Set streamlit page
st.set_page_config(page_title="Netflix Dashboard", layout="wide")
st.title("Netflix Data Dashboard")
st.write("Visualize and analyze Netflix titles by country, year and genre.")

#Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/archive/netflix_titles.csv")

df = load_data()

#Cleaning data
df["country"] = (
    df["country"].astype(str).str.strip(" ,")
    .replace({"nan" : None})
    .fillna("Unknown")
)
df["listed_in"] = (
    df["listed_in"].astype(str).str.strip(" ,")
    .replace({"nan" : None})
    .fillna("Unknown")
)

#Sidebar filters
st.sidebar.header("Filters")

countries = (
    df["country"].dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .replace("",pd.NA)
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)
countries = ["All countries"] + countries
country = st.sidebar.selectbox("Country", countries, index=0)

genres = (
    df["listed_in"].dropna()
    .str.split(",")
    .explode()
    .str.strip(" ,")
    .replace("", pd.NA)
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)
genres = ["All genres"] + genres
genre = st.sidebar.selectbox("Genre", genres, index=0)

year_range = st.sidebar.slider("Release year", 1950, 2021, (2000, 2021))

#Dinamic filtering
filtered = df[
    (
        True if country == "All countries"
        else df["country"].str.contains(country, case=False, na=False)
    )
    & (df["release_year"].between(year_range[0], year_range[1]))
    & (
        True if genre == "All genres"
        else df["listed_in"].str.contains(genre, na=False, case=False) if genre else True
    )
]

st.write(f"{filtered.shape[0]} titles found")
st.dataframe(filtered.head(10))

#Graph distribution per release year
st.subheader("Distribution per releasing year")
st.bar_chart(filtered["release_year"].value_counts().sort_index(), width="stretch", height="stretch")

#Graph count film vs series
st.subheader("Count by type (Movie vs TV Show)")
fig, ax = plt.subplots(figsize=(7,3))
sns.countplot(data=filtered, x="type", ax=ax)
st.pyplot(fig)

#Metrics
col1, col2 = st.columns(2)

with col1:
    num_countries = (
        df["country"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .nunique()
    )
    st.metric(label="Number of countries", value=num_countries)

with col2:
    st.metric(
        label="Average release year (filtered)",
        value=int(filtered["release_year"].mean())
        if not filtered.empty else "N/A"
    )

#Tabs
tab1, tab2, tab3 = st.tabs(["🎭 Genres", "🌍 Countries", "📆 Trend"])

with tab1:
    st.subheader("Top genres in current selection")
    top_genres = (
        filtered["listed_in"]
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(10)
    )
    st.bar_chart(top_genres)
    st.caption("💡 These are the most frequent genres in the current filter selection.")

with tab2:
    st.subheader("Top countries in current selection")
    top_countries = (
        filtered["country"]
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(10)
    )
    st.bar_chart(top_countries)

with tab3:
    st.subheader("Titles over time")
    st.line_chart(filtered["release_year"].value_counts().sort_index())
    st.caption("💡 Trend of titles in the selected time window.")
