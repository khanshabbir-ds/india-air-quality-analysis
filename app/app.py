import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_PATH = Path(__file__).resolve().parent.parent / 'Data' / 'india_aqi.csv'
TITLE = "🌍 Air Quality Dashboard"

# Set page config
st.set_page_config(page_title=TITLE, page_icon="🌍", layout="wide")

@st.cache_data
def load_data(file_path: Path) -> pd.DataFrame:
    """Load and preprocess the air quality dataset."""
    try:
        df = pd.read_csv(file_path)
        # Rename columns for consistency
        column_map = {
            'City': 'city',
            'Date': 'date',
            'PM2.5': 'pm25',
            'PM10': 'pm10',
            'NO': 'no',
            'NO2': 'no2',
            'NOx': 'nox',
            'NH3': 'nh3',
            'CO': 'co',
            'SO2': 'so2',
            'O3': 'o3',
            'Benzene': 'benzene',
            'Toluene': 'toluene',
            'Xylene': 'xylene',
            'AQI': 'aqi',
            'AQI_Bucket': 'aqi_bucket'
        }
        df = df.rename(columns=column_map)
        # Convert date and drop missing AQI
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'aqi'])
        df.columns = df.columns.str.lower()
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {file_path}. Please ensure the file exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def filter_data(df: pd.DataFrame, selected_city: str, date_range: tuple) -> pd.DataFrame:
    """Filter data based on city and date range."""
    if selected_city != "All Cities":
        df = df[df['city'] == selected_city]
    start_date, end_date = date_range
    # Convert dates to datetime for comparison
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
    return df

def plot_aqi_trend(filtered_df: pd.DataFrame):
    """Plot AQI trend over time."""
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(filtered_df['date'], filtered_df['aqi'], marker='o', linestyle='-', alpha=0.7)
    ax.set_title('AQI Trend Over Time', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('AQI')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_monthly_average(filtered_df: pd.DataFrame):
    """Plot monthly average AQI."""
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    monthly = filtered_df.set_index('date').resample('ME')['aqi'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly['date'], monthly['aqi'], marker='o', linestyle='-', color='orange')
    ax.set_title('Monthly Average AQI', fontsize=16)
    ax.set_xlabel('Month')
    ax.set_ylabel('Average AQI')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_top_polluted_cities(df: pd.DataFrame):
    """Plot top 10 polluted cities."""
    top10 = df.groupby('city')['aqi'].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top10.values, y=top10.index, ax=ax, palette='Reds_r')
    ax.set_title('Top 10 Polluted Cities by Average AQI', fontsize=16)
    ax.set_xlabel('Average AQI')
    ax.set_ylabel('City')
    st.pyplot(fig)

def plot_aqi_distribution(filtered_df: pd.DataFrame):
    """Plot AQI distribution histogram."""
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df['aqi'], bins=30, kde=True, ax=ax, color='skyblue')
    ax.set_title('AQI Distribution', fontsize=16)
    ax.set_xlabel('AQI')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def display_metrics(filtered_df: pd.DataFrame):
    """Display metric cards."""
    if filtered_df.empty:
        st.warning("No data available for metrics.")
        return
    avg_aqi = filtered_df['aqi'].mean()
    max_aqi = filtered_df['aqi'].max()
    min_aqi = filtered_df['aqi'].min()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average AQI", f"{avg_aqi:.1f}")
    with col2:
        st.metric("Max AQI", f"{max_aqi:.1f}")
    with col3:
        st.metric("Min AQI", f"{min_aqi:.1f}")

def main():
    st.title(TITLE)

    # Load data
    df = load_data(DATA_PATH)
    if df.empty:
        return

    # Sidebar
    st.sidebar.header("Filters")
    cities = ["All Cities"] + sorted(df['city'].unique())
    selected_city = st.sidebar.selectbox("Select City", cities)
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter data
    filtered_df = filter_data(df, selected_city, date_range)

    # Metrics
    st.header("Key Metrics")
    display_metrics(filtered_df)

    # Charts
    st.header("Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AQI Trend")
        plot_aqi_trend(filtered_df)

    with col2:
        st.subheader("Monthly Average AQI")
        plot_monthly_average(filtered_df)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top Polluted Cities")
        plot_top_polluted_cities(df)  # Note: This uses full dataset

    with col4:
        st.subheader("AQI Distribution")
        plot_aqi_distribution(filtered_df)

    # Data Table
    st.header("Data Preview")
    st.dataframe(filtered_df.head(100))

if __name__ == "__main__":
    main()