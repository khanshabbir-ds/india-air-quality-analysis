from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for file output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LinearRegression

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'Data' / 'india_aqi.csv'
OUTPUT_DIR = ROOT_DIR / 'output_graphs'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

POLLUTANT_COLUMNS = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'aqi']

plot_counter = 0


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(value: str) -> str:
    return ''.join(
        char if char.isalnum() or char in (' ', '_', '-') else '_'
        for char in value
    ).strip().replace(' ', '_')


def save_plot(title: str) -> None:
    global plot_counter
    plot_counter += 1
    filename = OUTPUT_DIR / f'{plot_counter:02d}_{sanitize_filename(title)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
        'AQI_Bucket': 'aqi_bucket',
    }
    df = df.rename(columns=column_map)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df.columns = df.columns.str.lower()
    return df


def prepare_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(['city', 'date'])[POLLUTANT_COLUMNS].mean().reset_index()
    daily['unhealthy'] = (daily['aqi'] > 200).astype(int)
    return daily


def plot_monthly_average_aqi(daily: pd.DataFrame) -> None:
    monthly = (
        daily.set_index('date')
        .groupby('city')['aqi']
        .resample('ME')
        .mean()
        .reset_index()
    )
    cities = ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru']
    plt.figure(figsize=(12, 6))
    for city in cities:
        city_data = monthly[monthly['city'] == city]
        if not city_data.empty:
            plt.plot(city_data['date'], city_data['aqi'], marker='o', label=city)
    plt.title('Monthly Average AQI for Sample Cities', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average AQI', fontsize=12)
    plt.legend(title='City')
    plt.grid(True)
    plt.tight_layout()
    save_plot('Monthly_Average_AQI')


def plot_top10_cities_by_aqi(daily: pd.DataFrame) -> None:
    top10 = daily.groupby('city')['aqi'].mean().nlargest(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top10.index, y=top10.values, palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Top 10 Cities by Average AQI', fontsize=16)
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Average AQI', fontsize=12)
    plt.tight_layout()
    save_plot('Top_10_Cities_by_Average_AQI')


def plot_aqi_scatter(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=daily,
        x='city',
        y='aqi',
        hue='city',
        palette='tab10',
        s=80,
        alpha=0.7,
        legend=False,
    )
    plt.xticks(rotation=45)
    plt.title('AQI Distribution Across Cities', fontsize=16)
    plt.xlabel('City', fontsize=12)
    plt.ylabel('AQI', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_plot('AQI_Scatter_by_City')


def plot_top10_pie(daily: pd.DataFrame) -> None:
    top10 = daily.groupby('city')['aqi'].mean().nlargest(10)
    plt.figure(figsize=(8, 8))
    colors = sns.color_palette('tab10', len(top10))
    plt.pie(
        top10,
        labels=top10.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        shadow=True,
    )
    plt.title('Top 10 Cities by Average AQI', fontsize=16)
    plt.axis('equal')
    save_plot('Top_10_Cities_AQI_Pie')


def plot_aqi_distribution(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(
        daily['aqi'],
        bins=40,
        kde=True,
        color='#FF6F61',
        edgecolor='black',
        alpha=0.8,
    )
    plt.title('AQI Distribution Across All Cities', fontsize=16, fontweight='bold')
    plt.xlabel('AQI', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot('AQI_Distribution')


def plot_city_boxplot(daily: pd.DataFrame) -> None:
    cities = ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Lucknow']
    subset = daily[daily['city'].isin(cities)]
    plt.figure(figsize=(12, 6))
    palette = {
        'Delhi': '#FF6F61',
        'Mumbai': '#6B5B95',
        'Kolkata': '#88B04B',
        'Bengaluru': '#F7CAC9',
        'Lucknow': '#92A8D1',
    }
    sns.boxplot(x='city', y='aqi', data=subset, palette=palette)
    plt.title('AQI Distribution by City', fontsize=16, fontweight='bold')
    plt.xlabel('City', fontsize=12)
    plt.ylabel('AQI', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot('AQI_Boxplot_by_City')


def plot_delhi_seasonal_distribution(daily: pd.DataFrame) -> None:
    daily['month'] = daily['date'].dt.month
    delhi_data = daily[daily['city'] == 'Delhi']
    if delhi_data.empty:
        return
    monthly_avg = delhi_data.groupby('month')['aqi'].mean()
    norm = Normalize(vmin=monthly_avg.min(), vmax=monthly_avg.max())
    cmap = get_cmap('plasma')
    palette = [cmap(norm(value)) for value in monthly_avg.sort_index()]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='month', y='aqi', data=delhi_data, palette=palette)
    plt.title('Seasonal AQI Distribution in Delhi', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('AQI', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot('Delhi_Seasonal_AQI_Distribution')


def plot_delhi_heatmap(daily: pd.DataFrame) -> None:
    delhi = daily[daily['city'] == 'Delhi'].copy()
    if delhi.empty:
        return
    delhi['dayofyear'] = delhi['date'].dt.dayofyear
    delhi['year'] = delhi['date'].dt.year
    pivot = delhi.pivot_table(index='dayofyear', columns='year', values='aqi')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='RdYlGn_r', cbar_kws={'label': 'AQI'})
    plt.title('Delhi Daily AQI Heatmap (by Year)', fontsize=16)
    save_plot('Delhi_AQI_Heatmap')


def plot_pm25_aqi_relationship(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x='pm25',
        y='aqi',
        data=daily,
        scatter=False,
        line_kws={'color': 'black', 'linewidth': 2, 'label': 'Regression Line'},
    )
    norm = Normalize(vmin=daily['aqi'].min(), vmax=daily['aqi'].max())
    plt.scatter(
        daily['pm25'],
        daily['aqi'],
        c=get_cmap('inferno')(norm(daily['aqi'])),
        alpha=0.6,
        edgecolor='white',
        linewidth=0.5,
    )
    plt.title('Relationship between PM2.5 and AQI', fontsize=16, fontweight='bold')
    plt.xlabel('PM2.5 Concentration (µg/m³)', fontsize=12)
    plt.ylabel('Air Quality Index (AQI)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot('PM25_vs_AQI')


def plot_distribution_moments(daily: pd.DataFrame) -> None:
    stats_df = daily.groupby('city')['aqi'].agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        skew=lambda x: skew(x.dropna()),
        kurtosis=lambda x: kurtosis(x.dropna()),
    ).reset_index()
    stats_df['pct_unhealthy'] = daily.groupby('city')['unhealthy'].mean().values * 100
    plt.figure(figsize=(14, 6))
    melted = stats_df.melt(
        id_vars='city',
        value_vars=['skew', 'kurtosis'],
        var_name='Metric',
        value_name='Value',
    )
    sorted_cities = stats_df.sort_values('skew', ascending=False)['city']
    melted['city'] = pd.Categorical(melted['city'], categories=sorted_cities, ordered=True)
    sns.barplot(data=melted, x='city', y='Value', hue='Metric', palette='Set2')
    plt.title('Skewness and Kurtosis of AQI by City', fontsize=16, fontweight='bold')
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Metric')
    plt.tight_layout()
    save_plot('AQI_Skewness_Kurtosis')
    print(stats_df.sort_values('mean', ascending=False).head(10).to_string(index=False))


def plot_correlation_heatmap(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        daily[POLLUTANT_COLUMNS].corr(),
        annot=True,
        cmap='Spectral',
        center=0,
        linewidths=0.5,
    )
    plt.title('Correlation among Pollutants and AQI', fontsize=14)
    save_plot('Pollutant_Correlation_Heatmap')


def plot_aqi_category_heatmap(daily: pd.DataFrame) -> None:
    city_avg = daily.groupby('city')['aqi'].mean().reset_index()
    category_map = {
        'Good': 1,
        'Moderate': 2,
        'Unhealthy': 3,
        'Very Unhealthy': 4,
        'Hazardous': 5,
    }
    city_avg['category'] = city_avg['aqi'].apply(
        lambda value: 'Good'
        if value <= 50
        else 'Moderate'
        if value <= 100
        else 'Unhealthy'
        if value <= 200
        else 'Very Unhealthy'
        if value <= 300
        else 'Hazardous'
    )
    city_avg['category_num'] = city_avg['category'].map(category_map)
    heatmap_data = city_avg.set_index('city')[['category_num']]
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=city_avg['category'].values.reshape(-1, 1),
        fmt='',
        cmap='RdYlGn_r',
        cbar_kws={'ticks': list(category_map.values()), 'label': 'AQI Category'},
    )
    plt.title('AQI Category by City', fontsize=16)
    save_plot('AQI_Category_Heatmap')


def plot_linear_regression(daily: pd.DataFrame) -> None:
    reg_data = daily[['aqi', 'pm25']].dropna()
    if reg_data.empty:
        return
    X = sm.add_constant(reg_data['pm25'])
    y = reg_data['aqi']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    sklearn_model = LinearRegression()
    X_values = reg_data['pm25'].values.reshape(-1, 1)
    y_values = reg_data['aqi'].values
    sklearn_model.fit(X_values, y_values)
    y_pred = sklearn_model.predict(X_values)

    slope = sklearn_model.coef_[0]
    intercept = sklearn_model.intercept_
    r2 = sklearn_model.score(X_values, y_values)

    plt.figure(figsize=(9, 6))
    norm = Normalize(vmin=y_values.min(), vmax=y_values.max())
    plt.scatter(
        reg_data['pm25'],
        reg_data['aqi'],
        c=get_cmap('viridis')(norm(y_values)),
        alpha=0.6,
        edgecolor='white',
        linewidth=0.5,
    )
    plt.plot(reg_data['pm25'], y_pred, color='black', linewidth=2, label='Regression Line')
    plt.text(
        0.05,
        0.95,
        f'AQI = {slope:.2f} × PM2.5 + {intercept:.2f}\nR² = {r2:.3f}',
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7),
    )
    plt.title('Linear Relationship Between PM2.5 and AQI', fontsize=16, fontweight='bold')
    plt.xlabel('PM2.5 Concentration (µg/m³)', fontsize=12)
    plt.ylabel('Air Quality Index (AQI)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot('Linear_Regression_PM25_AQI')


def plot_city_r2_scores(daily: pd.DataFrame) -> None:
    r2_records = []
    for city, group in daily.groupby('city'):
        subset = group[['aqi', 'pm25']].dropna()
        if len(subset) < 30:
            continue
        X_city = sm.add_constant(subset['pm25'])
        y_city = subset['aqi']
        model = sm.OLS(y_city, X_city).fit()
        r2_records.append({'city': city, 'r_squared': model.rsquared})
    if not r2_records:
        return
    r2_df = pd.DataFrame(r2_records).sort_values('r_squared', ascending=False)
    plt.figure(figsize=(14, 6))
    sns.barplot(data=r2_df, x='city', y='r_squared', palette='viridis')
    plt.title('City-wise R²: How Well PM2.5 Predicts AQI', fontsize=16, fontweight='bold')
    plt.xlabel('City', fontsize=12)
    plt.ylabel('R² Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_plot('Citywise_R2_PM25_AQI')


def print_unhealthy_winter_probability(daily: pd.DataFrame) -> None:
    delhi = daily[daily['city'] == 'Delhi']
    if delhi.empty:
        return
    winter_rate = delhi[delhi['date'].dt.month.isin([12, 1, 2])]['unhealthy'].mean()
    print(f'P(Unhealthy | Delhi, Winter) = {winter_rate:.3f}')


def main() -> None:
    ensure_output_directory(OUTPUT_DIR)
    df = load_data(DATA_PATH)
    daily = prepare_daily_summary(df)

    print(f'Loaded {len(daily)} daily records for {daily["city"].nunique()} cities.')

    plot_monthly_average_aqi(daily)
    plot_top10_cities_by_aqi(daily)
    plot_aqi_scatter(daily)
    plot_top10_pie(daily)
    plot_aqi_distribution(daily)
    plot_city_boxplot(daily)
    plot_delhi_seasonal_distribution(daily)
    plot_delhi_heatmap(daily)
    plot_pm25_aqi_relationship(daily)
    plot_distribution_moments(daily)
    plot_correlation_heatmap(daily)
    plot_aqi_category_heatmap(daily)
    plot_linear_regression(daily)
    plot_city_r2_scores(daily)
    print_unhealthy_winter_probability(daily)


if __name__ == '__main__':
    main()
