"""
Stock Market Clustering with K-Means
=====================================
This project clusters stocks based on their returns and volatility (variance)
to identify groups of stocks with similar behavior patterns.
"""

# ============ 1. IMPORT LIBRARIES ============
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============ 2. DEFINE STOCK TICKERS ============
# Select diverse stocks from different sectors
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech giants
    'JPM', 'BAC', 'WFC',  # Banking
    'JNJ', 'PFE', 'UNH',  # Healthcare
    'XOM', 'CVX', 'COP',  # Energy
    'WMT', 'COST', 'TGT',  # Retail
    'NFLX', 'DIS', 'CMCSA',  # Entertainment
    'BA', 'CAT', 'GE'  # Industrials
]

print(f"Analyzing {len(tickers)} stocks...")
print(f"Tickers: {tickers}\n")

# ============ 3. FETCH STOCK DATA ============
def fetch_stock_data(tickers, period='1y'):
    """
    Fetch historical stock data using yfinance
    """
    data = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                data[ticker] = hist
                print(f"✓ Fetched data for {ticker}")
            else:
                failed_tickers.append(ticker)
                print(f"✗ No data for {ticker}")
        except Exception as e:
            failed_tickers.append(ticker)
            print(f"✗ Error fetching {ticker}: {str(e)[:50]}")
    
    return data, failed_tickers

print("Fetching stock data...")
stock_data, failed = fetch_stock_data(tickers)

# Remove failed tickers
tickers = [t for t in tickers if t not in failed]
print(f"\nSuccessfully fetched {len(tickers)} stocks\n")

# ============ 4. CALCULATE METRICS ============
def calculate_stock_metrics(stock_data):
    """
    Calculate returns and volatility for each stock
    """
    metrics = []
    
    for ticker, data in stock_data.items():
        if len(data) > 1:
            # Calculate daily returns
            data['Daily_Return'] = data['Close'].pct_change()
            
            # Calculate annualized metrics (assuming 252 trading days)
            annual_return = data['Daily_Return'].mean() * 252
            annual_volatility = data['Daily_Return'].std() * np.sqrt(252)
            
            # Calculate Sharpe ratio (risk-adjusted return)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            metrics.append({
                'Ticker': ticker,
                'Annual_Return': annual_return,
                'Annual_Volatility': annual_volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Avg_Close': data['Close'].mean(),
                'Current_Price': data['Close'].iloc[-1]
            })
    
    return pd.DataFrame(metrics)

print("Calculating stock metrics...")
metrics_df = calculate_stock_metrics(stock_data)
print(metrics_df.round(4).to_string(index=False))
print()

# ============ 5. PREPARE DATA FOR CLUSTERING ============
# Select features for clustering
features = ['Annual_Return', 'Annual_Volatility']
X = metrics_df[features].copy()

print("Data Statistics:")
print(X.describe())
print()

# Handle any missing values
X = X.dropna()

# Standardize the features (crucial for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("After standardization:")
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Std: {X_scaled.std(axis=0)}\n")

# ============ 6. FIND OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD) ============
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow method plot
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method for Optimal k', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Silhouette score plot
axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score for Optimal k', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=max(silhouette_scores), color='green', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Determine optimal k (using elbow point and max silhouette)
optimal_k_elbow = 3  # Based on typical elbow pattern
optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
optimal_k = optimal_k_silhouette

print(f"Optimal clusters by silhouette score: {optimal_k}")
print(f"Best silhouette score: {max(silhouette_scores):.4f}\n")

# ============ 7. APPLY K-MEANS CLUSTERING ============
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
metrics_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Get cluster centers in original scale
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

# ============ 8. VISUALIZE CLUSTERING RESULTS ============
fig, ax = plt.subplots(figsize=(12, 8))

# Create scatter plot with different colors for clusters
scatter = ax.scatter(
    metrics_df['Annual_Volatility'], 
    metrics_df['Annual_Return'],
    c=metrics_df['Cluster'], 
    cmap='viridis', 
    s=200, 
    alpha=0.7,
    edgecolors='black',
    linewidth=1
)

# Add labels for each stock
for idx, row in metrics_df.iterrows():
    ax.annotate(
        row['Ticker'], 
        (row['Annual_Volatility'], row['Annual_Return']),
        xytext=(5, 5), 
        textcoords='offset points',
        fontsize=9,
        alpha=0.8
    )

# Plot cluster centers
ax.scatter(
    cluster_centers[:, 1],  # Volatility
    cluster_centers[:, 0],  # Return
    c='red', 
    s=300, 
    marker='X',
    edgecolors='black',
    linewidth=2,
    label='Cluster Centers'
)

ax.set_xlabel('Annual Volatility (Risk)', fontsize=12)
ax.set_ylabel('Annual Return', fontsize=12)
ax.set_title(f'Stock Clustering using K-Means (k={optimal_k})', fontsize=14)
ax.legend(*scatter.legend_elements(), title="Clusters")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============ 9. ANALYZE CLUSTER CHARACTERISTICS ============
print("\n" + "="*60)
print("CLUSTER ANALYSIS RESULTS")
print("="*60)

cluster_summary = metrics_df.groupby('Cluster').agg({
    'Ticker': 'count',
    'Annual_Return': 'mean',
    'Annual_Volatility': 'mean',
    'Sharpe_Ratio': 'mean'
}).round(4)

cluster_summary.columns = ['Count', 'Avg_Return', 'Avg_Volatility', 'Avg_Sharpe']
print("\nCluster Summary:")
print(cluster_summary)

print("\n" + "-"*60)
print("STOCKS IN EACH CLUSTER:")
print("-"*60)

for cluster in sorted(metrics_df['Cluster'].unique()):
    stocks_in_cluster = metrics_df[metrics_df['Cluster'] == cluster]['Ticker'].tolist()
    cluster_type = ""
    
    # Interpret cluster characteristics
    avg_return = cluster_summary.loc[cluster, 'Avg_Return']
    avg_vol = cluster_summary.loc[cluster, 'Avg_Volatility']
    
    if avg_return > 0.15 and avg_vol < 0.25:
        cluster_type = "(High Return, Low Risk) - PERFORMERS"
    elif avg_return > 0.15 and avg_vol >= 0.25:
        cluster_type = "(High Return, High Risk) - AGGRESSIVE"
    elif avg_return <= 0.15 and avg_vol < 0.25:
        cluster_type = "(Moderate Return, Low Risk) - STABLE"
    elif avg_return <= 0.15 and avg_vol >= 0.25:
        cluster_type = "(Low Return, High Risk) - VOLATILE"
    else:
        cluster_type = ""
    
    print(f"\nCluster {cluster} {cluster_type}:")
    print(f"  Stocks: {', '.join(stocks_in_cluster)}")
    print(f"  Count: {len(stocks_in_cluster)} stocks")
    print(f"  Avg Annual Return: {avg_return*100:.2f}%")
    print(f"  Avg Volatility: {avg_vol*100:.2f}%")

# ============ 10. ADVANCED: MULTI-FEATURE CLUSTERING ============
print("\n" + "="*60)
print("ADVANCED: 3D CLUSTERING WITH SHARPE RATIO")
print("="*60)

# Use 3 features for clustering
features_advanced = ['Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio']
X_advanced = metrics_df[features_advanced].copy().dropna()
scaler_advanced = StandardScaler()
X_advanced_scaled = scaler_advanced.fit_transform(X_advanced)

# Find optimal k for advanced clustering
inertias_advanced = []
for k in range(2, 11):
    kmeans_adv = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_adv.fit(X_advanced_scaled)
    inertias_advanced.append(kmeans_adv.inertia_)

# Apply clustering with k=3
kmeans_advanced = KMeans(n_clusters=3, random_state=42, n_init=10)
metrics_df['Cluster_Advanced'] = kmeans_advanced.fit_predict(X_advanced_scaled)

# 3D Visualization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue', 'green', 'purple', 'orange']
for cluster in range(3):
    cluster_data = metrics_df[metrics_df['Cluster_Advanced'] == cluster]
    ax.scatter(
        cluster_data['Annual_Return'],
        cluster_data['Annual_Volatility'],
        cluster_data['Sharpe_Ratio'],
        c=colors[cluster],
        s=100,
        label=f'Cluster {cluster}',
        alpha=0.7,
        edgecolors='black'
    )

ax.set_xlabel('Annual Return', fontsize=10)
ax.set_ylabel('Annual Volatility', fontsize=10)
ax.set_zlabel('Sharpe Ratio', fontsize=10)
ax.set_title('3D Stock Clustering: Return vs Risk vs Sharpe Ratio', fontsize=12)
ax.legend()

plt.tight_layout()
plt.show()

# ============ 11. INVESTMENT RECOMMENDATIONS ============
print("\n" + "="*60)
print("INVESTMENT RECOMMENDATIONS")
print("="*60)

# Identify best performers based on Sharpe ratio
best_sharpe = metrics_df.nlargest(5, 'Sharpe_Ratio')[['Ticker', 'Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio']]
print("\n🏆 Top 5 Stocks by Risk-Adjusted Returns (Sharpe Ratio):")
print(best_sharpe.to_string(index=False))

# Identify stable stocks (low volatility)
stable_stocks = metrics_df.nsmallest(5, 'Annual_Volatility')[['Ticker', 'Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio']]
print("\n🛡️  Most Stable Stocks (Lowest Volatility):")
print(stable_stocks.to_string(index=False))

# Identify high growth stocks
growth_stocks = metrics_df.nlargest(5, 'Annual_Return')[['Ticker', 'Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio']]
print("\n📈 Highest Growth Stocks (Best Returns):")
print(growth_stocks.to_string(index=False))

# Portfolio suggestions
print("\n" + "-"*60)
print("📊 SUGGESTED PORTFOLIOS:")
print("-"*60)

print("\n1. CONSERVATIVE PORTFOLIO (Low Risk):")
conservative = metrics_df.nsmallest(3, 'Annual_Volatility')['Ticker'].tolist()
print(f"   {', '.join(conservative)}")

print("\n2. AGGRESSIVE PORTFOLIO (High Return Potential):")
aggressive = metrics_df.nlargest(3, 'Annual_Return')['Ticker'].tolist()
print(f"   {', '.join(aggressive)}")

print("\n3. BALANCED PORTFOLIO (Best Risk-Adjusted):")
balanced = metrics_df.nlargest(3, 'Sharpe_Ratio')['Ticker'].tolist()
print(f"   {', '.join(balanced)}")

# ============ 12. EXPORT RESULTS ============
# Save results to CSV
metrics_df.to_csv('stock_clustering_results.csv', index=False)
print("\n" + "="*60)
print("✓ Results exported to 'stock_clustering_results.csv'")
print("="*60)