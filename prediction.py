# ============================================================
# SECTION 1 — DATA LOADING, CLEANING & PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1.1 — Load Raw Data
# ------------------------------------------------------------
DATA_FILE = "Dataset.xlsx"   # Make sure this file is in the same folder

print("Loading data from Excel...")

company_df = pd.read_excel(DATA_FILE, sheet_name="Company_Master")
daily_df = pd.read_excel(DATA_FILE, sheet_name="Daily_Prices")

print("Company_Master rows:", len(company_df))
print("Daily_Prices rows:", len(daily_df))
print("Unique symbols in Daily_Prices:", daily_df['Symbol'].nunique())
print()

# ------------------------------------------------------------
# 1.2 — Basic Cleaning: Trim spaces in column names (safety)
# ------------------------------------------------------------
company_df.columns = company_df.columns.str.strip()
daily_df.columns = daily_df.columns.str.strip()

# ------------------------------------------------------------
# 1.3 — Convert Date column to proper datetime
# ------------------------------------------------------------
print("Converting Date column to datetime...")

daily_df['Date'] = pd.to_datetime(daily_df['Date'], errors='coerce')

# Drop rows where Date could not be parsed
before_date = len(daily_df)
daily_df = daily_df.dropna(subset=['Date'])
after_date = len(daily_df)
print(f"Dropped {before_date - after_date} rows with invalid dates.")
print()

# ------------------------------------------------------------
# 1.4 — Convert Volume from text with M/B/K into numeric
#        Example: '58.67M' → 58670000
#                 '2.3B'   → 2300000000
#                 '900K'   → 900000
# ------------------------------------------------------------
def convert_volume(v):
    """
    Convert volume text like '58.67M', '2.3B', '900K' to numeric.
    Handles missing or malformed values safely.
    """
    if pd.isna(v):
        return np.nan
    
    s = str(v).replace(",", "").strip()
    if s == "" or s.lower() in ["nan", "none", "-"]:
        return np.nan

    multiplier = 1.0
    if s.endswith("M"):
        multiplier = 1_000_000
        s = s[:-1]
    elif s.endswith("B"):
        multiplier = 1_000_000_000
        s = s[:-1]
    elif s.endswith("K"):
        multiplier = 1_000

    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan

print("Converting Vol. column to numeric...")
daily_df['Vol.'] = daily_df['Vol.'].apply(convert_volume)

# ------------------------------------------------------------
# 1.5 — Ensure numeric types for OHLC and Change %
# ------------------------------------------------------------
numeric_cols = ["Open", "High", "Low", "Close", "Change %", "Vol."]

for col in numeric_cols:
    if col in daily_df.columns:
        daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')

print("Converted numeric columns:", numeric_cols)
print()

# ------------------------------------------------------------
# 1.6 — Sort data by Symbol and Date
# ------------------------------------------------------------
daily_df = daily_df.sort_values(by=["Symbol", "Date"]).reset_index(drop=True)

# ------------------------------------------------------------
# 1.7 — Feature Engineering
#       - Daily_Return: percentage change in Close
#       - Rolling_30D_Volatility: std dev of returns (30-day window)
#       - MA_20, MA_50: 20-day & 50-day moving averages of Close
#       - Direction: 1 if return > 0 else 0 (for classification)
# ------------------------------------------------------------

print("Creating engineered features (returns, volatility, moving averages)...")

# Group by Symbol so that rolling calculations are per stock
grouped = daily_df.groupby("Symbol", group_keys=False)

# Daily return (percentage change)
daily_df["Daily_Return"] = grouped["Close"].pct_change()

# Rolling 30-day volatility of returns
daily_df["Rolling_30D_Volatility"] = grouped["Daily_Return"].transform(
    lambda x: x.rolling(window=30, min_periods=5).std()
)

# Moving averages
for window in [20, 50]:
    daily_df[f"MA_{window}"] = grouped["Close"].transform(
        lambda x: x.rolling(window=window, min_periods=5).mean()
    )

# Direction label: Up(1) if return > 0, Down(0) if return <= 0
daily_df["Direction"] = np.where(daily_df["Daily_Return"] > 0, 1, 0)

print("Feature engineering complete.")
print()

# ------------------------------------------------------------
# 1.8 — Handle Missing Values (Basic Strategy)
#       We will drop rows where critical columns are missing.
#       This can be improved later if needed.
# ------------------------------------------------------------
critical_cols = ["Open", "High", "Low", "Close", "Vol.", "Daily_Return", "MA_20", "MA_50"]

before_drop = len(daily_df)
daily_df_clean = daily_df.dropna(subset=critical_cols)
after_drop = len(daily_df_clean)

print(f"Dropped {before_drop - after_drop} rows with missing critical values.")
print("Final cleaned rows:", after_drop)
print()

# ------------------------------------------------------------
# 1.9 — Save Cleaned Data for Later Use (and for Power BI)
# ------------------------------------------------------------
output_clean_file = "cleaned_full_dataset.xlsx"
daily_df_clean.to_excel(output_clean_file, sheet_name="cleaned_full_dataset", index=False)
print(f"Cleaned daily prices saved to: {output_clean_file}")
print()

# ------------------------------------------------------------
# 1.10 — Choose Ticker for ML Analysis (with default = AAPL)
# ------------------------------------------------------------
available_symbols = sorted(daily_df_clean["Symbol"].dropna().unique().tolist())
print("Available symbols (total:", len(available_symbols), "):")
print(available_symbols)
print()

default_symbol = "AAPL"
user_input = input(f"Enter stock ticker for ML analysis [default={default_symbol}]: ").strip().upper()

if user_input == "":
    chosen_symbol = default_symbol
elif user_input in available_symbols:
    chosen_symbol = user_input
else:
    print(f"Symbol '{user_input}' not found. Falling back to default: {default_symbol}")
    chosen_symbol = default_symbol

print(f"\nChosen symbol for analysis: {chosen_symbol}")

# Filter data for chosen symbol (this will be used in later sections)
symbol_df = daily_df_clean[daily_df_clean["Symbol"] == chosen_symbol].copy()

print("Rows for chosen symbol:", len(symbol_df))
print("Date range:", symbol_df["Date"].min(), "to", symbol_df["Date"].max())
print("Preview of cleaned data for", chosen_symbol)
print(symbol_df.head())

# ============================================================
# SECTION 2 — REGRESSION MODELS (SIMPLE, MULTIPLE, POLYNOMIAL)
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n============================================================")
print("SECTION 2 — Regression Models")
print("============================================================")

# ------------------------------------------------------------
# 2.0 — Prepare Data for Regression
# ------------------------------------------------------------

# Use Close price as target variable
target = "Close"

# Drop rows with missing values in regressors or target
reg_df = symbol_df.dropna(subset=["Open", "High", "Low", "Vol.", 
                                  "Daily_Return", "MA_20", "MA_50", "Close"])

# Reset index for clean splits
reg_df = reg_df.reset_index(drop=True)

print("Rows available for regression:", len(reg_df))

# ------------------------------------------------------------
# 2.1 — Simple Linear Regression (Close vs Open)
# ------------------------------------------------------------
print("\n2.1 — Simple Linear Regression (Close vs Open)")

X_simple = reg_df[["Open"]]
y = reg_df[target]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, shuffle=False)

simple_reg = LinearRegression()
simple_reg.fit(X_train_s, y_train_s)

y_pred_s = simple_reg.predict(X_test_s)

print("Simple Linear Regression Coefficient:", simple_reg.coef_[0])
print("Intercept:", simple_reg.intercept_)

# Evaluation Metrics
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
rmse_s = mse_s ** 0.5
r2_s = r2_score(y_test_s, y_pred_s)

print("MAE:", mae_s)
print("MSE:", mse_s)
print("RMSE:", rmse_s)
print("R²:", r2_s)

# ------------------------------------------------------------
# 2.2 — Multiple Linear Regression
# ------------------------------------------------------------
print("\n2.2 — Multiple Linear Regression")

features_multi = ["Open", "High", "Low", "Vol.", "Daily_Return", "MA_20", "MA_50"]

X_multi = reg_df[features_multi]
y = reg_df[target]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, shuffle=False)

multi_reg = LinearRegression()
multi_reg.fit(X_train_m, y_train_m)

y_pred_m = multi_reg.predict(X_test_m)

# Evaluation
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
rmse_m = mse_m ** 0.5
r2_m = r2_score(y_test_m, y_pred_m)

print("MAE:", mae_m)
print("MSE:", mse_m)
print("RMSE:", rmse_m)
print("R²:", r2_m)

# ------------------------------------------------------------
# 2.3 — Polynomial Regression (degree = 2)
# ------------------------------------------------------------
print("\n2.3 — Polynomial Regression (degree = 2)")

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(reg_df[["Open"]])   # polynomial on 1 feature

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, shuffle=False)

poly_reg = LinearRegression()
poly_reg.fit(X_train_p, y_train_p)

y_pred_p = poly_reg.predict(X_test_p)

# Metrics
mae_p = mean_absolute_error(y_test_p, y_pred_p)
mse_p = mean_squared_error(y_test_p, y_pred_p)
rmse_p = mse_p ** 0.5
r2_p = r2_score(y_test_p, y_pred_p)

print("MAE:", mae_p)
print("MSE:", mse_p)
print("RMSE:", rmse_p)
print("R²:", r2_p)

# ------------------------------------------------------------
# 2.4 — Summary Comparison Table
# ------------------------------------------------------------

print("\nRegression Model Performance Summary:")
print("------------------------------------------------------------")
print(f"Simple Linear Regression   RMSE={rmse_s:.4f}, R²={r2_s:.4f}")
print(f"Multiple Linear Regression RMSE={rmse_m:.4f}, R²={r2_m:.4f}")
print(f"Polynomial Regression      RMSE={rmse_p:.4f}, R²={r2_p:.4f}")

# ------------------------------------------------------------
# 2.5 — Plot Actual vs Predicted (Multiple Regression)
# ------------------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(y_test_m.values, label="Actual Close", linewidth=2)
plt.plot(y_pred_m, label="Predicted Close (Multiple Regression)", linewidth=2)
plt.title(f"Actual vs Predicted Close Price — {chosen_symbol}")
plt.xlabel("Test Sample Index")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# SECTION 3 — CLASSIFICATION MODELS
# KNN, Naive Bayes, Decision Tree, SVM
# ============================================================

print("\n============================================================")
print("SECTION 3 — Classification Models")
print("============================================================")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
import seaborn as sns


# ------------------------------------------------------------
# 3.0 — Prepare Features for Classification
# ------------------------------------------------------------

# Features chosen for classification
class_features = ["Open", "High", "Low", "Close", "Vol.", "Daily_Return",
                  "MA_20", "MA_50", "Rolling_30D_Volatility"]

# Remove missing rows
class_df = symbol_df.dropna(subset=class_features + ["Direction"]).reset_index(drop=True)

X = class_df[class_features]
y = class_df["Direction"]

# Normalize features (important for KNN, SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split (time-series -> no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Helper function to evaluate models
def evaluate_model(model_name, y_true, y_pred, prob_pred=None):
    print(f"\n--- {model_name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # AUC score (for models with probability output)
    if prob_pred is not None:
        auc = roc_auc_score(y_true, prob_pred)
        print("ROC-AUC :", auc)


# ------------------------------------------------------------
# 3.1 — K-Nearest Neighbors (KNN)
# ------------------------------------------------------------
print("\n3.1 — K-Nearest Neighbors (KNN)")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

# KNN does NOT have predict_proba for non-classification? It does → use it.
y_proba_knn = knn.predict_proba(X_test)[:, 1]

evaluate_model("KNN Classifier", y_test, y_pred_knn, y_proba_knn)


# ------------------------------------------------------------
# 3.2 — Naive Bayes
# ------------------------------------------------------------
print("\n3.2 — Naive Bayes")

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
y_proba_nb = nb.predict_proba(X_test)[:, 1]

evaluate_model("Naive Bayes", y_test, y_pred_nb, y_proba_nb)


# ------------------------------------------------------------
# 3.3 — Decision Tree Classifier
# ------------------------------------------------------------
print("\n3.3 — Decision Tree Classifier")

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

train_pred_dt = dt.predict(X_train)
print("Train Accuracy:", accuracy_score(y_train, train_pred_dt))

y_pred_dt = dt.predict(X_test)
y_proba_dt = dt.predict_proba(X_test)[:, 1]

evaluate_model("Decision Tree", y_test, y_pred_dt, y_proba_dt)


# ------------------------------------------------------------
# 3.4 — Support Vector Machine (SVM)
# ------------------------------------------------------------
print("\n3.4 — Support Vector Machine (SVM)")

svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
y_proba_svm = svm.predict_proba(X_test)[:, 1]

evaluate_model("SVM Classifier", y_test, y_pred_svm, y_proba_svm)


# ------------------------------------------------------------
# 3.5 — ROC Curve Comparison for all models
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

# Get FPR/TPR for all classifiers
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)

plt.plot(fpr_knn, tpr_knn, label="KNN")
plt.plot(fpr_nb, tpr_nb, label="Naive Bayes")
plt.plot(fpr_dt, tpr_dt, label="Decision Tree")
plt.plot(fpr_svm, tpr_svm, label="SVM")

plt.title(f"ROC Curve Comparison — {chosen_symbol}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# SECTION 4 — UNSUPERVISED LEARNING: CLUSTERING
# K-Means + Hierarchical Clustering
# ============================================================

print("\n============================================================")
print("SECTION 4 — Unsupervised Learning: Clustering")
print("============================================================")

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 4.0 — Prepare Data for Clustering
# ------------------------------------------------------------

cluster_features = ["Daily_Return", "Rolling_30D_Volatility", "Vol.", "MA_20", "MA_50"]

cluster_df = symbol_df.dropna(subset=cluster_features).reset_index(drop=True)

X_cluster = cluster_df[cluster_features]

# Scale features
scaler_c = StandardScaler()
X_cluster_scaled = scaler_c.fit_transform(X_cluster)

print("Clustering dataset shape:", X_cluster_scaled.shape)


# ------------------------------------------------------------
# 4.1 — Elbow Method (to choose optimal k)
# ------------------------------------------------------------
print("\n4.1 — Finding optimal clusters using Elbow Method...")

inertia_values = []
K_range = range(2, 10)   # Try cluster counts from 2 to 9

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia_values, marker='o')
plt.title(f"Elbow Method for Optimal k — {chosen_symbol}")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.grid()
plt.show()

# ------------------------------------------------------------
# 4.2 — Apply K-Means with selected k (k=3 default)
# ------------------------------------------------------------
print("\n4.2 — Applying K-Means Clustering (k=3)")

kmeans_model = KMeans(n_clusters=3, random_state=42)
clusters = kmeans_model.fit_predict(X_cluster_scaled)

cluster_df["Cluster"] = clusters

print("Cluster counts:")
print(cluster_df["Cluster"].value_counts())


# ------------------------------------------------------------
# 4.3 — Plot clusters using Daily_Return vs Volatility
# ------------------------------------------------------------
print("\n4.3 — Plot clusters using Daily_Return vs Volatility")

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    cluster_df["Daily_Return"], 
    cluster_df["Rolling_30D_Volatility"], 
    c=cluster_df["Cluster"], cmap='viridis'
)
plt.title(f"K-Means Clusters — {chosen_symbol}")
plt.xlabel("Daily Return")
plt.ylabel("30-Day Volatility")
plt.colorbar(scatter, label="Cluster")
plt.grid()
plt.show()

# ------------------------------------------------------------
# 4.4 — Hierarchical Clustering (Agglomerative)
# ------------------------------------------------------------
print("\n4.4 — Hierarchical Clustering Dendrogram")

linked = linkage(X_cluster_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title(f"Hierarchical Clustering Dendrogram — {chosen_symbol}")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.xticks(rotation=90)
plt.show()

# ============================================================
# SECTION 5 — DIMENSIONALITY REDUCTION: PCA
# ============================================================

print("\n============================================================")
print("SECTION 5 — PCA (Dimensionality Reduction)")
print("============================================================")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 5.0 — Prepare Data for PCA
# ------------------------------------------------------------

pca_features = ["Open", "High", "Low", "Close", "Vol.", 
                "Daily_Return", "Rolling_30D_Volatility", 
                "MA_20", "MA_50"]

pca_df = symbol_df.dropna(subset=pca_features).reset_index(drop=True)

X_pca = pca_df[pca_features]

# Standardize (Required for PCA)
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca)

print("Dataset shape for PCA:", X_pca_scaled.shape)


# ------------------------------------------------------------
# 5.1 — Apply PCA (keep 2 components for visualization)
# ------------------------------------------------------------

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_pca_scaled)

pca_df["PC1"] = pca_result[:, 0]
pca_df["PC2"] = pca_result[:, 1]


# ------------------------------------------------------------
# 5.2 — Explained Variance Ratio
# ------------------------------------------------------------

print("\nExplained Variance by PCA Components:")
print("PC1:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")

total_var = pca.explained_variance_ratio_.sum() * 100
print("Total Variance Explained by PC1 + PC2:", round(total_var, 2), "%")


# ------------------------------------------------------------
# 5.3 — PCA Scatter Plot (PC1 vs PC2)
# ------------------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c='blue', alpha=0.6)
plt.title(f"PCA — Dimension Reduction Scatter Plot ({chosen_symbol})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()

# ============================================================
# SECTION 6 — NEURAL NETWORK (MLP REGRESSOR)
# ============================================================

print("\n============================================================")
print("SECTION 6 — Neural Network (MLP Regressor)")
print("============================================================")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------------------------------------
# 6.0 — Prepare Data for ANN
# ------------------------------------------------------------

ann_features = ["Open", "High", "Low", "Vol.", "Daily_Return", 
                "MA_20", "MA_50", "Rolling_30D_Volatility"]

ann_df = symbol_df.dropna(subset=ann_features + ["Close"]).reset_index(drop=True)

X_ann = ann_df[ann_features]
y_ann = ann_df["Close"]

# Scaling is VERY important for neural networks
scaler_ann = StandardScaler()
X_ann_scaled = scaler_ann.fit_transform(X_ann)

# Train-test split (no shuffle because time-series)
X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(
    X_ann_scaled, y_ann, test_size=0.2, shuffle=False
)

print("Training samples:", len(X_train_ann))
print("Testing samples:", len(X_test_ann))


# ------------------------------------------------------------
# 6.1 — Build and Train MLP Neural Network
# ------------------------------------------------------------
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

mlp = MLPRegressor(
    hidden_layer_sizes=(32, 16),    # 2 layer nueral network
    activation='relu',
    solver='adam',
    max_iter=4000,
    random_state=42,
    learning_rate_init=0.001
)


print("\nTraining Neural Network...")
mlp.fit(X_train_ann, y_train_ann)

print("Training complete.")


# ------------------------------------------------------------
# 6.2 — Predictions and Evaluation Metrics
# ------------------------------------------------------------

y_pred_ann = mlp.predict(X_test_ann)

mse_ann = mean_squared_error(y_test_ann, y_pred_ann)
rmse_ann = mse_ann ** 0.5
r2_ann = r2_score(y_test_ann, y_pred_ann)

print("\nNeural Network Performance:")
print("RMSE:", rmse_ann)
print("R² Score:", r2_ann)


# ------------------------------------------------------------
# 6.3 — Plot Actual vs Predicted
# ------------------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(y_test_ann.values, label="Actual Close", linewidth=2)
plt.plot(y_pred_ann, label="Predicted Close (ANN)", linewidth=2)
plt.title(f"Neural Network Prediction — {chosen_symbol}")
plt.xlabel("Test Sample Index")
plt.ylabel("Close Price")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# SECTION 7 — 7-DAY STOCK FORECAST + EXPORT TO EXCEL
# ============================================================

print("\n============================================================")
print("SECTION 7 — 7-Day Forecasting + Export for Power BI")
print("============================================================")

from sklearn.linear_model import LinearRegression
import datetime

# --------------------------
# 7.1 Prepare data for forecasting
# --------------------------

forecast_features = ["Open", "High", "Low", "Vol.", 
                     "Daily_Return", "MA_20", "MA_50", "Rolling_30D_Volatility"]

# Use the same cleaned dataframe from earlier
forecast_df = symbol_df.dropna(subset=forecast_features + ["Close"]).reset_index(drop=True)

X_f = forecast_df[forecast_features]
y_f = forecast_df["Close"]

# Fit regression model for forecasting
forecast_model = LinearRegression()
forecast_model.fit(X_f, y_f)

print("\nForecast model trained successfully.")

# --------------------------
# 7.2 Generate 7 future dates
# --------------------------

today = datetime.date.today()
future_dates = pd.date_range(start=today + datetime.timedelta(days=1), periods=7)

# --------------------------
# 7.3 Create placeholder rows to compute future features
# --------------------------

future_data = []

prev_row = forecast_df.iloc[-1].copy()

for fut_date in future_dates:
    new_row = {}

    # Use previous day's close as next day's open (simple forecasting assumption)
    new_row["Open"] = prev_row["Close"]

    # Estimate High/Low based on volatility
    new_row["High"] = new_row["Open"] * (1 + prev_row["Rolling_30D_Volatility"])
    new_row["Low"] = new_row["Open"] * (1 - prev_row["Rolling_30D_Volatility"])

    # Carry forward last known volume
    new_row["Vol."] = prev_row["Vol."]

    # Calculate Daily_Return
    new_row["Daily_Return"] = (new_row["Open"] - prev_row["Close"]) / prev_row["Close"]

    # Rolling volatility stays same for forecasting window
    new_row["Rolling_30D_Volatility"] = prev_row["Rolling_30D_Volatility"]

    # Moving averages also approximated using last known values
    new_row["MA_20"] = prev_row["MA_20"]
    new_row["MA_50"] = prev_row["MA_50"]

    # Save row
    new_row["Date"] = fut_date

    # Convert to DataFrame for prediction
    temp_df = pd.DataFrame([new_row])

    # 2. Predict close price
    predicted_close = forecast_model.predict(temp_df[forecast_features])[0]

    # 3. Store prediction
    new_row["Predicted_Close"] = predicted_close

    # 4. Add to future list
    future_data.append(new_row)

    # 5. Update prev_row with predicted Close for next day's Open
    prev_row = pd.Series(new_row)
    prev_row["Close"] = predicted_close

future_df = pd.DataFrame(future_data)

# --------------------------
# 7.4 Predict CLOSE price for future rows
# --------------------------

future_X = future_df[forecast_features]
future_df["Predicted_Close"] = forecast_model.predict(future_X)

print("\n7-Day Forecast Completed Successfully!\n")

print(future_df[["Date", "Predicted_Close"]])

# --------------------------
# 7.5 Export cleaned dataset + forecast dataset for Power BI
# --------------------------

future_df["Symbol"] = chosen_symbol

# Find company name from Company_Master
company_name = company_df.loc[company_df["Symbol"] == chosen_symbol, "Company Name"].values
company_name = company_name[0] if len(company_name) > 0 else "Unknown"
future_df["Company Name"] = company_name

symbol_df["Symbol"] = chosen_symbol
symbol_df["Company Name"] = company_name

clean_output = "cleaned_selected_stock.xlsx"
forecast_output = "forecast_selected_stock.xlsx"

symbol_df.to_excel(clean_output, sheet_name="cleaned_selected_stock", index=False)
future_df.to_excel(forecast_output, sheet_name="forecast_selected_stock", index=False)

print("\nExported the following files for Power BI:")
print(f"1. Cleaned Dataset     → {clean_output}")
print(f"2. 7-Day Forecast Data → {forecast_output}")

# ============================================================
# SECTION 8 — LSTM DEEP LEARNING FORECASTING (CLOSE PRICE ONLY)
# ============================================================

print("\n============================================================")
print("SECTION 8 — LSTM Deep Learning Model (Close Price Forecasting)")
print("============================================================")

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# ============================================================
# SECTION 8 — MULTIVARIATE HYBRID GRU+LSTM FORECASTING (CLOSE)
# ============================================================

print("\n============================================================")
print("SECTION 8 — Multivariate GRU+LSTM Deep Learning Model")
print("============================================================")

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# ============================================================
# SECTION 8 — OPTIMIZED MULTIVARIATE GRU+LSTM FORECASTING
# ============================================================

print("\n============================================================")
print("SECTION 8 — Optimized Multivariate GRU+LSTM Model")
print("============================================================")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# ------------------------------------------------------------
# 8.1 — Choose the BEST features for LSTM forecasting
# ------------------------------------------------------------

lstm_features = ["Open", "High", "Low", "Close", "Vol."]

lstm_df = symbol_df[["Date"] + lstm_features].dropna().reset_index(drop=True)

print("Total rows for LSTM:", len(lstm_df))

# ------------------------------------------------------------
# 8.2 — Scale multivariate features
# ------------------------------------------------------------

scaler_lstm = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler_lstm.fit_transform(lstm_df[lstm_features])

# ------------------------------------------------------------
# 8.3 — Create 90-day lookback sequences
# ------------------------------------------------------------

sequence_length = 90
X_lstm, y_lstm = [], []

for i in range(sequence_length, len(scaled_data)):
    X_lstm.append(scaled_data[i-sequence_length:i])
    y_lstm.append(scaled_data[i, lstm_features.index("Close")])  # predict close only

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

print("X_lstm shape:", X_lstm.shape)
print("y_lstm shape:", y_lstm.shape)

# ------------------------------------------------------------
# 8.4 — Train-Test Split
# ------------------------------------------------------------

train_size = int(len(X_lstm) * 0.8)

X_train_lstm = X_lstm[:train_size]
y_train_lstm = y_lstm[:train_size]

X_test_lstm = X_lstm[train_size:]
y_test_lstm = y_lstm[train_size:]

print("Train samples:", len(X_train_lstm))
print("Test samples:", len(X_test_lstm))

# ------------------------------------------------------------
# 8.5 — Build Optimized GRU+LSTM Model
# ------------------------------------------------------------

model_lstm = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),

    LSTM(64, return_sequences=False),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')

print("\nTraining Hybrid GRU+LSTM model (100 epochs)...")
history = model_lstm.fit(
    X_train_lstm, y_train_lstm,
    epochs=80,
    batch_size=16,
    validation_data=(X_test_lstm, y_test_lstm),
    verbose=1
)

print("\nTraining Complete.")

# ------------------------------------------------------------
# 8.6 — Evaluate GRU+LSTM Performance on Test Data
# ------------------------------------------------------------

# Model output is only the scaled "Close" value
pred_scaled = model_lstm.predict(X_test_lstm)  # shape: (n_samples, 1)

# Index of "Close" in our feature list
close_idx = lstm_features.index("Close")

# ----- Build full feature vectors for inverse scaling -----
# For predictions
pred_full_scaled = np.zeros((len(pred_scaled), len(lstm_features)))
pred_full_scaled[:, close_idx] = pred_scaled[:, 0]

# For actual y values
y_test_scaled = y_test_lstm.reshape(-1, 1)
actual_full_scaled = np.zeros((len(y_test_scaled), len(lstm_features)))
actual_full_scaled[:, close_idx] = y_test_scaled[:, 0]

# Inverse transform back to original price scale
predicted_full = scaler_lstm.inverse_transform(pred_full_scaled)
actual_full = scaler_lstm.inverse_transform(actual_full_scaled)

predicted_prices = predicted_full[:, close_idx]
actual_prices = actual_full[:, close_idx]

# RMSE calculation
rmse_lstm = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))

print("\nHybrid GRU+LSTM Test Performance:")
print("RMSE:", rmse_lstm)

# ------------------------------------------------------------
# 8.7 — Plot prediction vs actual
# ------------------------------------------------------------

plt.figure(figsize=(12,6))
plt.plot(actual_prices, label="Actual Close Price", linewidth=2)
plt.plot(predicted_prices, label="Hybrid GRU+LSTM Predicted Price", linewidth=2)
plt.title(f"Hybrid GRU+LSTM Prediction vs Actual — {chosen_symbol}")
plt.xlabel("Test Sample Index")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------
# 8.8 — Forecast future N days
# ------------------------------------------------------------

max_forecast = 30
user_days = input(f"\nEnter number of forecast days (1–{max_forecast}) [default=30]: ").strip()

forecast_days = 30 if user_days == "" else int(user_days)
forecast_days = min(forecast_days, max_forecast)

print(f"\nGenerating {forecast_days}-Day Forecast using GRU+LSTM...\n")

last_seq = scaled_data[-sequence_length:]
future_predictions = []

current_seq = last_seq.copy()

for _ in range(forecast_days):
    pred_scaled = model_lstm.predict(current_seq.reshape(1, sequence_length, len(lstm_features)))[0]

    # Insert predicted close price into feature vector
    pred_full = np.zeros(len(lstm_features))
    pred_full[lstm_features.index("Close")] = pred_scaled

    future_predictions.append(pred_full)

    # Add predicted day to sequence
    current_seq = np.vstack([current_seq[1:], pred_full])

# Convert scaled predictions to real values
future_prices = scaler_lstm.inverse_transform(future_predictions)[:, lstm_features.index("Close")]

# Create future dates
future_dates_lstm = pd.date_range(
    start=lstm_df["Date"].max() + pd.Timedelta(days=1),
    periods=forecast_days
)

# Add stock info
company_name = company_df.loc[company_df["Symbol"] == chosen_symbol, "Company Name"].values
company_name = company_name[0] if len(company_name) > 0 else chosen_symbol

lstm_forecast_df = pd.DataFrame({
    "Company Name": company_name,
    "Symbol": chosen_symbol,
    "Date": future_dates_lstm,
    "LSTM_Predicted_Close": future_prices
})

print("GRU+LSTM Forecast Completed Successfully!")
print(lstm_forecast_df.head())

# ------------------------------------------------------------
# 8.9 — Export Forecast
# ------------------------------------------------------------

lstm_output = "lstm_forecast_selected_stock.xlsx"
lstm_forecast_df.to_excel(lstm_output, sheet_name="lstm_forecast_selected_stock", index=False)

print(f"\nExported LSTM Forecast to Excel → {lstm_output}")

# ============================================================
# SECTION 9 — Multi-Company LSTM Forecast (7, 14, 30 Days)
# ============================================================

print("\n============================================================")
print("SECTION 9 — Multi-Company LSTM Forecast (7, 14, 30 Days)")
print("============================================================")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# --------------------------
# 9.0 — Settings
# --------------------------

sequence_length = 60                 # last 60 closes -> next day
forecast_horizons = [7, 14, 30]      # multi-horizon
max_symbols = None                   # limit during testing (None = all)

# --------------------------
# Helper: Build LSTM model
# --------------------------

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# --------------------------
# 9.1 — Loop over all symbols
# --------------------------

all_forecast_rows = []

all_symbols = sorted(daily_df_clean["Symbol"].dropna().unique().tolist())
if max_symbols is not None:
    all_symbols = all_symbols[:max_symbols]

print(f"Total symbols to process: {len(all_symbols)}\n")

for idx, sym in enumerate(all_symbols, start=1):
    print(f"[{idx}/{len(all_symbols)}] Processing: {sym}")

    sym_df = daily_df_clean[daily_df_clean["Symbol"] == sym].sort_values("Date").copy()
    close_series = sym_df["Close"].dropna().reset_index(drop=True)

    # Check data requirement
    if len(close_series) < sequence_length + 30:
        print(f"Skipped {sym}: insufficient data ({len(close_series)} rows).")
        continue

    # ---- Scale close prices ----
    scaler_sym = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler_sym.fit_transform(close_series.values.reshape(-1, 1))

    # ---- Create training sequences ----
    X_seq = []
    y_seq = []

    for i in range(sequence_length, len(scaled_close)):
        X_seq.append(scaled_close[i-sequence_length:i, 0])
        y_seq.append(scaled_close[i, 0])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Reshape for LSTM
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    # ---- Train LSTM model on full data ----
    model_sym = build_lstm_model((sequence_length, 1))
    model_sym.fit(X_seq, y_seq, epochs=20, batch_size=32, verbose=0)

    # ---- Prepare last 60 days ----
    last_60_scaled = scaled_close[-sequence_length:].flatten()
    last_date = sym_df["Date"].max()

    # ---- Generate forecasts for each horizon ----
    for horizon in forecast_horizons:
        print(f"Forecasting {horizon}-day horizon...")

        future_inputs = last_60_scaled.copy()
        pred_scaled_list = []

        for _ in range(horizon):
            model_input = future_inputs[-sequence_length:].reshape(1, sequence_length, 1)
            pred_scaled = model_sym.predict(model_input, verbose=0)[0, 0]
            pred_scaled_list.append(pred_scaled)
            future_inputs = np.append(future_inputs, pred_scaled)

        # Convert to real prices
        pred_scaled_arr = np.array(pred_scaled_list).reshape(-1, 1)
        pred_prices = scaler_sym.inverse_transform(pred_scaled_arr).flatten()

        # Future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=horizon)

        # Store results
        for d, p in zip(future_dates, pred_prices):
            all_forecast_rows.append({
                "Symbol": sym,
                "Date": d,
                "Horizon_Days": horizon,
                "LSTM_Predicted_Close": float(p)
            })

    print(f"Completed {sym}.\n")

# --------------------------
# 9.2 — Export to Excel
# --------------------------

if len(all_forecast_rows) == 0:
    print("No forecast data generated!")
else:
    final_df = pd.DataFrame(all_forecast_rows)

    output_file = "all_companies_lstm_forecast.xlsx"
    final_df.to_excel(output_file, sheet_name="all_companies_lstm_forecast", index=False)

    print("\n============================================================")
    print("Multi-Company LSTM Forecasting Completed Successfully!")
    print(f"Saved file: {output_file}")
    print("============================================================\n")

    print(final_df.head(10))
