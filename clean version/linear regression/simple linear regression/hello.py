import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import Parallel, delayed

# -----------------------------
# STEP 1: Load your dataset
# -----------------------------
df = pd.read_csv('NIFTY_50_COMPANIES.csv')


print("Original Shape:", df.shape)

# -----------------------------
# STEP 2: Clean & preprocess
# -----------------------------
df = df.dropna()

# Convert Date to numeric (heavy processing)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].astype('int64') // 10**9

# Encode Ticker
df['Ticker'] = df['Ticker'].astype('category').cat.codes

# -----------------------------
# STEP 3: Create Target
# Predict if next day price goes UP or DOWN
# -----------------------------
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

# -----------------------------
# STEP 4: Features & Scaling
# -----------------------------
X = df.drop(['Target'], axis=1)
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# STEP 5: Heavy Task Function
# -----------------------------
def heavy_task(seed):
    np.random.seed(seed)

    # Random subset (still large)
    idx = np.random.choice(len(X_scaled), size=50000, replace=False)
    X_sub = X_scaled[idx]
    y_sub = y.iloc[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.3, random_state=seed
    )

    # Model 1: Random Forest (heavy)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Model 2: SVM (VERY heavy)
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)

    # -----------------------------
    # EXTRA HEAVY COMPUTATION
    # -----------------------------

    # Large matrix multiplication (RAM + CPU stress)
    big_matrix = np.random.rand(3000, 3000)
    result = np.dot(big_matrix, big_matrix)

    # Artificial loop stress
    total = 0
    for i in range(8000000):
        total += (i * i) % 1234567

    return rf.score(X_test, y_test)

# -----------------------------
# STEP 6: Parallel Execution
# -----------------------------
start = time.time()

results = Parallel(n_jobs=-1)(
    delayed(heavy_task)(i) for i in range(10)
)

end = time.time()

# -----------------------------
# STEP 7: Output
# -----------------------------
print("Results:", results)
print("Time Taken:", end - start, "seconds")