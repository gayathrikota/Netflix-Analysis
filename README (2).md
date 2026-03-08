# 🎬 Netflix Content Analysis & Machine Learning
### Python · Pandas · Scikit-learn · Random Forest · K-Means · PCA · Seaborn

> **Objective:** Analyze 8,807+ Netflix titles to uncover content trends and build machine learning models that classify and cluster titles based on metadata features.

---

## 📊 Project Highlights

| Model | Task | Key Result |
|-------|------|-----------|
| 🌲 Random Forest | Classify Movie vs TV Show | `duration_num` = #1 most important feature |
| 📍 K-Means Clustering | Group titles into natural clusters | 4 clusters found via elbow method |
| 🔍 PCA | Visualize clusters in 2D | Clear separation between content groups |

---

## 🗂️ Project Structure

```
netflix-content-analysis/
│
├── netflix_analysis.ipynb     # Full analysis notebook (EDA + ML)
├── netflix_titles.csv         # Dataset — 8,807 Netflix titles from Kaggle
└── README.md
```

---

## 🛠️ Tools & Libraries

| Library | Purpose |
|---------|---------|
| **Pandas** | Data loading, cleaning, feature engineering |
| **NumPy** | Numerical operations |
| **Matplotlib / Seaborn** | EDA visualizations (countplot, histplot, boxplot, heatmap) |
| **Scikit-learn** | Random Forest, K-Means, PCA, StandardScaler, train_test_split |
| **Collections.Counter** | Genre frequency counting |

---

## 📋 Analysis Breakdown

### 1. Background & Problem Definition
Working with the Netflix Movies and TV Shows dataset from Kaggle containing **8,807 rows and 12 columns**. Each row is one Netflix title with fields like type, title, director, cast, country, release year, rating, duration, and genre.

**Two core problems:**
- **Classification:** Can we predict if a title is a Movie or TV Show using metadata?
- **Clustering:** Can we find natural groupings of titles based on content features?

---

### 2. Data Cleaning
Handled missing values and created new numeric features:

| Step | What was done | Why |
|------|--------------|-----|
| Fill NaN in `country`, `rating`, `listed_in` | Replaced with `"Unknown"` | Prevents errors in ML pipeline |
| `parse_duration()` function | Extracted numeric value from strings like `"90 min"` or `"2 Seasons"` | ML models need numbers, not strings |
| Fill remaining NaN in `duration_num` | Used **median per type** (Movies vs TV Shows separately) | More accurate than a single overall median |
| `num_countries` | Count of countries per title | New feature for ML |
| `num_genres` | Count of genres per title | New feature for ML |

---

### 3. Exploratory Data Analysis (EDA)

**Key findings:**
- 📽️ **Movies outnumber TV Shows** significantly in the dataset
- 📅 **Sharp increase in titles after 2010** — Netflix content explosion
- ⏱️ **Most movies:** 80–120 minutes | **Most TV shows:** 1–2 seasons
- 🎭 **Top genres:** International Movies, Dramas, Comedies dominate

---

### 4. Classification — Random Forest

**Why Random Forest?**
- Handles mixed data types well
- Robust to noisy data
- Provides feature importance scores — great for interpretation

**Features used:**
- `release_year`, `duration_num`, `num_countries`, `num_genres`
- `rating` → one-hot encoded (e.g., TV-MA, PG-13, R)

**Model setup:**
```python
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
```

**Result:** `duration_num` was the single most important feature by far — makes sense because movies are measured in minutes while TV shows are measured in seasons.

---

### 5. Clustering — K-Means

**Why K-Means?**
- Unsupervised — finds patterns without needing labels
- Simple, interpretable, and scalable

**Features used:** `release_year`, `duration_num`, `num_genres`

**Why StandardScaler first?**
- K-Means uses distance calculations
- Without scaling, `duration_num` (0–300+ minutes) would dominate over `num_genres` (1–3)
- Scaling puts all features on the same range so each contributes equally

**Elbow Method → chose K=4**

```python
# Elbow method to find optimal K
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(cluster_scaled)
    inertias.append(km.inertia_)
```

**4 Clusters discovered:**

| Cluster | Profile |
|---------|---------|
| 0 | Older movies with fewer genres — classic catalog content |
| 1 | Short, newer titles with niche genre combinations |
| 2 | Long multi-genre titles — prestige TV shows |
| 3 | Recent high-volume content — mainstream movies & shows |

**PCA:** Used to compress 3 features → 2 dimensions for scatter plot visualization of clusters.

---

### 6. Conclusion

- ✅ Random Forest can **accurately classify** Movie vs TV Show using simple metadata
- ✅ `duration_num` is the **strongest predictor** by a wide margin
- ✅ K-Means reveals **4 meaningful content groups** that reflect real Netflix catalog patterns
- ✅ Netflix's catalog is **dominated by post-2010 content** with International Movies and Dramas as the top categories

---

## 👩‍💻 Author

**Gayathri Kota**  
B.S. Data Science | Arizona State University  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-gayathrikota-blue?style=flat&logo=linkedin)](https://linkedin.com/in/gayathrikota)
[![GitHub](https://img.shields.io/badge/GitHub-gayathrikota-black?style=flat&logo=github)](https://github.com/gayathrikota)
