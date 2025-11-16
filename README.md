# Load ARFF file
data = arff.load(open("dataset.arff"))
df = pd.DataFrame(data["data"], columns=[a[0] for a in data["attributes"]])

# Select features and target
X = df[['LOC', 'Complexity', 'DomainKnowledge', 'TeamExperience']]
y = df['Effort']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# KNN Model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

# Evaluation
print("Linear Regression R²:", r2_score(y_test, pred_lr))
print("KNN Regression R²:", r2_score(y_test, pred_knn))
