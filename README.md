X = df[['LOC', 'Complexity', 'DomainKnowledge', 'TeamExperience']]
y = df['Effort']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression().fit(X_train, y_train)
knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

print("LR R²:", r2_score(y_test, lr.predict(X_test)))
print("KNN R²:", r2_score(y_test, knn.predict(X_test)))
