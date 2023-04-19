
""" This script involves the KNN algorithm for the project to predict whether a pull request will be merged or not."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Load the data
df = pd.read_csv('commons-math-last.csv')

# Select the features
X = df[['column1', 'column2', 'column3', '...', 'columnN']]
y = df['Yaxis_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the KNN classifier model
# K is the number of Neighbors that you want your model consider
knn = KNeighborsClassifier(n_neighbors= K)
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
#################################################################
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix with purple color
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, cmap='PuRd', cbar=False, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KNN Model')
plt.show()
###################################################################

# Make predictions on the testing set
y_pred_prob = knn.predict_proba(X_test)[:,1]

# Calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)

# Plot the ROC curve
plt.plot(fpr, tpr, color='purple', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KNN Model')
plt.legend()
plt.show()
