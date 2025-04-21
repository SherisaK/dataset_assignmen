import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#check datasets and subset selection
try:
    df = pd.read_csv('PCH_Palliative_Care_NATIONAL.csv')
except FileNotFoundError:
    print("Error: 'PCH_Palliative_Care_NATIONAL.csv' not found.")
    exit()

print("Information about the dataset: ")
df.info()

print("\nDescriptive statistivs of numerical columns: ")
print(df.describe())

#using visualization to understand the data:
plt.figure(figsize=(10, 6))
sns.barplot(x='Measure ID', y='National Rate', data=df)
plt.title('National Rates for Palliative Care Measures')
plt.xlabel('Measure ID')
plt.ylabel('National Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('nationalrates_barplot.png')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['National Rate'], kde=True)
plt.title('Distribution of National Rates')
plt.xlabel('National Rate')
plt.ylabel('Frequency')
plt.savefig('nationalrates_distribution.png')


plt.subplot(1, 2, 2)
sns.boxplot(y='National Rate', data=df)
plt.title('Box Plot of National Rates')
plt.ylabel('National Rate')
plt.tight_layout()
plt.savefig('nationalrates_boxplot.png')
plt.show()

#training data and testing
label_encoder = LabelEncoder()
df['Measure_ID_Encoded'] = label_encoder.fit_transform(df['Measure ID'])

X = df[['Measure_ID_Encoded']]
y = df['National Rate']

#divide into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train
model = LinearRegression()
model.fit(X_train, y_train)

#predictions
y_pred = model.predict(X_test)

#evaluate
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")
