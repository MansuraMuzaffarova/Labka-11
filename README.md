# Labka-11
import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
 
df = pd.read_csv("C:/Users/XE/Downloads/datasetage_kz.csv")
 
X = df.drop(columns=['Age'])
y = df['Income']
label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])
X['Married'] = label_encoder.fit_transform(X['Married'])
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Использование CatBoost Pool для обработки категориальных признаков
train_pool = cb.Pool(X_train, label=y_train, cat_features=['Gender', 'Married'])
test_pool = cb.Pool(X_test, label=y_test, cat_features=['Gender', 'Married'])
 
params = {
    'loss_function': 'MAE',
    'iterations': 50,
    'depth': 6,
    'learning_rate': 0.05,
    'colsample_bylevel': 0.9,
    'random_seed': 42,
}
 
model = cb.CatBoostRegressor(**params)
model.fit(train_pool, eval_set=test_pool, verbose=10)
y_pred = model.predict(test_pool)
 
# Оценка модели
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
 
print(f'R-квадрат: {r2}')
print(f'Средняя абсолютная ошибка (MAE): {mae}')
 
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Фактический доход')
plt.ylabel('Предсказанный доход')
plt.title('Фактический vs Предсказанный доход')
plt.show()
 
feature_names = model.feature_names_
feature_importance = model.get_feature_importance()
plt.figure(figsize=(8, 6))
plt.bar(range(len(feature_names)), feature_importance, tick_label=feature_names)
plt.xlabel('Признаки')
plt.ylabel('Важность')
plt.title('Важность признаков')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(8, 6))
df_male = df[df['Gender_Male'] == 1]
df_female = df[df['Gender_Female'] == 1]
 
plt.scatter(df_male['Income'], df_male['Income'], label='Мужчины', alpha=0.5, color='blue')
plt.scatter(df_female['Income'], df_female['Income'], label='Женщины', alpha=0.5, color='pink')
 
plt.xlabel('Фактический доход')
plt.ylabel('Фактический доход')
plt.title('Заработок мужчин и женщин')
plt.legend()
plt.show()
