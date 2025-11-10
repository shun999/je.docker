# Import the libraries you'll need
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_excel("/root/workspace/1.2 脳血流と問題カテゴリの関連 - 背外側追加.xlsx", sheet_name="ディアビアイ")


df = df.drop(columns=['subject', 'question', 'q_category', #前半部分
                   'dataset_no', 'sheet_name']) # 後半部分


# 脳血流のオキシヘモグロビンだけ使う場合
df = df.drop(columns=['CH1.1_std', 'CH2.1_std', 'CH3.1_std',
       'CH4.1_std', 'CH5.1_std', 'CH6.1_std', 'CH7.1_std', 'CH8.1_std',
       'CH9.1_std', 'CH10.1_std', 'CH11.1_std', 'CH12.1_std', 'CH13.1_std',
       'CH14.1_std', 'CH15.1_std', 'CH16.1_std', 'CH17.1_std', 'CH18.1_std',
       'CH19.1_std', 'CH20.1_std', 'CH21.1_std', 'CH22.1_std', 'CH1.2_std',
       'CH2.2_std', 'CH3.2_std', 'CH4.2_std', 'CH5.2_std', 'CH6.2_std',
       'CH7.2_std', 'CH8.2_std', 'CH9.2_std', 'CH10.2_std', 'CH11.2_std',
       'CH12.2_std', 'CH13.2_std', 'CH14.2_std', 'CH15.2_std', 'CH16.2_std',
       'CH17.2_std', 'CH18.2_std', 'CH19.2_std', 'CH20.2_std', 'CH21.2_std',
       'CH22.2_std' ]) # 後半部分

df = df.drop(columns=['right_pupil_std', 'CH1_std', 'CH2_std',
       'CH3_std', 'CH4_std', 'CH5_std', 'CH6_std', 'CH7_std', 'CH8_std',
       'CH9_std', 'CH10_std', 'CH11_std', 'CH12_std',
       'CH13_std', 'CH14_std', 'CH15_std', 'CH16_std', 'CH17_std',
       'CH18_std', 'CH19_std', 'CH20_std', 'CH21_std',
       'CH22_std']) # 主要な特徴量のみを残す場合

df['score'] = df['score'].replace(2, 1)

df.columns



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 相関行列の計算
corr_matrix = df.corr(method='pearson')

# ヒートマップの描画
plt.figure(figsize=(30, 24))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


df.columns = [col.replace(':', '_')
                .replace('/', '_')
                .replace('[', '')
                .replace(']', '')
                .replace(' ', '_')
                .replace('.', '_')
                .replace(',', '_')
                .replace('(', '_')
                .replace(')', '_')
                for col in df.columns]


X = df.drop(columns=['score'])
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = CatBoostClassifier(random_state=42, verbose=0)
model.fit(X_train, y_train)



# Define hyperparameter search space
param_dist = {
    "iterations": optuna.distributions.IntDistribution(200, 1000),
    "depth": optuna.distributions.IntDistribution(3, 10),
    "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
    "l2_leaf_reg": optuna.distributions.FloatDistribution(1, 10),
    "bagging_temperature": optuna.distributions.FloatDistribution(0, 1),
    "border_count": optuna.distributions.IntDistribution(32, 255),
}

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

optuna_search = OptunaSearchCV(
    model,
    param_dist,
    cv=cv, # 交差検証の設定
    n_trials=50, # トライアル回数
    scoring='accuracy', # 評価指標
    n_jobs=-1, # 計算するコアの数
    verbose=1
)

optuna_search.fit(X_train, y_train) # モデルの学習
y_pred = optuna_search.predict(X_test) # テストデータに対する予測

print("Best parameters: ", optuna_search.best_params_) # 最良のパラメータの表示
print("Test set accuracy: ", accuracy_score(y_test, y_pred)) # テストデータに対する予測精度の表示
print(accuracy_score(y_train, optuna_search.predict(X_train))) # 学習データに対する予測精度の表示

explainer = shap.TreeExplainer(model=optuna_search.best_estimator_)
shap_values = explainer.shap_values(X)
explanation = explainer(X)


shap_values.shape
shap_values


feat_names = list(X.columns)
shap.plots.violin(shap_values, features=X, feature_names=feat_names, plot_type="layered_violin")

shap.plots.heatmap(explanation, max_display=12)

print(X.columns)
shap.plots.beeswarm(explanation, max_display=20)

# shap.plots.scatter(explanation[:, "left_pupil_std", 1], color=explanation[:, "right_pupil_std", 1])
shap.plots.scatter(explanation[:,11])

shap.plots.beeswarm(explanation.abs, color="shap_red")

shap.plots.bar(explanation.abs.mean(0))

shap.plots.bar(explanation)
shap.plots.bar(explanation.cohorts(2).abs.mean(0))

shap.plots.waterfall(explanation[0])