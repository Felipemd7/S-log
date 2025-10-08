import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

TIME_INTERVAL = "180s"

BASE_DIR = Path(__file__).resolve().parents[1]
RESOURCES_DIR = BASE_DIR / "data" / "NOVA" / "resources" / TIME_INTERVAL / "classification_data"
input_csv = RESOURCES_DIR / f"classification_FTI_{TIME_INTERVAL}_clusters_10_.csv"
if not input_csv.exists():
    raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_csv}")

data = pd.read_csv(input_csv)
data["target"] = data["target"].fillna("NO_FAILURE")

groups = data["ID"].reset_index()
groups["create_ID"] = groups.ID.apply(lambda x: "_".join(np.array(x.rsplit("_"))[[0, -1]]))
groups = groups.loc[:, ["index", "create_ID"]]
groups1 = groups.groupby(["create_ID"]).groups

y = data["target"].values
data = data.drop(columns=["ID", "target", "Unnamed: 0"], errors="ignore")
X = data.values

le = LabelEncoder()
encoded_y = le.fit_transform(y)
y = encoded_y  

skfold = StratifiedKFold(5)

scores = []
cf_matrix = []

predictions = []

for indecies_train, indecies_test in skfold.split(X, y):
    x_train, x_test = X[indecies_train], X[indecies_test]
    y_train, y_test = y[indecies_train], y[indecies_test]

    # model = RandomForestClassifier(500, min_samples_split=8, min_samples_leaf=4, max_features=45, class_weight={0:0.5, 1:1.5}) # 0.43
    # model = DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=4, max_features=None, class_weight={0:0.6, 1:1}) # 0.4
    # model = LogisticRegression(C=10, class_weight={0:0.5, 1:2}, l1_ratio=0.25)

    # model = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1, max_features=None, class_weight={0:0.5, 1:2}) # 0.4
    # model = RandomForestClassifier(100, min_samples_split=2, min_samples_leaf=1, max_features=10, class_weight={0:0.2, 1:1.}) # 0.43
    model = RandomForestClassifier(100, min_samples_split=2, min_samples_leaf=1)  # 0.43

    # model = RandomForestClassifier(50, min_samples_split=8, min_samples_leaf=4, max_features=40, class_weight={0:0.2, 1:1., 2:1., 3:1.}) # 0.43

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    scores.append(f1_score(y_test, preds, average="macro")) # used for 2 class
    predictions.append(pd.DataFrame((indecies_test, preds, y_test)).T)
    # scores.append(f1_score(y_test, preds, average="macro")) # used for k class
    cf_matrix.append(confusion_matrix(y_test, preds))
print("F1 score individual sequences is {} {}".format(np.mean(scores).round(2), np.std(scores).round(2)))

preds = pd.concat(predictions)
preds = preds.reset_index().iloc[:, 1:]
preds.columns = ["ID", "predictions", "ground_truth"]
preds["test_id"] = groups.create_ID
res_test = preds.groupby("test_id").sum()
res_test.predictions = np.where(res_test.predictions>0, 1, 0)
res_test.ground_truth = np.where(res_test.ground_truth>0, 1, 0)
