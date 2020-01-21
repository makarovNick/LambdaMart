from LambdaMart import LambdaMART

from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_svmlight_file
from joblib import Memory
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
mem = Memory("./mycache")

@mem.cache
def get_data(filename):
    print("Loading data ...")
    data = load_svmlight_file(filename, query_id=True)
    return data[0], data[1], data[2]

def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg_score(ground_truth, predictions, k=5):
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []
    print("Scoring ...")
    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

X_train, y_train, qid_train = get_data("data/train.txt")
params = {   'max_depth':  10,
         'learning_rate': 0.2,
          'n_estimators': 3000,
                'n_jobs': -1,
}
model = LambdaMART()
#model.load("model")
model.fit(X_train, y_train, qid_train)

X_test, _, qid_test = get_data("data/test.txt")
preds = model.predict(X_test, qid_test)
subm = pd.read_csv("data/sample_submission.csv")
subm["DocumentId"] = preds
#model.save("model")
subm.to_csv("subm4.csv", index = False)

