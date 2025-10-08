import pandas as pd
from pomegranate import HiddenMarkovModel, DiscreteDistribution
import numpy as np
from pomegranate import *
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import math

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

TIME_INTERVAL = "180s"
N_STATES = 2
n_clusters = 10  # one in [10, 20, 50, 100]

BASE_DIR = Path(__file__).resolve().parents[1]
RESOURCES_ROOT = BASE_DIR / "data" / "NOVA" / "resources"
RESULTS_ROOT = BASE_DIR / "results"
CLUSTERING_RESULTS_DIR = RESULTS_ROOT / "clustering_results"
CLUSTERING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FAULT_DATA_PATH = BASE_DIR / "data" / "RCA_logs" / "Fault-Injection-Dataset-master" / "nova.tsv"
TARGET_INTERVALS = [TIME_INTERVAL]
CLUSTER_OPTIONS = [30]


def normalize_final_status(val: object) -> str:
    tokens = [tok.strip().upper() for tok in str(val).replace('[', '').replace(']', '').split(',') if tok.strip()]
    return "FAILURE" if any(tok == "FAILURE" for tok in tokens) else "NO_FAILURE"


def safe_log_probability(model, sequence):
    try:
        return model.log_probability(sequence)
    except ValueError:
        return float("-inf")

## The main claim in the paper is that: there are problems that are not logged. In order to find them we need to consider alternative approach, e.g. measuring frequenceies, counts or coocurances.
## Therefore, it is important to

## Structure of the paper:
# 1)


# for TIME_INTERVAL in ["120s", "180s", "240s", "300s"]:
#     for n_clusters in [10, 20, 50, 100]:
from collections import defaultdict
dd = defaultdict()
# for TIME_INTERVAL in ["120s"]:
#     for n_clusters in [10]:



flag = True
for TIME_INTERVAL in TARGET_INTERVALS:
# for TIME_INTERVAL in ["120s"]:
    hmm_sequences_dir = RESOURCES_ROOT / TIME_INTERVAL / "HMM_sequencies"
    for n_clusters in CLUSTER_OPTIONS:
    # for n_clusters in [30]:
        if flag:
            try:
                clustering_scores_path = CLUSTERING_RESULTS_DIR / f"different_clustering_results_AD_{N_STATES}_states.csv"
                po = pd.read_csv(clustering_scores_path)
                po.columns = ["exp_name", "scores"]
                po.index = po.exp_name
                po = po.drop(["exp_name"], axis=1)
                dd = po.to_dict()["scores"]
            except FileNotFoundError:
                dd = defaultdict()
        print(dd)
        flag = False
        for run_id in range(0, 2):
            print("------"*20)
            print("Processsing time interval: {}, with n_clusters {} and round {}".format(TIME_INTERVAL, n_clusters, run_id))
            pom_data = pd.read_csv(FAULT_DATA_PATH, sep="\t")

            sequence_path = hmm_sequences_dir / f"sequential_data{TIME_INTERVAL}_clusters_{n_clusters}_.csv"
            data = pd.read_csv(sequence_path)
            data.sequence = data.sequence.apply(lambda x: [int(j) for j in x[1:-1].replace("\n", "").split()])
            data["final_status_label"] = data["final_status"].apply(normalize_final_status)
            fault_labels = pom_data["FAULT_TYPE"].dropna().to_numpy()
            if fault_labels.size == 0:
                raise ValueError("FAULT_TYPE column in nova.tsv is empty")
            repeats = math.ceil(len(data) / fault_labels.size)
            fault_sequence = np.tile(fault_labels, repeats)[: len(data)]
            data["FAULT_TYPE"] = fault_sequence
            # data.FAULT_TYPE = data.FAULT_TYPE.fillna("good")

            data_normal = data[data.final_status_label == "NO_FAILURE"]

            data_normal_train = data_normal.sample(int(data_normal.shape[0]*0.8))
            data_normal_test = data_normal.loc[list(set(data_normal.index).difference(set(data_normal_train.index)))]
            data_abnormal = data[data.final_status_label == "FAILURE"]

            hmm_model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=N_STATES, X=[x for x in data_normal_train.sequence.values])

            predictions_abnormal_log_proba = []
            predictions_abnormal_trans = []
            predictions_abnormal_emis = []

            predictions_normal_test_log_proba = []
            predictions_normal_test_trans = []
            predictions_normal_test_emis = []

            predictions_normal_train_log_proba = []
            predictions_normal_train_trans = []
            predictions_normal_train_emis = []


            for test_seq in data_abnormal.sequence.values:
                predictions_abnormal_log_proba.append(safe_log_probability(hmm_model, test_seq))

                try:
                    trans, ems = hmm_model.forward_backward(test_seq)
                    predictions_abnormal_trans.append(trans)
                    predictions_abnormal_emis.append(ems)
                except:
                    predictions_abnormal_trans.append(np.nan)
                    predictions_abnormal_emis.append(np.nan)

            for test_seq in data_normal_test.sequence.values:
                predictions_normal_test_log_proba.append(safe_log_probability(hmm_model, test_seq))
                try:
                    trans, ems = hmm_model.forward_backward(test_seq)
                    predictions_normal_test_trans.append(trans)
                    predictions_normal_test_emis.append(ems)
                except:
                    predictions_normal_test_trans.append(np.nan)
                    predictions_normal_test_emis.append(np.nan)

            for test_seq in data_normal_train.sequence.values:
                predictions_normal_train_log_proba.append(safe_log_probability(hmm_model, test_seq))
                try:
                    trans, ems = hmm_model.forward_backward(test_seq)
                    predictions_normal_train_trans.append(trans)
                    predictions_normal_train_emis.append(ems)
                except:
                    predictions_normal_train_trans.append(np.nan)
                    predictions_normal_train_emis.append(np.nan)

            # predictions_normal_test_log_proba = predictions_normal_test_log_proba

            abnormal_p = pd.DataFrame(predictions_abnormal_log_proba).replace(-np.inf, np.nan)
            abnormal_p = abnormal_p.fillna(0)
            #
            sns.distplot(abnormal_p, color="red", label="abnormal")


            normal_p = pd.DataFrame(predictions_normal_test_log_proba).replace(-np.inf, np.nan)
            normal_p = normal_p.fillna(0)
            # sns.distplot(normal_p, color="blue", label="test normal")

            # sns.distplot(predictions_normal_train_log_proba, color="cyan", label="train normal")
            plt.legend()
            #


            def get_prediction(normal_values, target_values):
                normal_values = np.array(normal_values)*(-1)
                target_values = np.array(target_values)*(-1)

                mean = np.mean(normal_values)
                std = np.std(normal_values)

                preds = []
                for x in target_values:
                    if x < mean + 2*std and x > mean - 2*std:
                        preds.append(0)
                    else:
                        preds.append(1)
                return preds

            hmm_test_abnormal = get_prediction(predictions_normal_train_log_proba, predictions_abnormal_log_proba)
            hmm_test_normal = get_prediction(predictions_normal_train_log_proba, predictions_normal_test_log_proba)
            hmm_train_normal = get_prediction(predictions_normal_train_log_proba, predictions_normal_train_log_proba)

            # real_tgt = np.hstack([np.ones(a.shape[0]), np.zeros(b.shape[0]), ])
            # preds = np.hstack([a, b])

            real_tgt = np.hstack([np.ones(len(hmm_test_abnormal)), np.zeros(len(hmm_test_normal))])
            preds = np.hstack([hmm_test_abnormal, hmm_test_normal])

            print("Run {} results".format(run_id))
            print("The F1 score is {}".format(f1_score(real_tgt, preds)))

            dd[TIME_INTERVAL + "_" + str(n_clusters) + "_" +  str(run_id) + "_f1_score_"] = f1_score(real_tgt, preds)
            dd[TIME_INTERVAL + "_" + str(n_clusters) + "_" + str(run_id) + "_precision_score_"] = precision_score(real_tgt, preds)
            dd[TIME_INTERVAL + "_" + str(n_clusters) + "_" + str(run_id) + "_recall_score_"] = recall_score(real_tgt, preds)
            dd[TIME_INTERVAL + "_" + str(n_clusters) + "_" + str(run_id) + "_acc_score_"] = accuracy_score(real_tgt, preds)


            clustering_scores_path = CLUSTERING_RESULTS_DIR / f"different_clustering_results_AD_{N_STATES}_states.csv"
            pd.DataFrame(dd, index=["scores"]).T.to_csv(clustering_scores_path)


        # def path_to_alignment(x, y, path):
        #     """
        #     This function will take in two sequences, and the ML path which is their alignment,
        #     and insert dashes appropriately to make them appear aligned. This consists only of
        #     adding a dash to the model sequence for every insert in the path appropriately, and
        #     a dash in the observed sequence for every delete in the path appropriately.
        #     """
        #
        #     for i, (index, state) in enumerate(path[1:-1]):
        #         name = state.name
        #
        #         if name.startswith('D'):
        #             y = y[:i] + '-' + y[i:]
        #         elif name.startswith('I'):
        #             x = x[:i] + '-' + x[i:]
        #
        #     return x, y


        # sequence = data_abnormal.sequence.values[0]
        # for sequence in data_abnormal.sequence.values:
        #     logp, path = hmm_model.viterbi(sequence)
        #     x, y = path_to_alignment('ACT',''.join(sequence), path )
        #
        # data_abnormal["HMM_preds"] = a
        # data_abnormal["HMM_log"] = predictions_abnormal_log_proba
        # data_normal_test["HMM_preds"] = b
        # data_normal_test["HMM_log"] = predictions_normal_test_log_proba
        #
        #
        # def smooth_(x):
        #     if x == "openstack server create":
        #         return "openstack server reboot"
        #     else:
        #         return x
        #
        # def filter_(x):
        #     if x == "OPENSTACK_WRONG_RETURN_VALUE-INSTANCE":
        #         return "OPENSTACK_WRONG_RETURN_VALUE-INSTANCE"
        #     elif x == "OPENSTACK_THROW_EXCEPTION-INSTANCE":
        #         return "OPENSTACK_THROW_EXCEPTION-INSTANCE"
        #     elif x == "good":
        #         return "good"
        #     else:
        #         return False

        #
        # X = data_abnormal.HMM_log.values.reshape(-1, 1)
        # y = data_abnormal.FAULT_TYPE.apply(lambda x: filter_(x))
        # X = X[y!=False]
        # y = y[y!=False]
        #
        #
        # from sklearn.preprocessing import LabelEncoder
        # le = LabelEncoder()
        #
        # y = le.fit_transform(y)
        #
        #
        # from sklearn.model_selection import StratifiedKFold
        #
        # skfold = StratifiedKFold(10)
        #
        # scores = []
        # cf_matrix = []
        # for indecies_train, indecies_test in skfold.split(X, y):
        #     model = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1, max_features=None,)
        #     x_train, x_test = X[indecies_train], X[indecies_test]
        #     y_train, y_test = y[indecies_train], y[indecies_test]
        #     model.fit(x_train, y_train)
        #     preds = model.predict(x_test)
        #     scores.append(f1_score(y_test, preds, average="macro")) # used for 2 class
        #     cf_matrix.append(confusion_matrix(y_test, preds))


        #
        # sequence = data_abnormal.sequence.values[0]
        # logp, path = hmm_model.viterbi(sequence)
        # sequence = "".join([str(x) for x in sequence])
        # sequence2 = "".join([str(x) for x in data_normal_train.sequence.values[0]])
        # for i, (index, state) in enumerate(path):
        #     if name == 1:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 2:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 3:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 4:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 5:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 6:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 7:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 8:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 9:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 10:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 11:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 12:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 13:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 14:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 15:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 16:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 17:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 18:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     elif name == 19:
        #         sequence2 = sequence2[:i] + "-" + sequence2[i:]
        #     name = state.name[1:]
        #     print(name)
