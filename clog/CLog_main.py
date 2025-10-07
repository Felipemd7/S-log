import copy
import time
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn as nn

CLOG_DIR = Path(__file__).resolve().parent

sys.path.append(str((CLOG_DIR / "classes").resolve()))

from classes.utils import warm_up_MSP, evaluate_MSP, solver
from clustering import *
from create_dataloaders import create_train_valid_data_loaders, create_test_data_loader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 60s: pad_len 16, epochs 200

# 120s: pad_len 64, epochs 400

# 120s: pad_len 64, epochs 600

# 240s: pad_len 64, epochs (10: 600, 20: 400, 30:

BASE_DIR = Path(__file__).resolve().parents[1]

RESOURCES_ROOT = BASE_DIR / "data" / "NOVA" / "resources"



def discover_available_intervals():
    intervals = []
    for interval_dir in RESOURCES_ROOT.iterdir():
        if not interval_dir.is_dir():
            continue
        name = interval_dir.name
        mask_dir = interval_dir / "masked_train_valid_test"
        split_dir = interval_dir / "train_valid_test_splits"
        if (mask_dir / f"train_normal_{name}.csv").exists() or (split_dir / f"train_normal_{name}.csv").exists():
            intervals.append(name)
    return sorted(intervals)


def _prepare_dataset(df):
    df = df.copy()
    if "MSP_Target" not in df.columns:
        df["MSP_Target"] = 0
    df["MSP_Target"] = df["MSP_Target"].fillna(0).astype("int32")
    mask = df["encoded_labels"].astype(str).str.strip().str.len() > 2
    df = df[mask].reset_index(drop=True)
    if df.empty:
        raise ValueError("No sequences with encoded labels available after filtering.")
    return df


AVAILABLE_INTERVALS = discover_available_intervals()
if not AVAILABLE_INTERVALS:
    raise FileNotFoundError(f"No datasets found under {RESOURCES_ROOT}.")

for TIME_INTERVAL_ in AVAILABLE_INTERVALS:

    TIME_INTERVAL_ = TIME_INTERVAL_

    print("<<<<>>>>>"*10)

    print("CURRENTLY PROCESSING TIME INTERVAL {}".format(TIME_INTERVAL_))

    if TIME_INTERVAL_ == "60s":

        PAD_LEN = 16

    else:

        PAD_LEN = 64

    BATCH_SIZE = 2048

    def get_embeddings_from_clustering(embeddings, epoch):

        return np.hstack(embeddings[epoch])

    def get_cluster_ids(predictions, epoch):

        return np.hstack(predictions[epoch])

    def get_masked_word_prediction_clustering(masked_word_prediction, epoch):

        return np.hstack(masked_word_prediction[epoch])

    def performance_scores_macro(msp_score_estimates, ground_truth, file_name, top_k=10):

        pre = np.argsort(msp_score_estimates, axis=1)

        pre = pre[:, (-1)*top_k:]

        tar = ground_truth

        pppp = []

        for idx, x in enumerate(pre):

            if tar[idx] in x:

                pppp.append(1)

            else:

                pppp.append(0)

        pr = pre[:, -1]

        print("The accuracy on validation normal top-1 is {}".format(accuracy_score(y_true=tar, y_pred=pr)))

        print("The precision on validation normal top-1 is {}".format(precision_score(y_true=tar, y_pred=pr, average="macro")))

        print("The recall on validation normal top-1 is {}".format(recall_score(y_true=tar, y_pred=pr, average="macro")))

        print("The f1 on validation normal top-1 is {}".format(f1_score(y_true=tar, y_pred=pr, average="macro")))

        s = ""

        with open(file_name, "a") as file:

            s += "The accuracy on validation normal top-1 is {}".format(accuracy_score(y_true=tar, y_pred=pr)) + "\n"

            s += "The macro precision on validation normal top-1 is {}".format(precision_score(y_true=tar, y_pred=pr, average="macro")) + "\n"

            s += "The macro recall on validation normal top-1 is {}".format(recall_score(y_true=tar, y_pred=pr, average="macro")) + "\n"

            s += "The macro f1 on validation normal top-1 is {}".format(f1_score(y_true=tar, y_pred=pr, average="macro")) + "\n"

            s += "-----------------"*10 + "\n"

            file.writelines(s)

        print("Accuracy top-{} {}".format(top_k, np.sum(pppp)/len(pppp)))

    def performance_scores_topk(msp_score_estimates, ground_truth, file_name, top_k=10):

        pre = np.argsort(msp_score_estimates, axis=1)

        pre = pre[:, (-1)*top_k:]

        tar = ground_truth

        pppp = []

        for idx, x in enumerate(pre):

            if tar[idx] in x:

                pppp.append(1)

            else:

                pppp.append(0)

        pr = pre[:, -1]

        print("Accuracy top-{} {}".format(top_k, np.sum(pppp)/len(pppp)))

        s = ""

        with open(file_name, "a") as file:

            s += "Accuracy top-{} {}".format(top_k, np.sum(pppp)/len(pppp)) + "\n"

            s += "#########################" * 10 + "\n"

            file.writelines(s)

        return np.sum(pppp)/len(pppp)

    def performance_scores_micro(msp_score_estimates, ground_truth, file_name, top_k=10):

        pre = np.argsort(msp_score_estimates, axis=1)

        pre = pre[:, (-1)*top_k:]

        tar = ground_truth

        pppp = []

        for idx, x in enumerate(pre):

            if tar[idx] in x:

                pppp.append(1)

            else:

                pppp.append(0)

        pr = pre[:, -1]

        print("The accuracy on validation normal top-1 is {}".format(accuracy_score(y_true=tar, y_pred=pr)))

        print("The precision on validation normal top-1 is {}".format(precision_score(y_true=tar, y_pred=pr, average="micro")))

        print("The recall on validation normal top-1 is {}".format(recall_score(y_true=tar, y_pred=pr, average="micro")))

        print("The f1 on validation normal top-1 is {}".format(f1_score(y_true=tar, y_pred=pr, average="micro")))

        s = ""

        with open(file_name, "a") as file:

            s += "The accuracy on validation normal top-1 is {}".format(accuracy_score(y_true=tar, y_pred=pr)) + "\n"

            s += "The micro precision on validation normal top-1 is {}".format(precision_score(y_true=tar, y_pred=pr, average="micro")) + "\n"

            s += "The micro recall on validation normal top-1 is {}".format(recall_score(y_true=tar, y_pred=pr, average="micro")) + "\n"

            s += "The micro f1 on validation normal top-1 is {}".format(f1_score(y_true=tar, y_pred=pr, average="micro")) + "\n"

            s += "-----------------"*10 + "\n"

            file.writelines(s)

        print("Accuracy top-{} {}".format(top_k, np.sum(pppp)/len(pppp)))

    def find_max(clusters):

        return pd.value_counts(clusters).index[0]

    def augument_data_predictions_per_epoch(train_data_main, cluster_train_normal_preds):

        prediction_keys = []

        for epoch in range(clustering_epochs):

            cluster_ids_epoch0 = get_cluster_ids(cluster_train_normal_preds, epoch)

            train_data_main["pred_"+str(epoch+1)] = cluster_ids_epoch0

            prediction_keys.append("pred_"+str(epoch+1))

        train_data_main["pred_kmeans"] = model.kmeans.kmeans_preds

        prediction_keys.append("pred_kmeans")

        return train_data_main, prediction_keys

    def cluster_sequences(train_data_main, prediction_keys):

        train_data_main_tmp = copy.copy(train_data_main)

        col = train_data_main_tmp.columns[2]

        sequences = train_data_main_tmp.groupby([col]).groups

        dic = {}

        for pred in prediction_keys:

            d = {}

            for key in sequences.keys():

                d[key] = train_data_main.loc[sequences[key], pred].value_counts().index[0]

            dic[pred] = d

        d = pd.DataFrame(dic)

        d.columns = prediction_keys

        train_data_main_tmp = train_data_main_tmp.loc[train_data_main_tmp.drop_duplicates([col]).index]

        train_data_main_tmp.index = train_data_main_tmp.loc[:, col]

        train_data_main_tmp.loc[:, prediction_keys] = d

        return train_data_main_tmp

    interval_dir = RESOURCES_ROOT / TIME_INTERVAL_

    candidate_dirs = [

        interval_dir / "masked_train_valid_test",

        interval_dir / "train_valid_test_splits",

    ]

    data_dir = None
    train_normal_path = None
    valid_normal_path = None
    for candidate in candidate_dirs:
        candidate_train = candidate / f"train_normal_{TIME_INTERVAL_}.csv"
        candidate_valid = candidate / f"valid_normal_{TIME_INTERVAL_}.csv"
        if candidate_train.exists() and candidate_valid.exists():
            data_dir = candidate
            train_normal_path = candidate_train
            valid_normal_path = candidate_valid
            break

    if data_dir is None:
        raise FileNotFoundError(
            f"No dataset directory with train/valid splits found for interval {TIME_INTERVAL_}. "
            f"Checked: {candidate_dirs}"
        )

    joint_path = data_dir / f"joint_dataset_{TIME_INTERVAL_}.csv"

    required = [train_normal_path, valid_normal_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required dataset files for interval {TIME_INTERVAL_}: {missing}. "
            "Run clog/2_create_train_test_data.py to prepare train/valid splits before running clog/CLog_main.py."
        )

    if joint_path.exists():
        base_train_df = pd.read_csv(joint_path)
    else:
        print(f"Joint dataset {joint_path} not found. Falling back to train_normal for interval {TIME_INTERVAL_}.")
        base_train_df = pd.read_csv(train_normal_path)

    base_valid_df = pd.read_csv(valid_normal_path)

    base_train_df = _prepare_dataset(base_train_df)
    base_valid_df = _prepare_dataset(base_valid_df)

    # flag = input("CHOOSE MODE OF OPERATION: (1) DEBUGGING OR (2) EVALUATION")

    #

    # if flag == '1':

    #     train_data_main = pd.read_csv(path+"joint_dataset_"+TIME_INTERVAL_+".csv").iloc[:10000]   ### THIS IS FOR DEBUGGING PURPOSES

    #     valid_normal_data_main = pd.read_csv(path+"train_normal_"+TIME_INTERVAL_+".csv").iloc[:10000]

    #

    #     warmup_epochs = 2

    #     clustering_epochs = 2

    #

    # elif flag == '2':

    #     train_data_main = pd.read_csv(path+"joint_dataset_"+TIME_INTERVAL_+".csv")

    #     valid_normal_data_main = pd.read_csv(path + "train_normal_" + TIME_INTERVAL_ + ".csv")

    #

    #     warmup_epochs = 400

    #     clustering_epochs = 5

    # test_normal_data_main = pd.read_csv(path+"train_normal_"+TIME_INTERVAL_+".csv")

    # valid_abnormal_data_main = pd.read_csv(path+"train_normal_"+TIME_INTERVAL_+".csv")

    # test_abnormal_data_main = pd.read_csv(path+"train_normal_"+TIME_INTERVAL_+".csv")

    for n_clusters in [30]:

        print("*****************"*10)

        print("*****************"*10)

        print("*****************"*10)

        print("CURRENTLY PROCESSING N_CLUSTERS {}".format(n_clusters))

        train_data_main = base_train_df.copy()

        valid_normal_data_main = base_valid_df.copy()

        if TIME_INTERVAL_ in ["60s", "120s"]:

            warmup_epochs = 200

            clustering_epochs = 5

        else:

            warmup_epochs = 200

            clustering_epochs = 5

        train_load = train_data_main.encoded_labels.apply(lambda x: np.array(x[1:-1].rsplit(",")).astype("int32")).values

        train_labels = train_data_main.MSP_Target.values

        train_data_loader_for_training, train_data_loader_val = create_train_valid_data_loaders(load_train=train_load,

                                                                                                labels_train=train_labels,

                                                                                                load_test=train_load,

                                                                                                labels_test=train_labels,

                                                                                                pad_len=PAD_LEN,

                                                                                                batch_size=BATCH_SIZE)

        valid_normal_load = valid_normal_data_main.encoded_labels.apply(lambda x: np.array(x[1:-1].rsplit(",")).astype("int32")).values

        valid_normal_labels = valid_normal_data_main.MSP_Target.values

        valid_data_loader_normal = create_test_data_loader(

                        load_test=valid_normal_load,

                        labels_test=valid_normal_labels,

                        pad_len=PAD_LEN,

                        batch_size=BATCH_SIZE

        )

        number_heads = 4

        number_layers = 2

        input_log_events = output_size = 471  # CHANGE THIS TO THE NUMBER OF TOKENS YOU HAVE IN THE INPUT

        d_model = 128  #

        size_feedforward = d_model

        dropout = 0.01

        max_len_of_input = PAD_LEN

        l2_regularization = 0.00001

        learning_rate = 0.0001

        adam_betas_b1 = 0.9

        adam_betas_b2 = 0.999

        device = "cuda" if torch.cuda.is_available() else "cpu"

        beta = 0.00001

        lambda_ = 0.1

        # criterion = nn.CrossEntropyLoss(weight=torch.tensor(weightss, dtype=torch.float32).cuda())

        criterion = nn.CrossEntropyLoss()

        model_opt = torch.optim.Adam

        model = CA(input_log_events=input_log_events,

                   output_size=output_size,

                   d_model=d_model,

                   number_layers=number_layers,

                   number_heads=number_heads,

                   dropout=dropout,

                   max_len=max_len_of_input,

                   n_clusters=n_clusters,

                   beta=beta,

                   lambda_=lambda_,

                   optimizer=model_opt,

                   device=device,

                   learning_rate=learning_rate,

                   adam_betas_b1=adam_betas_b1,

                   adam_betas_b2=adam_betas_b2,

                   weight_decay=l2_regularization,

                   criterion=criterion,

                   initial_weights_embeddings=0,

                   )

        # model.init_clusters_train(train_dataloader, epoch=1)

        # model.fit(train_dataloader, epochs=2)

        file_name = "../results/output_files_per_experiment/CLog_time_" + TIME_INTERVAL_ + "_clusters_" + str(n_clusters) + "_.txt"

        print("$$$$$$"*10)

        print("------"*10)

        print("1) WARM UP STARTED!!!")

        print("------"*10)

        model = warm_up_MSP(model, train_data_loader_for_training, warmup_epochs)  # change the traindataloader to correspond to mask sentance prediction

        print("------"*10)

        print("1.1) Validate MSP training")

        msp_score_estimates = evaluate_MSP(model, train_data_loader_val)

        performance_scores_macro(msp_score_estimates, train_labels, file_name, top_k=3)

        performance_scores_micro(msp_score_estimates, train_labels,  file_name, top_k=3)

        top_k_res = []

        for tk in [1, 2, 3, 5, 20, 50, 100]:

            print("------------")

            tttt = performance_scores_topk(msp_score_estimates, train_labels,  file_name, top_k=tk)

            top_k_res.append(tttt)

        print("FINISH WARMUP")

        print("$$$$$$"*10)

        print("$$$$$$"*10)

        print("------"*10)

        print("2) START TRAINING CLUSTERING!!!")

        print("------"*10)

        print("2.1) Initialize the clusters")

        cluster_train_normal_preds, embeddings_normal_train, msp_pred_normal_train, model = solver(model, train_data_loader_for_training, train_data_loader_val, clustering_epochs)

        train_data_main, prediction_keys = augument_data_predictions_per_epoch(train_data_main, cluster_train_normal_preds)

        print("$$$$$$"*10)

        print("------"*10)

        print("3) Cluster sequences")

        clustered_sequences = cluster_sequences(train_data_main, prediction_keys)

        clustered_sequences.to_csv("./clustering_holistic_outptu_results_"+TIME_INTERVAL_+"_clusters_" + str(n_clusters) +  "_.csv")

        print("4) Finish")

        print("*****************" * 10)

        print("*****************" * 10)

        print("*****************" * 10)

    ## Uniform distrubiton of the events in the clusters is important. It reducecs the overhead for the operators too analyzie too many sequences.

    def create_tgt(sample):

        if sample.api_error1 != "['NO_FAILURE']" and sample.roun == 1:

            return 1

        else:

            return 0

