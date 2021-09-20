import time
import numpy as np
import pandas as pd
from libreco.data import split_by_ratio_chrono, DatasetPure, TransformedSet
from libreco.algorithms import (
    SVD, SVDpp, NCF, ALS, UserCF, ItemCF, RNN4Rec, KnnEmbedding,
    KnnEmbeddingApproximate, BPR
)
from matplotlib import pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    plot_precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
    auc
)

# remove unnecessary tensorflow logging
import os
import tensorflow as tf
from libreco.evaluation import metrics as met
from libreco.evaluation.evaluate import sample_user
from libreco.evaluation import evaluate as eval
from libreco.evaluation import computation as comp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def my_evaluate(model, data, eval_batch_size=8192, metrics=None, k=10,
             sample_user_num=2048, neg_sample=False, update_features=False,
             **kwargs):
    seed = kwargs.get("seed", 42)
    ax = kwargs.get("ax")
    if isinstance(data, pd.DataFrame):
        data = comp.build_transformed_data(
            model, data, neg_sample, update_features, seed
        )
    assert isinstance(data, TransformedSet), (
        "The data from evaluation must be TransformedSet object."
    )
    if not metrics:
        metrics = ["loss"]
    metrics = model._check_metrics(metrics, k)
    eval_result = dict()


    if met.POINTWISE_METRICS.intersection(metrics):
        y_prob, y_true = comp.compute_probs(model, data, eval_batch_size,)
    if met.LISTWISE_METRICS.intersection(metrics):
        chosen_users = sample_user(data, seed, sample_user_num)
        y_reco_list, users = comp.compute_recommends(model, chosen_users, k)
        y_true_list = data.user_consumed

    for m in metrics:
        if m in ["log_loss", "loss"]:
            eval_result[m] = log_loss(y_true, y_prob, eps=1e-7)
        elif m == "balanced_accuracy":
            y_pred = np.round(y_prob)
            eval_result[m] = balanced_accuracy_score(y_true, y_pred)
        elif m == "roc_auc":
            eval_result[m] = roc_auc_score(y_true, y_prob)
        elif m == "pr_auc":
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            average_precision = average_precision_score(y_true, y_prob)
            viz = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision, estimator_name=model.__class__.__name__, pos_label=None)
            viz.plot(ax=ax)
            eval_result[m] = auc(recall, precision)
        elif m == "precision":
            eval_result[m + "@" +str(k)] = met.precision_at_k(y_true_list, y_reco_list, users, k)
        elif m == "recall":
            eval_result[m + "@" +str(k)] = met.recall_at_k(y_true_list, y_reco_list,users, k)
        elif m == "map":
            eval_result[m + "@" +str(k)] = met.map_at_k(y_true_list, y_reco_list, users, k)
        elif m == "ndcg":
            eval_result[m + "@" +str(k)] = met.ndcg_at_k(y_true_list, y_reco_list, users, k)
    return eval_result


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)

def evaluate(model_name, dataset_name, model, eval_data, metrics):
    t = {"model": model_name, "dataset": dataset_name}
    t.update(my_evaluate(model, eval_data, metrics=metrics, k=1, neg_sample=False))
    t.update(my_evaluate(model, eval_data, metrics=metrics, k=3, neg_sample=False))
    t.update(my_evaluate(model, eval_data, metrics=metrics, k=5, neg_sample=False))
    t.update(my_evaluate(model, eval_data, metrics=metrics, k=10, neg_sample=False))
    return t


if __name__ == "__main__":
    start_time = time.perf_counter()
    fig, ax = plt.subplots()
    dtypes = {'user': 'string', 'item': 'string', 'time': 'int64'}

    #dataset = "yoochoose_dense"
    #data = pd.read_csv("./data/curated/yoochoose_dense.csv", header=None, names=["user", "item", "time"], dtype=dtypes, parse_dates=["time"]) #columns: user, item, time, label

    #dataset = "yoochoose"
    #data = pd.read_csv("./data/curated/yoochoose.csv", header=None, names=["user", "item", "time"], dtype=dtypes, parse_dates=["time"]) #columns: user, item, time, label

    #dataset = "ratailrocket_dense"
    #data = pd.read_csv("./data/curated/ratailrocket_dense.csv", header=None, names=["user", "item", "time"], dtype=dtypes, parse_dates=["time"]) #columns: user, item, time, label

    #dataset = "b2b_dense"
    #data = pd.read_csv("./data/curated/b2b_dense.csv", header=None, names=["user", "item", "time"], dtype=dtypes, parse_dates=["time"]) #columns: user, item, time, label

    #dataset = "b2b"
    #data = pd.read_csv("./data/curated/b2b.csv", header=None, names=["user", "item", "time"], dtype=dtypes, parse_dates=["time"]) #columns: user, item, time, label

    #dataset = "b2c_dense"
    #data = pd.read_csv("./data/curated/b2c_dense.csv", header=None, names=["user", "item", "time"], dtype=dtypes, parse_dates=["time"]) #columns: user, item, time, label

    dataset = "b2c"
    data = pd.read_csv("./data/curated/b2c.csv", header=None, names=["user", "item", "time"], dtype=dtypes, parse_dates=["time"]) #columns: user, item, time, label

    data['label'] = 2
    metrics = ["loss", "balanced_accuracy", "roc_auc", "pr_auc",
               "precision", "recall", "map", "ndcg"]
    metrics = ["precision", "recall", "map", "ndcg"]

    print(data.head())

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)

    # do negative sampling, assume the data only contains positive feedback
    train_data.build_negative_samples(data_info, item_gen_mode="random", num_neg=1, seed=2020)
    eval_data.build_negative_samples(data_info, item_gen_mode="random", num_neg=1, seed=2222)

    evals = []
    reset_state("SVD")
    svd = SVD("ranking", data_info, embed_size=16, n_epochs=15, lr=0.001, reg=None, batch_size=256, batch_sampling=False, num_neg=1)
    svd.fit(train_data, eval_data=eval_data,  metrics=metrics, shuffle=True, dataset_name=dataset, verbose=2)
    evals = evals + svd.evals

    print("prediction: ", svd.predict(user=1, item=2333))
    print("recommendation: ", svd.recommend_user(user=1, n_rec=7))

    reset_state("SVD++")
    svdpp = SVDpp(task="ranking", data_info=data_info, embed_size=16, n_epochs=15, lr=0.001, reg=None, batch_size=256)
    svdpp.fit(train_data, eval_data=eval_data,  metrics=metrics, dataset_name=dataset, verbose=2)
    evals = evals + svdpp.evals

    print("prediction: ", svdpp.predict(user=1, item=2333))
    print("recommendation: ", svdpp.recommend_user(user=1, n_rec=7))

    reset_state("NCF")
    ncf = NCF("ranking", data_info, embed_size=16, n_epochs=15, lr=0.001, lr_decay=False, reg=None, batch_size=256, num_neg=1, use_bn=True,
              dropout_rate=None, hidden_units="128,64,32", tf_sess_config=None)
    ncf.fit(train_data, eval_data=eval_data,  metrics=metrics, shuffle=True, dataset_name=dataset, verbose=2)
    evals = evals + ncf.evals

    print("prediction: ", ncf.predict(user=1, item=2333))
    print("recommendation: ", ncf.recommend_user(user=1, n_rec=7))

    reset_state("ALS")
    als = ALS(task="ranking", data_info=data_info, embed_size=16, n_epochs=15, reg=5.0, alpha=10, seed=42)
    als.fit(train_data, eval_data=eval_data,  metrics=metrics, dataset_name=dataset, verbose=2, use_cg=True, n_threads=8)
    evals = evals + als.evals

    print("prediction: ", als.predict(user=1, item=2333))
    print("recommendation: ", als.recommend_user(user=1, n_rec=7))

    reset_state("BPR")
    bpr = BPR("ranking", data_info, embed_size=16, n_epochs=15, lr=3e-4, reg=None, batch_size=256, num_neg=1, use_tf=True)
    bpr.fit(train_data, eval_data=eval_data,  metrics=metrics, dataset_name=dataset, verbose=2, num_threads=16, optimizer="adam")
    evals = evals + bpr.evals

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
    pd.DataFrame(evals).to_csv("./data/consumer/evaluation-" + dataset + ".csv", index=False)
    plt.show()
