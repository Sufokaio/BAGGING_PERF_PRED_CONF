import argparse
import os
import pickle
import json

import numpy as np
from deepperf import DeepPerf
from rHINNPERF import HINNPerf

from KNN import KNN
from RT import RT
from KRR import KRR
from LR import LR
from RF import RF

from SVRWrapper import SVRWrapper as SVR
from baseline import Baseline

import time
import copy
import tensorflow as tf

import time
import json
import argparse
import numpy as np
from collections import defaultdict



import multiprocessing

import os

cpu_count = os.cpu_count()




MODEL_REGISTRY = {
    "baseline": Baseline,
    # "LR": LR,
    "SVR": SVR,
    # "KNN": KNN,
    # "RF": RF,
    "RT": RT,
    "KRR": KRR,
    # "DeepPerf": DeepPerf,
    "HINNPerf": HINNPerf


 }
 # Dataset name mapped to its corresponding sample sizes
DATASET_REGISTRY = {

    "apache": np.array( [9,18,27,36,45]),

    "bdbc": np.array([18,36,54,72,90]),
    "kanzi": np.array([31,62,93,124,155]),
    "x264": np.array([16,32,48,64,80]),


    "lrzip": np.array([127,295,386,485,907]),
    "dune": np.array([224,692,1000,1365,1612]),
    "hipacc": np.array([261,528,736,1281,2631]),
    "hsmgp": np.array([77,173,384,480,864]),

}

# Evaluate and rank top-10 models based on multiple metrics
# Assignment of Borda scores and ranks for ensemble construction
def eval_top10(metrics_results):
    for entry in metrics_results:
        mean_sa = np.mean(entry["Metrics"]["SA"])
        mean_sa_5 = np.mean(entry["Metrics"]["SA_5"])
        entry["Mean_SA"] = mean_sa
        entry["SA_vs_SA_5"] = f"SA: {mean_sa:.4f} | SA_5: {mean_sa_5:.4f}"

    metrics_results.sort(key=lambda x: x["Mean_SA"], reverse=True)

    for entry in metrics_results:
        entry["Mean_MAE"] = np.mean(entry["Metrics"]["MAE"])
        entry["Mean_MRE"] = np.mean(entry["Metrics"]["MRE"])
        entry["Mean_MBRE"] = np.mean(entry["Metrics"]["MBRE"])
        entry["Mean_MIBRE"] = np.mean(entry["Metrics"]["MIBRE"])

    def rank_entries(metric_name):
        sorted_entries = sorted(metrics_results, key=lambda x: x[f"Mean_{metric_name}"])
        for rank, entry in enumerate(sorted_entries, start=1):
            entry[f"Rank_{metric_name}"] = rank

    for metric in ["MAE", "MRE", "MBRE", "MIBRE"]:
        rank_entries(metric)

    for entry in metrics_results:
        entry["Borda_Score"] = (
            entry["Rank_MAE"] +
            entry["Rank_MRE"] +
            entry["Rank_MBRE"] +
            entry["Rank_MIBRE"]
        )

    sorted_by_borda = sorted(metrics_results, key=lambda x: (x["Borda_Score"], x["Mean_MAE"]))

    for rank, entry in enumerate(sorted_by_borda, start=1):
        entry["Borda_Rank"] = rank

    for i, entry in enumerate(metrics_results):
        print(
            f"Entry {i}: "
            f"MAE={entry['Mean_MAE']:.4f} (Rank {entry['Rank_MAE']}), "
            f"MRE={entry['Mean_MRE']:.4f} (Rank {entry['Rank_MRE']}), "
            f"MBRE={entry['Mean_MBRE']:.4f} (Rank {entry['Rank_MBRE']}), "
            f"MIBRE={entry['Mean_MIBRE']:.4f} (Rank {entry['Rank_MIBRE']}), "
            f"Borda Score={entry['Borda_Score']}, Borda Rank={entry['Borda_Rank']}"
        )
    return sorted_by_borda


# Metric computations for ensemble results
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mre = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100
    mbre = np.mean(np.abs(y_true - y_pred) / (np.minimum(y_true, y_pred) + 1e-8))
    mibre = np.mean(np.abs(y_true - y_pred) / (np.maximum(y_true, y_pred) + 1e-8))
    return {
        "MAE": mae,
        "MRE": mre,
        "MBRE": mbre,
        "MIBRE": mibre
    }

def log_time(message):
    with open("times.txt", "a") as f:
        f.write(message + "\n")



def aggregate_ensemble_metrics(ensemble_results):
    metrics = {"MAE": [], "MRE": [], "MBRE": [], "MIBRE": []}
    runs = []

    for run in ensemble_results:
        m = compute_metrics(run["y_test"], run["y_pred"])
        for k in metrics:
            metrics[k].append(m[k])
        runs.append({
            # "y_test": run["y_test"],
            # "y_pred": run["y_pred"],
            # "y_train": run.get("y_train", []),  # Include y_train
            # "X_train": run.get("X_train", []),  # Include X_train
            # "X_test": run.get("X_test", []),    # Include X_test
            # "feature_sets": run.get("feature_sets", [])  # Include feature sets
        })

    # Compute means and medians for each metric
    mean_metrics = {f"Mean_{k}": float(np.mean(metrics[k])) for k in metrics}
    median_metrics = {f"Median_{k}": float(np.median(metrics[k])) for k in metrics}

    # Return in the same format as single results
    return {
        "Metrics": metrics,
        **mean_metrics,
        **median_metrics,
        "Runs": runs
    }


def feature_frac_tag(frac):
    if frac is None:
        return "all"
    if frac == "sqrt":
        return "sqrt"
    return f"{int(frac * 100)}pct"

# Run experiment task for multiprocessing
def run_experiment_task(model_name, dataset, sample_index, s):
    print(f"Running model '{model_name}' on dataset '{dataset}' with sample size {sample_index}")
    evaluator = MODEL_REGISTRY[model_name](
        dataset,
        split_mode=False,
        split=s,
        num_runs=30
    )

    results_dir = f"results/{dataset}/{s}"
    os.makedirs(results_dir, exist_ok=True)

    metrics_results_path = os.path.join(
        results_dir,
        f"{model_name}_single_metrics.json"
    )

    # --------------------------------------------------
    # 1) RUN SINGLE-LEARNER GRID SEARCH (UNCHANGED)
    # --------------------------------------------------
    if os.path.exists(metrics_results_path):
        print(f"Loading single-model results from {metrics_results_path}")
        with open(metrics_results_path, "r") as f:
            metrics_results = json.load(f)
    else:
        t0 = time.time()
        metrics_results = evaluator.run_experiment()
        elapsed = time.time() - t0
        log_time(
            f"Completed SINGLE {model_name} on {dataset} sample {s} "
            f"in {elapsed:.2f} seconds"
        )

        # Convert numpy → list
        for entry in metrics_results:
            if "Metrics" in entry:
                for key, value in entry["Metrics"].items():
                    if isinstance(value, np.ndarray):
                        entry["Metrics"][key] = value.tolist()

        # if not model_name == "baseline":
        #     metrics_results = eval_top10(metrics_results)

        with open(metrics_results_path, "w") as f:
            json.dump(metrics_results, f, indent=2)
        if model_name == "baseline":
            return f"Completed baseline on {dataset} sample {s}"


    # --------------------------------------------------
    # 2) EXTRACT TOP-1 CONFIGURATION (IMPORTANT)
    # --------------------------------------------------
    top1_params = metrics_results[0]["Params"]
    print(f"Top-1 params for {model_name}: {top1_params}")

    # --------------------------------------------------
    # 3) DEFINE ENSEMBLE CONFIGURATIONS
    # --------------------------------------------------
    ensemble_configs = []

    N_LIST = [3,5,7,10]

    COMBINERS = ["mean", "median"]
    FEATURE_FRACS = [0.5, "sqrt"]

        # ---- Bagging (sample diversity) ----
    ensemble_configs.append({
        "strategy": "bagging",
        "n_estimators_list": N_LIST,
        "feature_fraction": None
    })

    # ---- Random Subspace & Random Patches ----
    for frac in FEATURE_FRACS:
        ensemble_configs.append({
            "strategy": "rs",
            "n_estimators_list": N_LIST,
            "feature_fraction": frac
        })
        ensemble_configs.append({
            "strategy": "rp",
            "n_estimators_list": N_LIST,
            "feature_fraction": frac
        })

    # --------------------------------------------------
    # 4) RUN ENSEMBLES
    # --------------------------------------------------
    for cfg in ensemble_configs:
        t0 = time.time()

        # Run the ensemble experiment (returns results for all n_estimators in list)
        ensemble_results = evaluator.run_ensemble_experiment(
            params=top1_params,
            ensemble_cfg=cfg
        )

        elapsed = time.time() - t0
        log_time(
            f"Completed ENSEMBLE {model_name} {cfg['strategy']} on {dataset} sample {s} "
            f"in {elapsed:.2f} seconds"
        )

        # Group results by n_estimators and combiner
        results_by_n = defaultdict(lambda: {"mean": [], "median": []})
        for res in ensemble_results:
            n = res["n_estimators"]
            combiner = res["combiner"]
            results_by_n[n][combiner].append(res)

        # For each n_estimators, save separate files
        for n_estimators, combiner_dict in results_by_n.items():
            tag_base = (
                f"{model_name}_"
                f"{cfg['strategy']}_"
                f"N{n_estimators}_"
                f"F{feature_frac_tag(cfg['feature_fraction'])}"
            )
            out_path_mean = os.path.join(results_dir, f"{tag_base}_mean.json")
            out_path_median = os.path.join(results_dir, f"{tag_base}_median.json")

            # Skip if both files exist
            if os.path.exists(out_path_mean) and os.path.exists(out_path_median):
                print(f"Skipping existing ensemble: {tag_base}")
                continue

            mean_runs = combiner_dict["mean"]
            median_runs = combiner_dict["median"]

            if mean_runs:
                mean_metrics = aggregate_ensemble_metrics(mean_runs)
                with open(out_path_mean, "w") as f:
                    json.dump(mean_metrics, f, indent=2)

            if median_runs:
                median_metrics = aggregate_ensemble_metrics(median_runs)
                with open(out_path_median, "w") as f:
                    json.dump(median_metrics, f, indent=2)



    return f"Completed {model_name} on {dataset} sample {s}"

def main():
    parser = argparse.ArgumentParser(description='Starting ...')
    parser.add_argument('--dataset', type=str, default='all', help='Dataset name')
    parser.add_argument('--model', type=str, default="all", help='Model name')
    parser.add_argument('--mode', type=str, default="normal", help='Execution mode')
    parser.add_argument('--num_runs', type=int, default=30 , help='Number of experiments')

    args = parser.parse_args()
    

    if args.dataset == "all":
        datasets = DATASET_REGISTRY
    else:
        datasets = [args.dataset]

    if args.model == "all":
        models = [m for m in MODEL_REGISTRY.keys() if m != "baseline"]
    else:
        models = [args.model]

    if args.mode == "normal":
        task_args = []
        for model_name in models:
            for dataset, sample_sizes in datasets.items():
                for i, s in enumerate(sample_sizes):
                    task_args.append((model_name, dataset, i, s))
        with multiprocessing.Pool(processes=cpu_count - 3) as pool:

            results = pool.starmap(run_experiment_task, task_args)
        end_time = time.time()
        for res in results:
            print(res)
        print(f"All experiments completed at {end_time}")


if __name__ == "__main__":
    main()
