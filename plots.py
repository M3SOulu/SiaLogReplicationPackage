def plot_noisy_input(eval_result, file_path):
    import pandas as pd
    import seaborn as sns
    data = {classifier: [v["F1"] for _, v in results["Noisy input"].items()] for classifier, results in
            eval_result.items()}
    ratios = [float(k.replace("Ratio ", "")) for k in list(eval_result.values())[0]["Noisy input"]]
    ratios.insert(0, 0.0)
    for k, v in data.items():
        v.insert(0, eval_result[k]["Test"]["F1"])
    df = pd.DataFrame(data)
    df.index = ratios
    fig = sns.lineplot(data=df, palette="tab10", linewidth=2).get_figure()
    fig.savefig(file_path)
    print("Noisy input plot saved to:", file_path)


def plot_distribution_drift(drift_result, file_path, csv_output_file=None):
    import pandas as pd
    import seaborn as sns
    import csv
    data = {"Fitness Score": list(drift_result.values()),
            "Noise Ratio": list(drift_result.keys())}
    if csv_output_file is not None:
        with open(csv_output_file, "w") as fp:
            writer = csv.writer(fp)
            for ratio, score in zip(data["Noise Ratio"], data["Fitness Score"]):
                writer.writerow([ratio, score])

    df = pd.DataFrame(data)
    fig = sns.lineplot(x="Noise Ratio", y="Fitness Score", data=df, palette="tab10", linewidth=2).get_figure()
    fig.savefig(file_path)
    print("Noisy input plot saved to:", file_path)
