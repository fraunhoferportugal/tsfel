import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets.empirical1000_dataset import Empirical1000Dataset

import tsfel

DOMAINS = [
    "statistical",
    "temporal",
    "spectral",
]  # It specifies the domains included in the analysis.

emp_dataset = Empirical1000Dataset()

#########################################################################################################
#                  Experiment 1 - Overall executing time benchmarking per domain                        #
#########################################################################################################
execution_times_global = pd.DataFrame(
    {
        "Domain": pd.Series(dtype="str"),
        "Execution Time (s)": pd.Series(dtype="int"),
        "Normalized Execution Time (s/feature)": pd.Series(dtype="float32"),
    },
)

for domain in DOMAINS:
    n_features_domain = len(tsfel.get_features_by_domain(domain)[domain])

    init_time = time.perf_counter()
    for _id, ts in emp_dataset.raw.items():
        _ = tsfel.time_series_features_extractor(
            tsfel.get_features_by_domain(domain),
            ts.sig,
            fs=100,
            verbose=False,
        )
    end_time = time.perf_counter()
    total_time = end_time - init_time

    execution_times_global = pd.concat(
        [
            execution_times_global,
            pd.DataFrame(
                {
                    "Domain": [domain],
                    "Execution Time (s)": [int(total_time)],
                    "Normalized Execution Time (s/feature)": [
                        total_time / n_features_domain,
                    ],
                },
            ),
        ],
        ignore_index=True,
    )

#########################################################################################################
# Experiment 2 - Detailed time benchmarking as function of time series length grouped by feature domain #
#########################################################################################################
execution_times_individual = pd.DataFrame(
    {
        "Domain": pd.Series(dtype="str"),
        "ID": pd.Series(dtype="int"),
        "Execution Time (s)": pd.Series(dtype="float32"),
        "Length": pd.Series(dtype="int"),
    },
)

for domain in ["statistical", "temporal", "spectral"]:
    for id, ts in emp_dataset.raw.items():
        init_time = time.perf_counter()
        _ = tsfel.time_series_features_extractor(
            tsfel.get_features_by_domain(domain),
            ts.sig,
            fs=100,
            verbose=False,
        )
        end_time = time.perf_counter()
        total_time = end_time - init_time

        execution_times_individual = pd.concat(
            [
                execution_times_individual,
                pd.DataFrame(
                    {
                        "Domain": [domain],
                        "ID": [id],
                        "Execution Time (s)": [total_time],
                        "Length": [ts.len],
                    },
                ),
            ],
            ignore_index=True,
        )

# Some lengths have a low number of time series for statistical analysis (< 25 samples). We only consider lengths that
# have at least 25 samples for the analysis. For each length, we calculate the average and standard deviation and plot
# them on normal and log-log scales.
length_counts = execution_times_individual["Length"].value_counts()
valid_lengths = length_counts[length_counts >= 25].index
filtered_df = execution_times_individual[execution_times_individual["Length"].isin(valid_lengths)]
grouped_stats = filtered_df.groupby(["Length", "Domain"])["Execution Time (s)"].agg(["mean", "std"]).reset_index()


fig, (ax0, ax1) = plt.subplots(ncols=2)
for domain, marker in zip(DOMAINS, ["o", "v", "^"]):
    _data_slice = grouped_stats[grouped_stats["Domain"] == domain]
    ax0.errorbar(
        _data_slice["Length"],
        _data_slice["mean"],
        yerr=_data_slice["std"],
        label=domain,
        marker=marker,
        capsize=4,
    )
    ax1.errorbar(
        _data_slice["Length"],
        _data_slice["mean"],
        yerr=_data_slice["std"],
        label=domain,
        marker=marker,
        capsize=4,
    )

[ax.set(xlabel="Length (#)", ylabel="Execution Time (s)") for ax in [ax0, ax1]]
[ax.spines[loc].set_visible(False) for ax in [ax0, ax1] for loc in ["top", "right"]]
[ax.legend() for ax in [ax0, ax1]]
ax1.set_xscale("log")
ax1.set_yscale("log")

#########################################################################################################
#                    Experiment 4 - Measure the overall execution time per feature                      #
#########################################################################################################
execution_times_feature = pd.DataFrame(columns=["Domain", "Feature_Name", "Execution_Time"])
cfg = tsfel.get_features_by_domain()
for domain in DOMAINS:
    for feature in cfg[domain]:
        print(domain, feature)
        init_time = time.perf_counter()
        for _, ts in emp_dataset.raw.items():
            _ = tsfel.time_series_features_extractor(
                {domain: {feature: cfg[domain][feature]}},
                ts.sig,
                fs=100,
                verbose=False,
            )
        execution_time = time.perf_counter() - init_time

        execution_times_feature.loc[len(execution_times_feature)] = {
            "Domain": domain,
            "Feature_Name": feature,
            "Execution_Time": execution_time,
        }

fig, ax = plt.subplots(1, 1)
sns.barplot(execution_times_feature, x="Feature_Name", y="Execution_Time", hue="Domain", ax=ax)
ax.tick_params(axis="x", rotation=90)
plt.show(block=False)
