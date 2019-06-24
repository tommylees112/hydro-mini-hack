import pandas as pd
from sklearn import preprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats


data_dir = Path('/Users/tommylees/Downloads')

# -------------------------------------------------------
# PREPROCESS INDEX DATA
# -------------------------------------------------------
indices = pd.read_csv(data_dir / '6_indices.csv')
# sort by date and set date as index
indices = (
    indices
    .sort_values('date')
    .set_index('date')
    .drop(columns=['year','month'])
)
indices.index = pd.to_datetime(indices.index)

# convert from LONG to WIDE format
indices = indices.pivot_table(
    values='value', index=indices.index, columns='index'
)
# select the same starting date as the flow data
indices = indices.loc['1950':]

fig, ax = plt.subplots()
indices.TPI.plot(ax=ax)

# resample
indices = indices.resample('Q-DEC').mean()

# --------------------------------------------------
# PREPROCESS FLOW DATA
# --------------------------------------------------
flow = pd.read_csv(data_dir / '39001_gdf.csv')
header_info = flow.iloc[:20]
# drop the header information
flow = (
    flow
    .iloc[19:]
    .drop(columns='2019-06-24T13:17:11')
    .rename(columns={'file':'date', 'timestamp':'flow'})
    .set_index('date')
)
flow.index = pd.to_datetime(flow.index)
flow = flow.loc['1950':]
flow = flow.astype(float)

# get the 95th percentile
flow_q95 = flow.resample('Q-DEC').quantile(q=0.95)

# normalise the flow data by season
out = []
for season in np.unique(flow_q95.index.month.values):
    mean = flow_q95.loc[flow_q95.index.month == season].mean()
    std = flow_q95.loc[flow_q95.index.month == season].std()
    value = (flow_q95.loc[flow_q95.index.month == season] - mean) / std

    out.append(value)

flow_norm = pd.concat(out)

# # MIN MAX Rescaling
# f = flow.flow
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# flow['flow_norm'] = flow_scaled

# --------------------------------------------------
# Plot correlations
# --------------------------------------------------
df = flow_norm.join(indices)
fig, ax = plt.subplots()

a = sns.scatterplot(
    x="NAO", y="flow", data=df, ax=ax
)


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


fig, axs = plt.subplots(
    len(indices.columns), 4, figsize=(10,20),
    sharey=True
)
# a.annotate(stats.pearsonr, (0, 0))
season_lookup = {
    3: 'MAM',
    6: 'JJA',
    9: 'SON',
    12: 'DJF'
}

color_lookup = {
    3: sns.color_palette()[0],
    6: sns.color_palette()[1],
    9: sns.color_palette()[2],
    12: sns.color_palette()[3],
}

r_dict = {
    3: [],
    6: [],
    9: [],
    12: [],
}

def _pearson_r(x,y):
    index = (x.isnull() | y.isnull())
    x, y = x[index], y[index]
    return stats.pearsonr(x, y)[0]

for i, season in enumerate(np.unique(df.index.month.values)):
    for j, index in enumerate(df.iloc[:,2:].columns):
        s_data = df.loc[df.index.month == season]
        ax = axs[j, i]
        sns.regplot(
            x=index, y="flow", data=s_data, ax=ax,
            color=color_lookup[season]
        )
        # r2(s_data[index], s_data.flow)
        try:
            r_value = _pearson_r(s_data[index], s_data.flow)
        except:
            r_value = np.nan
        r = f"{r_value:.2f}"
        r_dict[season].append(r_value)

        if j == 0:
            ax.set_title(f'{season_lookup[season]} - {r}')
        else:
            ax.set_title(f'{r}')
        if i ==0:
            ax.set_ylabel(f'{index.upper()}')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticklabels('')
        ax.set_yticklabels('')
fig.suptitle('Indices Correlations with Mean Seasonal Q95 Thames Flow')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('index_thames_correlation_lin_plot.png')

r_df = pd.DataFrame(r_dict)


















#
