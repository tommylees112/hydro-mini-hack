import pandas as pd
from sklearn import preprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import xarray as xr

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

# get the high/low percentile
flow_q95 = flow.resample('Q-DEC').quantile(q=0.95)
flow_q05 = flow.resample('Q-DEC').quantile(q=0.05)


def normalise_data_by_month(data) -> pd.DataFrame:
    out = []
    for season in np.unique(data.index.month.values):
        mean = data.loc[data.index.month == season].mean()
        std = data.loc[data.index.month == season].std()
        value = (data.loc[data.index.month == season] - mean) / std

        out.append(value)
    return pd.concat(out)


# normalise the flow data by season
out = []
for i, data in enumerate([flow_q05, flow_q95]):
    df = normalise_data_by_month(data)
    if i == 0:
        df = df.rename(columns={'flow':'q05'})
    else:
        df = df.rename(columns={'flow':'q95'})
    out.append(df)

flow_hilo = out[0].join(out[1])


# --------------------------------------------------
# Plot correlations
# --------------------------------------------------
df = flow_hilo.join(indices)
fig, ax = plt.subplots()

a = sns.scatterplot(
    x="NAO", y="flow", data=df, ax=ax
)

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

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
    null_index = (x.isnull() | y.isnull())
    x, y = x[~null_index], y[~null_index]
    return stats.pearsonr(x, y)[0]


for hilo_flow in ['q05', 'q95']:

    fig, axs = plt.subplots(
        len(indices.columns), 4, figsize=(10,20),
        sharey=True
    )

    for i, season in enumerate(np.unique(df.index.month.values)):
        for j, index in enumerate(df.iloc[:,3:].columns):
            s_data = df.loc[df.index.month == season]
            ax = axs[j, i]
            r_value = _pearson_r(s_data[index], s_data[hilo_flow])
            color = sns.color_palette()[0] if abs(r_value) > 0.1 else "#95a5a6"

            sns.regplot(
                x=index, y=hilo_flow, data=s_data, ax=ax,
                color=color
            )

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
    fig.suptitle(f'Indices Correlations with Mean Seasonal {hilo_flow} Thames Flow')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'index_thames_correlation_{hilo_flow}.png')



r_df = pd.DataFrame(r_dict)

# ideas:
# look at high vs. low flow
# look at lagged predictability (shift by one season) vs. concurrency

# -----------------------------------------------------------------------------
# plot a hydrograph of the thames
# http://pyhyd.blogspot.com/2017/11/matplotlib-template-for-precipitation.html
# -----------------------------------------------------------------------------

# convert to xarray
ds = flow.to_xarray()
f_med = (
    ds.groupby('date.month').median().rename({'flow': 'flow_median'}).to_dataframe()
)
f_95 = (
    ds
    .groupby('date.month')
    .reduce(np.nanpercentile, dim='date', q=95)
    .rename({'flow': 'flow_q95'})
    .to_dataframe()
)
f_05 = (
    ds
    .groupby('date.month')
    .reduce(np.nanpercentile, dim='date', q=5)
    .rename({'flow': 'flow_q05'})
    .to_dataframe()
)
climatology = f_med.join(f_95).join(f_05)

fig, ax = plt.subplots()
climatology.flow_median.plot(ax=ax)
plt.fill_between(climatology.index.values, climatology.flow_q05, climatology.flow_q95, alpha=0.3)
ax.set_ylabel('Flow [m-3 s-1]')
ax.set_title('Annual Seasonality of Thames River Flow')
fig.savefig('thames_flow_climatology.png')

#
