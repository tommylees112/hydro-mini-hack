# run.py
from pathlib import Path
from classes import IndexPreprocessor
# from classes import IndexPreprocessor, FlowPreprocessor
# from classes.flow_preprocessor import FlowPreprocessor
from calc_anomalies import normalise_data_by_month

data_dir = Path('/Users/tommylees/Downloads')
i = IndexPreprocessor(data_dir)
i.preprocess()

print(i.data.head())
