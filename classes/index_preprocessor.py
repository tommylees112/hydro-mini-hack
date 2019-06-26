import pandas as pd
from pathlib import Path

class IndexPreprocessor:
    # data_dir = Path('/Users/tommylees/Downloads')
    # raw_data = pd.read_csv(data_dir / '6_indices.csv')
    # shape = raw_data.shape

    def __init__(self, data_dir: Path) -> None:
        self.raw_data = pd.read_csv(data_dir / '6_indices.csv')
        self.shape = self.raw_data.shape

    def __len__(self):
        return self.raw_data.size

    def __repr__(self):
        return f"Tommy and Timo are Great:\n{self.raw_data.columns}"

    @staticmethod
    def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
        # convert from LONG to WIDE format
        return df.pivot_table(
            values='value', index=df.index, columns='index'
        )

    @staticmethod
    def resample_seasons(df: pd.DataFrame) -> pd.DataFrame:
        return df.resample('Q-DEC').mean()

    def preprocess(self) -> None:
        df = (
            self.raw_data
            .sort_values('date')
            .set_index('date')
            .drop(columns=['year','month'])
        )
        df.index = pd.to_datetime(df.index)

        df = self.long_to_wide(df)

        # select the same starting date as the flow data
        df = df.loc['1950':]

        # resample
        df = self.resample_seasons(df)

        self.data = df
