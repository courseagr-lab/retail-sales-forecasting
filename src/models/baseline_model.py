import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.data.data_cleaning import load_config, project_path


def time_based_split(df, config):
    pass