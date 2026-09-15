import pandas as pd
import numpy as np

df = pd.read_csv('../model_inputs/physchem_properties.csv', sep = '\t', index_col = 0)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median())
df.to_csv('../model_inputs/physchem_properties_imputed.csv', sep = '\t')