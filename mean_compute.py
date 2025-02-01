import pandas as pd
import numpy as np

data = (0.255, 0.255, 0.256, 0.258, 0.257, 0.255, 0.256, 0.256, 0.255, 0.254)
df = pd.DataFrame(data)

# Compute mean for each column
column_means = df.mean()

# Print means
print("Mean value")
print(column_means)

