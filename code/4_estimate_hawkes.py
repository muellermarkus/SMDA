import pandas as pd
import os
import scipy.optimize as opt

# load simulated data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
df = pd.read_csv(data_path + '/simulations/hawkes.csv')
