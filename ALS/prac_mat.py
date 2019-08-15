import numpy
from scipy.sparse import csr_matrix,lil_matrix
import numpy.matlib
from scipy import sparse
import pandas as pd
df = pd.read_excel("ratings_train.xlsx", header=None)
t = df.as_matrix()
print(df[0,0])
