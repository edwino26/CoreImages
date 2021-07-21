#T2 VALIDATION
import pandas as pd


#Load Stack
UVStack = pd.read_excel('./ML_Results/T2_validation/ImgStack.xls')
UVStack.to_numpy()

istr = 0
iend = 42344


