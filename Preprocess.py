import pandas as pd

db = pd.read_csv(r"C:/Users/aaron/Desktop/Coding/FPGA AI Project/mass_case_description_train_set.csv", skip_blank_lines=True, chunksize=100)

for i, chunk in enumerate(db):
    chunk.to_csv('massTrain{}.csv'.format(i), index=False)
