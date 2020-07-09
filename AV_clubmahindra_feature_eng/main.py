import pandas as pd
from fe_eng_job import get_features


I_TRAIN_PATH = "train_data/train.csv"
I_TEST_PATH = "test_data/test.csv"
O_OUTFILE = "features_df.csv"

if __name__ == "__main__":

    features_df = get_features(I_TRAIN_PATH, I_TEST_PATH)

    features_df.to_csv(O_OUTFILE, index=False)