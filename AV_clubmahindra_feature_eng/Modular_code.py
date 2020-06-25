import pandas as pd
from functools import reduce

from fe_modules import date_col_derivation, date_diff, agg_col, groupby_time_delta


if __name__ == "__main__":

    train_df = pd.read_csv("train_data/train.csv")
    test_df = pd.read_csv("test_data/test.csv")
    print(train_df.shape, test_df.shape)

    date_cols = ["booking_date", "checkin_date", "checkout_date"]
    train_df = date_col_derivation(df=train_df, col_list=date_cols, date_format="%d/%m/%y")
    test_df = date_col_derivation(df=test_df, col_list=date_cols, date_format="%d/%m/%y")

    train_df["days_stay"] = date_diff(train_df, "checkout_date", "checkin_date")
    test_df["days_stay"] = date_diff(test_df, "checkout_date", "checkin_date")
    train_df["days_advance_booking"] = date_diff(train_df, "checkin_date", "booking_date")
    test_df["days_advance_booking"] = date_diff(test_df, "checkin_date", "booking_date")

    train_df = train_df.drop("amount_spent_per_room_night_scaled", axis=1)

    all_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    all_df = all_df.sort_values(by="checkin_date").reset_index(drop=True)
    print(all_df.shape)

    additional_cols = ["memberid", "resort_id", "state_code_residence", "checkin_date", "booking_date"]
    new_df = all_df[["reservation_id"] + additional_cols]

    ### Aggregate columns by mean
    cols_mean = ["booking_date_in_seconds", "checkin_date_in_seconds", "days_stay", "days_advance_booking",
                 "roomnights"]
    df_mean = agg_col(df=all_df, groupy_col="memberid", agg_col_list=cols_mean, aggrigate_method="mean")

    ### Aggregate columns by sum
    cols_sum = ["days_stay", "roomnights"]
    df_sum = agg_col(df=all_df, groupy_col="memberid", agg_col_list=cols_sum, aggrigate_method="sum")

    ### Aggregate columns by count of unique values
    cols_nunique = ["resort_id"]
    df_nunique = agg_col(df=all_df, groupy_col="memberid", agg_col_list=cols_nunique, aggrigate_method="nunique")

    # Merge all dfs
    dfs_list = [new_df, df_mean, df_sum, df_nunique]

    new_df = reduce(lambda left, right: pd.merge(left, right, on='memberid', how="left"), dfs_list)

    test_df = all_df
    # Create a list of columns and number of perviou/next events to look for each event.

    time_shift_cols = [["booking_date_in_seconds", 1],
                       ["booking_date_in_seconds", -1],
                       ["checkin_date_in_seconds", 1],
                       ["checkin_date_in_seconds", -1],
                       ["checkout_date_in_seconds", 1],
                       ["checkout_date_in_seconds", -1],
                       ["days_stay", 1],
                       ["days_stay", -1],
                       ["roomnights", 1],
                       ["roomnights", -1],
                       ["days_advance_booking", 1],
                       ["days_advance_booking", -1]
                       ]
    group_by_col = "memberid"

    new_shift_cols = []
    new_delta_cols = []

    # Iterate among each column, shift and extract time deltas
    for col, shift in time_shift_cols:
        all_df, shfit_col, delta_col = groupby_time_delta(all_df, groupby_col=group_by_col, time_col=col,
                                                          shift_by=shift)
        new_shift_cols.append(shfit_col)
        new_delta_cols.append(delta_col)

    # Join all new columns to the new_df
    new_df = pd.merge(new_df, all_df[["reservation_id"] + new_delta_cols], on="reservation_id")

    ### Info on prev and next visits (no change from old code)
    for col in ["channel_code", "room_type_booked_code", "resort_type_code", "main_product_code"]:
        all_df["prev_" + col] = all_df.groupby("memberid")[col].shift(1)
        new_df["prev_diff_" + col] = (all_df[col] == all_df["prev_" + col]).astype(int)

        all_df["next_" + col] = all_df.groupby("memberid")[col].shift(-1)
        new_df["next_diff_" + col] = (all_df[col] == all_df["next_" + col]).astype(int)

    ### pivot on member and resort ( no change from old code)
    gdf = pd.pivot_table(all_df, index="memberid", columns="resort_id", values="reservation_id", aggfunc="count",
                         fill_value=0).reset_index()
    new_df = pd.merge(new_df, gdf, on="memberid", how="left")

    gdf = pd.pivot_table(all_df, index="memberid", columns="checkin_date_year", values="reservation_id",
                         aggfunc="count", fill_value=0).reset_index()
    new_df = pd.merge(new_df, gdf, on="memberid", how="left")

    gdf = pd.pivot_table(all_df, index="memberid", columns="resort_type_code", values="reservation_id", aggfunc="count",
                         fill_value=0).reset_index()
    new_df = pd.merge(new_df, gdf, on="memberid", how="left")

    gdf = pd.pivot_table(all_df, index="memberid", columns="room_type_booked_code", values="reservation_id",
                         aggfunc="count", fill_value=0).reset_index()
    new_df = pd.merge(new_df, gdf, on="memberid", how="left")
