import datetime as dt
import pandas as pd


def date_col_derivation(df, col_list, date_format="%d/%m/%y"):
    """
    Extract date features from data columns
    :param df:
    :param col_list:
    :param date_format:
    :return:
    """
    for col in col_list:
        df[col] = pd.to_datetime(df[col], format=date_format)
        df[col + "_in_seconds"] = (df[col] - dt.datetime(1970, 1, 1)).dt.total_seconds()
        df[col + "_month"] = df[col].dt.month
        df[col + "_year"] = df[col].dt.year
        df[col + "_week"] = df[col].dt.week
    return df


def date_diff(df, date_col1, date_col2, diff_format="days"):
    """
    Return date difference in expected format
    :param df:
    :param date_col1:
    :param date_col2:
    :param diff_format:
    :return:
    """
    return getattr((df[date_col1] - df[date_col2]).dt, diff_format)


def agg_col(df, groupy_col, agg_col_list, aggrigate_method):
    """
    Function to group , aggregate multiple columms by any(pandas supported) method
    :param df:
    :param groupy_col:
    :param agg_col_list:
    :param aggrigate_method:
    :return:
    """
    new_df = None
    for agg_col in agg_col_list:

        gdf = getattr(df.groupby(groupy_col)[agg_col], aggrigate_method)().reset_index()
        gdf.columns = [groupy_col, "member_" + agg_col + "_" + aggrigate_method]
        if new_df is None:
            new_df = gdf
        else:
            new_df = pd.merge(new_df, gdf, on=groupy_col, how="left")

    return new_df


def groupby_time_delta(df, groupby_col, time_col, shift_by):
    """
    Extract time delta from a current event to previous events/next events
    :param df:
    :param groupby_col:
    :param time_col:
    :param shift_by:
    :return:
    """
    if shift_by > 0:
        shift_type = "next"
    else:
        shift_type = "prev"

    shift_col_name = shift_type + "_" + time_col
    new_delta_col = "time_gap_" + time_col + "_" + shift_type + "_" + str(abs(shift_by))

    df[shift_col_name] = df.groupby(groupby_col)[time_col].shift(shift_by)
    df[new_delta_col] = date_diff(df, time_col, shift_col_name, diff_format="days")

    return df, shift_col_name, new_delta_col
