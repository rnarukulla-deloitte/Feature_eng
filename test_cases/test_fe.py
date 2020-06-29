import unittest
import pandas as pd
import numpy as np
from fe_modules import date_col_derivation, date_diff, agg_col, groupby_time_delta
from pandas.util.testing import assert_frame_equal, assert_series_equal

class test_fe_modules(unittest.TestCase):

    def test_fdate_col_derivation(self):
        # Input df
        df = pd.DataFrame({'a': [1], 'b': ["05/06/20"]})

        # Expected output
        o_df = pd.DataFrame({'a': pd.Series([1], dtype='int64'),
                             'b': pd.Series(['2020-06-05T00:00:00.000000000'], dtype='datetime64[ns]'),
                             'b_in_seconds': pd.Series([1.591315e+09], dtype='int64'),
                             'b_month': pd.Series([6], dtype='int64'),
                             'b_year': pd.Series([2020], dtype='int64'),
                             'b_week': pd.Series([23], dtype='int64')
                             }
                            )
        assert_frame_equal(date_col_derivation(df=df, col_list=["b"], date_format="%d/%m/%y"), o_df, check_dtype=False)

    def test_date_diff(self):
        # Input df
        df = pd.DataFrame({'a': pd.Series(['2020-06-10T00:00:00.000000000'], dtype='datetime64[ns]'),
                           'b': pd.Series(['2020-06-5T00:00:00.000000000'], dtype='datetime64[ns]'),
                           }
                          )
        # expected Output
        o_df = pd.Series([5])
        assert_series_equal(date_diff(df, 'a', 'b', diff_format="days"), o_df, check_dtype=False)

    def test_agg_col_sum(self):
        # Input df
        df = pd.DataFrame({'id': pd.Series(['1', '1', '2'], dtype='object'),
                           'value_col1': pd.Series([2, 5, 8], dtype='int'),
                           'value_col2': pd.Series([1, 5, 8], dtype='int')
                           }
                          )
        o_df = pd.DataFrame({'id': pd.Series(['1', '2'], dtype='object'),
                             'member_value_col1_sum': pd.Series([7, 8], dtype='int'),
                             'member_value_col2_sum': pd.Series([6, 8], dtype='int')
                             }
                            )
        assert_frame_equal(
            agg_col(df, groupy_col='id', agg_col_list=['value_col1', 'value_col2'], aggrigate_method='sum'), o_df,
            check_dtype=False)

    def test_agg_col_min(self):
        # Input df
        df = pd.DataFrame({'id': pd.Series(['1', '1', '2'], dtype='object'),
                           'value_col1': pd.Series([2, 5, 8], dtype='int'),
                           'value_col2': pd.Series([1, 5, 8], dtype='int')
                           }
                          )
        o_df = pd.DataFrame({'id': pd.Series(['1', '2'], dtype='object'),
                             'member_value_col1_min': pd.Series([2, 8], dtype='int'),
                             'member_value_col2_min': pd.Series([1, 8], dtype='int')
                             }
                            )
        assert_frame_equal(
            agg_col(df, groupy_col='id', agg_col_list=['value_col1', 'value_col2'], aggrigate_method='min'), o_df,
            check_dtype=False)

    def test_groupby_time_delta_df(self):
        # Input df
        df = pd.DataFrame({'id': pd.Series(['1', '1', '2', '2'], dtype='object'),
                           'booking': pd.Series(['2020-04-03T00:00:00.000000000', '2020-06-10T00:00:00.000000000',
                                                 '2020-01-03T00:00:00.000000000', '2020-05-05T00:00:00.000000000'],
                                                dtype='datetime64[ns]')
                           }
                          )

        # Expected output df
        o_df = pd.DataFrame({'id': pd.Series(['1', '1', '2', '2'], dtype='object'),
                             'booking': pd.Series(['2020-04-03T00:00:00.000000000', '2020-06-10T00:00:00.000000000',
                                                   '2020-01-03T00:00:00.000000000', '2020-05-05T00:00:00.000000000'],
                                                  dtype='datetime64[ns]'),
                             'next_booking': pd.Series(
                                 ['NaT', '2020-04-03T00:00:00.000000000', 'NaT', '2020-01-03T00:00:00.000000000'],
                                 dtype='datetime64[ns]'),
                             'time_gap_booking_next_1': pd.Series([np.nan, 68.0, np.nan, 123.0], dtype='float64')
                             }
                            )
        assert_frame_equal(groupby_time_delta(df, groupby_col='id', time_col="booking", shift_by=1)[0], o_df,
                           check_dtype=False)

    def test_groupby_time_delta_colnames(self):
        # Input df
        df = pd.DataFrame({'id': pd.Series(['1', '1', '2', '2'], dtype='object'),
                           'booking': pd.Series(['2020-04-03T00:00:00.000000000', '2020-06-10T00:00:00.000000000',
                                                 '2020-01-03T00:00:00.000000000', '2020-05-05T00:00:00.000000000'],
                                                dtype='datetime64[ns]')
                           }
                          )

        # Expected output df
        o_df = pd.DataFrame({'id': pd.Series(['1', '1', '2', '2'], dtype='object'),
                             'booking': pd.Series(['2020-04-03T00:00:00.000000000', '2020-06-10T00:00:00.000000000',
                                                   '2020-01-03T00:00:00.000000000', '2020-05-05T00:00:00.000000000'],
                                                  dtype='datetime64[ns]'),
                             'next_booking': pd.Series(
                                 ['NaT', '2020-04-03T00:00:00.000000000', 'NaT', '2020-01-03T00:00:00.000000000'],
                                 dtype='datetime64[ns]'),
                             'time_gap_booking_next_1': pd.Series([np.nan, 68.0, np.nan, 123.0], dtype='float64')
                             }
                            )
        self.assertEqual(groupby_time_delta(df, groupby_col='id', time_col="booking", shift_by=1)[1:],
                         ('next_booking', 'time_gap_booking_next_1'))
