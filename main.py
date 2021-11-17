import os
import pandas as pd
import numpy as np
from glob import glob
from typing import Tuple
from scipy.stats import linregress
import plotly.graph_objs as go


def root_path(*args, **kwargs) -> str:
    return 'L:/bucket/hypercapnia'


def get_files(*, path: str, **kwargs) -> list:
    files = []
    for file in glob(os.path.join(path, '0838_*hypercapnia.txt')):
        files.append(file)
    return files


def make_df(*, files: list, **kwargs) -> pd.DataFrame:
    lst = []
    for file in files:
        df = pd.read_csv(file, sep='\t', index_col=None, header=0, skiprows=range(1, 3))

        # get subject id
        subject_id = "_".join(os.path.basename(file).split("_", 2)[:2])
        df['subject_id'] = subject_id
        lst.append(df)
    df = pd.concat(lst, axis=0, ignore_index=True)
    return df


def set_dtypes(*, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # set column types
    df['date'] = pd.to_datetime(df['date'])
    return df


def hack_for_subjects_with_id_001(*, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    there are currently 2 subjects with id 001. Until this is rectified, need to do a workaround

    :param df:
    :param kwargs:
    :return:
    """

    df.loc[df['date'] == df['date'].min(), 'subject_id'] = "0838_001_a"
    return df


def drop_columns(*, df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(' .4', axis=1)


def rename_columns(*, df: pd.DataFrame) -> pd.DataFrame:
    original_cols = df.columns.to_list()

    # make columns lowercase and replace spaces with underscore
    named_cols = [col.lower().replace(' ', '_') for col in df.columns.to_list()]
    new_cols = ['date', 'time', 'duration', 'time_point'] + named_cols[4:]

    mapping = dict(zip(original_cols, new_cols))

    return df.rename(columns=mapping)


def rearrange_columns(*, df: pd.DataFrame) -> pd.DataFrame:
    # move 'subject_id' from last index to first
    cols = df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    return df[cols]


def normalize_time(*, df: pd.DataFrame) -> pd.DataFrame:
    df['time'] = df.groupby(['subject_id'])['duration'].transform(pd.Series.cumsum)
    return df


def replace_nan(*, df: pd.DataFrame, columns: str) -> pd.DataFrame:
    df[columns].loc[df[columns] == '#NUM!'] = np.nan
    return df


def get_compact_df(*, df: pd.DataFrame) -> pd.DataFrame:
    """
    return a compact df that has only the time points we want for CVR

    :param df:
    :return:
    """
    return df.loc[df['time'].isin([300, 480, 660])]


def calculate_cvr_vs_homa(*, df: pd.DataFrame, **kwargs) -> linregress:
    x = df['homa_ir'].values.reshape(-1, 1)
    y = df['cvr'].values.reshape(-1, 1)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return linregress(x[mask], y[mask])


def calculate_cvr(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    x = df['etco2'].values.reshape(-1, 1)
    y = df['tcd1'].values.reshape(-1, 1)
    lm = linregress(x[:, 0], y[:, 0])
    return lm.slope


def calculate_delta(*, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    calculate the delta for each variable vs. baseline value
    source: https://stackoverflow.com/questions/63346512/subtract-fixed-row-value-in-reference-to-column-value-in-pandas-dataframe?newreg=b813a38e555b46fc8e879d05d0661689

    :return:
    """
    variable_cols = df.columns.to_list()
    cols_to_remove = [
        'subject_id',
        'date',
        'time',
        'duration',
        'time_point'
    ]
    for col in cols_to_remove:
        variable_cols.remove(col)

    for col in variable_cols:
        s = df.groupby("subject_id")[col].transform('first')
        new_col_name = f"delta_{col}"
        df[new_col_name] = df[col] - s

    return df


def get_homa_ir(*, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    some magic numbers until ETL from REDCap to SQL can occur

    :param df:
    :param kwargs:
    :return:
    """
    subjects = [
        '0838_001',
        '0838_001_a',
        '0838_003',
        '0838_004',
        '0838_005',
        '0838_010',
        '0838_011',
        '0838_016',
        '0838_018',
        '0838_020',
        '0838_022'
    ]

    homa_ir = [
        8.209876543,
        1.950617284,
        4.096296296,
        1.6,
        np.nan,
        1.296296296,
        1.87654321,
        0.864197531,
        1.204938272,
        1.155555556,
        1.572839506
    ]
    mapping = dict(zip(subjects, homa_ir))
    df['homa_ir'] = df['subject_id'].map(mapping)
    return df


def main() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, linregress, go.Figure]:
    path = root_path()
    files = get_files(path=path)
    df = make_df(files=files)
    df = drop_columns(df=df)
    df = rename_columns(df=df)
    df = rearrange_columns(df=df)
    df = replace_nan(df=df, columns='tidal_volume')
    df = replace_nan(df=df, columns='minute_ventilation')
    df = set_dtypes(df=df)
    df = hack_for_subjects_with_id_001(df=df)
    df = normalize_time(df=df)
    df_compact = get_compact_df(df=df)
    df_compact = calculate_delta(df=df_compact)
    df_cvr = df_compact.groupby("subject_id").apply(calculate_cvr)
    df_cvr = df_cvr.to_frame()
    df_cvr.reset_index(inplace=True)
    df_cvr.rename(columns={0: "cvr"}, inplace=True)
    df_cvr = get_homa_ir(df=df_cvr)
    lm = calculate_cvr_vs_homa(df=df_cvr)
    fig = plot_results(df=df_cvr, lm=lm)
    return df, df_compact, df_cvr, lm, fig


def plot_results(*, df: pd.DataFrame, lm: linregress, **kwargs) -> go.Figure:
    y_pred = lm.intercept + lm.slope * df['homa_ir']
    fig = go.Figure([
        go.Scatter(
            name='data',
            x=df['homa_ir'],
            y=df['cvr'],
            mode='markers',
            marker={
                'color': 'red'
            },
            text=df['subject_id'],
            hovertemplate="<b>%{text}</b><br><br>" +
                          "CVR = %{y:.2f} cm/s/mmHg<br>" +
                          "HOMA-IR = %{x:.2f}<br>"
        ),
        go.Scatter(
            name='regression',
            x=np.linspace(df['homa_ir'].min(), df['homa_ir'].max(), 1000),
            y=np.linspace(y_pred.min(), y_pred.max(), 1000),
            mode='lines',
            marker={
                'color': 'black'
            },
            hovertext=[f"slope = {lm.slope:.2f}\nstderr = {lm.stderr:.2f}\nr = {lm.rvalue:.2f}\np = {lm.pvalue:.2f}"] * 1000
        )
    ])
    fig.update_layout(
        title="Cerebrovascular Reactivity as a Function of HOMA-IR in Adolescents",
        xaxis_title="HOMA-IR",
        yaxis_title="CVR (cm/s/mmHg)"
    )
    return fig


if __name__ == '__main__':
    main()

"""
0361 id | 0838 id | glucose | insulin
93 | 1 << treat as separate subjects though due to length of time between studies
263 | 3
301 | 4
259 | 5
285 | 6
282 | 7
279 | 8
298 | 9
280 | 10
251 | 11
319 | 12
250 | 13
294 | 14
307 | 15
264 | 16
302 | 18
262 | 19
313 | 20
224 | 21 | 86 | 5
223 | 22 | 91 | 7
334 | 23
335 | 24
28 | 26 | 79 | 10

"""