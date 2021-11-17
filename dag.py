from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from main import (
    set_dtypes,
    drop_columns,
    rename_columns,
    normalize_time,
    replace_nan,
    calculate_delta,
    calculate_cvr,
)

with DAG(dag_id='hypercapnia-processing', schedule_interval=None, start_date=datetime(2021, 11, 16), catchup=False) as dag:
    # probably start here for an individual file
    _drop_columns = PythonOperator(
        task_id='drop-columns',
        python_callable=drop_columns,
        op_kwargs={
            'df': ""
        }
    )

    _rename_columns = PythonOperator(
        task_id='rename-columns',
        python_callable=rename_columns,
        op_kwargs={
            'df': ""
        }
    )
    _drop_columns >> _rename_columns

    # don't need to rearrange columns anymore as it's no longer interactive

    _replace_nan = PythonOperator(
        task_id='replace-nan',
        python_callable=replace_nan,
        op_kwargs={
            'df': ""
        }
    )
    _rename_columns >> _replace_nan

    _set_dtypes = PythonOperator(
        task_id='set-dtypes',
        python_callable=set_dtypes,
        op_kwargs={
            'df': ""
        }
    )
    _replace_nan >> _set_dtypes

    _normalize_time = PythonOperator(
        task_id='normalize-time',
        python_callable=normalize_time,
        op_kwargs={
            'df': ""
        }
    )
    _set_dtypes >> _normalize_time

    _calculate_deltas = PythonOperator(
        task_id='calculate-deltas',
        python_callable=calculate_delta,
        op_kwargs={
            'df': ""
        }
    )
    _normalize_time >> _calculate_deltas

    _calculate_cvr = PythonOperator(
        task_id='calculate-cvr',
        python_callable=calculate_cvr,
        op_kwargs={
            'df': ""
        }
    )
    _calculate_deltas >> _calculate_cvr

    # todo: load to sql db