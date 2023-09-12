import hashlib
import os
import pandas as pd

from functools import wraps
from typing import Callable
from db_engine import DBEngine


def hash_args_kwargs(func, *args, **kwargs):
    data = f"{func.__name__}{args}{kwargs}".encode()
    return hashlib.md5(data).hexdigest()


def cache_db_data(func: Callable):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        file_name = hash_args_kwargs(func, args, kwargs) + '.csv'
        if os.path.exists(file_name):
            return pd.read_csv(file_name)
        df: pd.DataFrame = func(self, *args, **kwargs)
        df.to_csv(file_name)
        return df

    return wrapper


class DB(object):
    def __init__(self):
        self._log_engine = DBEngine.get_digikala_log_engine()
        self._engine = DBEngine.get_digikala_engine()

    @cache_db_data
    def get_funnel_steps_log(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        query = f'''
        select entity_id as cart_id,
               changed_at,
               json_extract(changes, '$.funnel_step') as funnel_step
        from digikala_log.carts_log
        where changed_at > '{start_date}'
          and json_extract(changes, '$.funnel_step') is not null
        '''

        if end_date is not None:
            query += f'and changed_at < {end_date}'

        query += 'order by id desc;'

        df = pd.read_sql(query, self._log_engine)
        df['changed_at'] = pd.to_datetime(df['changed_at'])
        df.sort_values(by=['cart_id', 'changed_at'], inplace=True)
        return df

    @cache_db_data
    def get_user_data(self, start_date: str, end_date: str = None):
        query = f'''
        select c.id as cart_id, u.id as user_id, u.gender, u.foreigner, u.customer_type, u.customer_clustering_rate, u.user_category
        from digikala.carts c
                 join digikala.users u on c.user_id = u.id
        where c.created_at > '{start_date}'
          and c.site = 'digikala'
          and c.business_type = 'b2c'
        '''

        if end_date is not None:
            query += f'and changed_at < {end_date}'

        query += ';'

        df = pd.read_sql(query, self._engine)
        return df

    @cache_db_data
    def get_cart_data(self, start_date: str, end_date: str = None):
        query = f'''
        select id, status, payable_price, source, source_close
        from digikala.carts
        where created_at > '{start_date}'
          and site = 'digikala'
          and business_type = 'b2c'
          and source != 'core'
          and source != 'mehr'
        '''

        if end_date is not None:
            query += f'and changed_at < {end_date}'

        query += ';'

        df = pd.read_sql(query, self._engine)
        return df
