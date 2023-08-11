import hashlib
import os
from datetime import datetime
from functools import wraps
from typing import Callable

import sqlalchemy
import pandas as pd


def hash_args_kwargs(func, *args, **kwargs):
    current_hour = datetime.now().hour
    data = f"{func.__name__}{args}{kwargs}{current_hour}".encode()
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
        user = 'user'
        password = 'pass'
        host = 'localhost'
        port = '80'
        database = 'db_log'
        assert user != "" and password != "", "please enter your user and password"
        self._log_engine = sqlalchemy.create_engine(
            f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

        user = 'user'
        password = 'pass'
        host = 'localhost'
        port = '80'
        database = 'db'
        assert user != "" and password != "", "please enter your user and password"
        self._engine = sqlalchemy.create_engine(
            f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

    @cache_db_data
    def get_funnel_steps_log(self, start_date: str) -> pd.DataFrame:
        query = f'''
        select entity_id as cart_id,
               changed_at,
               json_extract(changes, '$.funnel_step') as funnel_step
        from digikala_log.carts_log
        where changed_at > '{start_date}'
          and json_extract(changes, '$.funnel_step') is not null
        order by id desc;
        '''

        df = pd.read_sql(query, self._log_engine)
        df['changed_at'] = pd.to_datetime(df['changed_at'])
        df.sort_values(by=['cart_id', 'changed_at'], inplace=True)
        return df

    @cache_db_data
    def get_user_data(self, start_date: str):
        query = f'''
        select c.id as cart_id, u.id as user_id, u.gender, u.foreigner, u.customer_type, u.customer_clustering_rate, u.user_category
        from digikala.carts c
                 join digikala.users u on c.user_id = u.id
        where c.created_at > '{start_date}'
          and c.site = 'digikala'
          and c.business_type = 'b2c';
        '''

        df = pd.read_sql(query, self._engine)
        return df

    @cache_db_data
    def get_cart_data(self, start_date: str):
        query = f'''
        select id, status, payable_price, source, source_close
        from digikala.carts
        where created_at > '{start_date}'
          and site = 'digikala'
          and business_type = 'b2c'
          and source != 'core'
          and source != 'mehr';
        '''

        df = pd.read_sql(query, self._engine)
        return df
