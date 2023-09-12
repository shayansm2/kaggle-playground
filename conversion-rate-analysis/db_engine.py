import sqlalchemy


class DBEngine(object):
    @staticmethod
    def get_digikala_engine() -> sqlalchemy.Engine:
        user = 'user'
        password = 'pass'
        host = 'localhost'
        port = '80'
        database = 'db_log'
        assert user != "" and password != "", "please enter your user and password"
        return sqlalchemy.create_engine(
            f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

    @staticmethod
    def get_digikala_log_engine() -> sqlalchemy.Engine:
        user = 'user'
        password = 'pass'
        host = 'localhost'
        port = '80'
        database = 'db'
        assert user != "" and password != "", "please enter your user and password"
        return sqlalchemy.create_engine(
            f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')
