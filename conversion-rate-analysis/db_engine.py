import sqlalchemy


class DBEngine(object):
    @staticmethod
    def get_digikala_engine() -> sqlalchemy.Engine:
        user = 's.shafieemoghadam'
        password = 'WITd0Xnfnmos3XlvnHoBFA=='
        host = '172.30.5.64'
        port = '13306'
        database = 'digikala'
        assert user != "" and password != "", "please enter your user and password"
        return sqlalchemy.create_engine(
            f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

    @staticmethod
    def get_digikala_log_engine() -> sqlalchemy.Engine:
        user = 'h.zand'
        password = '951c898a1f9af97acec3fb9b8df4'
        host = '172.30.5.214'
        port = '13306'
        database = 'digikala_log'
        assert user != "" and password != "", "please enter your user and password"
        return sqlalchemy.create_engine(
            f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')
