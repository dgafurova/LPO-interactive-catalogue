from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine


class DB:
    engine: Engine

    def __init__(self, dsn: str):
        self.engine = create_engine(dsn)

# def new_postgres_db(config: dict) -> Engine:
#     return create_engine(
#         f'postgresql+psycopg2://{config["user"]}:{config["password"]}@{config["hostname"]}/{config["db_name"]}'
#     )
