from sqlalchemy.engine.base import Engine
from internal.model.libration_point import LibrationPoint
from internal.model.error import NotFoundError
from sqlalchemy.orm import Session


class LibrationPointRepo:
    engine: Engine

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_by_id(self, idx: int) -> LibrationPoint:
        with Session(self.engine) as session:
            libration_points = session.query(LibrationPoint).get({"id": idx})
            if not libration_points:
                raise NotFoundError

        return libration_points
