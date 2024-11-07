from sqlalchemy.engine.base import Engine
from internal.model.poincare_section import PoincareSection
from sqlalchemy.exc import SQLAlchemyError
from internal.model.error import NotFoundError
from sqlalchemy.orm import Session
from typing import List


class PoincareSectionRepo:
    engine: Engine

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_by_pk(self, idx: int, plane: str) -> List:
        with Session(self.engine) as session:
            trajectory = session.query(PoincareSection).filter(
                PoincareSection.orbit_id == idx, PoincareSection.plane == plane).first()
            if not trajectory:
                raise NotFoundError

        return trajectory.v

    def create_poincare_section(self, p: PoincareSection):
        with Session(self.engine) as session:
            try:
                session.add(p)
            except SQLAlchemyError as e:
                session.rollback()
                raise RuntimeError(f"Database error occurred: {e}")

            session.commit()

    def get_by_ids(self, ids: list, plane: str):
        with Session(self.engine) as session:
            trajectories = session.query(PoincareSection).filter(
                PoincareSection.orbit_id.in_(ids), PoincareSection.plane == plane
            ).all()

        return trajectories


# if __name__ == "__main__":
#     from internal.app.app import db
#     p_r = PoincareSectionRepo(db.engine)
#     t = p_r.get_by_pk(2, 'x')
#     print(t)
