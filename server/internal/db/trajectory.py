from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from internal.model.trajectory import Trajectory
from internal.model.error import NotFoundError
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from typing import List


class TrajectoryRepo:
    engine: Engine

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_by_id(self, orbit_id: int) -> List:
        with Session(self.engine) as session:
            trajectory = session.query(Trajectory).filter(Trajectory.orbit_id == orbit_id).first()
            if not trajectory:
                raise NotFoundError

        return trajectory.v

    def get_nearest(self, x: float, y: float, z: float, vx: float, vy: float, vz: float):
        """Return list of vectors for nearest trajectory by L2 norm"""
        raw_sql = f"select orbit_id, v from trajectories where" \
                  f"(x - {x})^2 + (y - {y})^2 + (z - {z})^2 + (vx - {vx})^2 + (vy - {vy})^2 + (vz - {vz})^2 = (" \
                  f"SELECT MIN((x - {x})^2 + (y - {y})^2 + (z - {z})^2 + " \
                  f"(vx - {vx})^2 + (vy - {vy})^2 + (vz - {vz})^2) as res " \
                  f"FROM trajectories);"

        with self.engine.connect() as connection:
            result = connection.execute(text(raw_sql))
            trajectory = result.fetchone()

            if trajectory is None:
                raise NotFoundError()

        return trajectory

    def create_trajectory(self, t: Trajectory):
        with Session(self.engine) as session:
            try:
                session.add(t)
            except SQLAlchemyError as e:
                session.rollback()
                raise RuntimeError(f"Database error occurred: {e}")

            session.commit()


# if __name__ == "__main__":
#     from internal.app.app import db
#     t_r = TrajectoryRepo(db.engine)
#     trajectory = t_r.get_nearest(1, 1, 1, 1, 1, 1)
#     print(trajectory)
