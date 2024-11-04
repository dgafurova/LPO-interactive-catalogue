from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from internal.model.orbit import Orbit
from internal.model.error import NotFoundError
from sqlalchemy.orm import Session
from typing import List


class OrbitRepo:
    engine: Engine

    def __init__(self, engine: Engine):
        self.engine = engine

    def create_orbit(self, orbit: Orbit) -> int:
        with Session(self.engine) as session:
            try:
                session.add(orbit)
                session.commit()
                return orbit.id
            except SQLAlchemyError as e:
                session.rollback()
                raise RuntimeError(f"Database error occurred: {e}")

    def get_by_id(self, idx: int) -> Orbit:
        with Session(self.engine) as session:
            orbit = session.query(Orbit).get({"id": idx})
            if not orbit:
                raise NotFoundError

        return orbit

    def get_all_map(self, filters: dict, ids: list = None) -> List:
        with Session(self.engine) as session:
            orbits = session.query(Orbit).with_entities(Orbit.id, Orbit.x, Orbit.z, Orbit.v)

            if filters["t_min"] is not None:
                orbits = orbits.filter(Orbit.t_period >= filters["t_min"])

            if filters["t_max"] is not None:
                orbits = orbits.filter(Orbit.t_period <= filters["t_max"])

            if filters["ax_min"] is not None:
                orbits = orbits.filter(Orbit.ax >= filters["ax_min"])

            if filters["ax_max"] is not None:
                orbits = orbits.filter(Orbit.ax <= filters["ax_max"])

            if filters["ay_min"] is not None:
                orbits = orbits.filter(Orbit.ay >= filters["ay_min"])

            if filters["ay_max"] is not None:
                orbits = orbits.filter(Orbit.ay <= filters["ay_max"])

            if filters["az_min"] is not None:
                orbits = orbits.filter(Orbit.az >= filters["az_min"])

            if filters["az_max"] is not None:
                orbits = orbits.filter(Orbit.az <= filters["az_max"])

            if filters["cj_min"] is not None:
                orbits = orbits.filter(Orbit.cj >= filters["cj_min"])

            if filters["cj_max"] is not None:
                orbits = orbits.filter(Orbit.cj <= filters["cj_max"])

            if ids is not None:
                orbits = orbits.filter(Orbit.id.in_(ids))

            orbits = orbits.all()

        return orbits

    def get_all_meta(self, filters: dict, ids: list = None) -> List[Orbit]:
        with Session(self.engine) as session:
            orbits = session.query(Orbit).with_entities(
                Orbit.id, Orbit.x, Orbit.z, Orbit.v, Orbit.t_period,
                Orbit.ax, Orbit.ay, Orbit.az, Orbit.cj, Orbit.stable
            )

            if filters["t_min"] is not None:
                orbits = orbits.filter(Orbit.t_period >= filters["t_min"])

            if filters["t_max"] is not None:
                orbits = orbits.filter(Orbit.t_period <= filters["t_max"])

            if filters["ax_min"] is not None:
                orbits = orbits.filter(Orbit.ax >= filters["ax_min"])

            if filters["ax_max"] is not None:
                orbits = orbits.filter(Orbit.ax <= filters["ax_max"])

            if filters["ay_min"] is not None:
                orbits = orbits.filter(Orbit.ay >= filters["ay_min"])

            if filters["ay_max"] is not None:
                orbits = orbits.filter(Orbit.ay <= filters["ay_max"])

            if filters["az_min"] is not None:
                orbits = orbits.filter(Orbit.az >= filters["az_min"])

            if filters["az_max"] is not None:
                orbits = orbits.filter(Orbit.az <= filters["az_max"])

            if filters["cj_min"] is not None:
                orbits = orbits.filter(Orbit.cj >= filters["cj_min"])

            if filters["cj_max"] is not None:
                orbits = orbits.filter(Orbit.cj <= filters["cj_max"])

            if len(ids) is not None:
                orbits = orbits.filter(Orbit.id.in_(ids))

            # from sqlalchemy.dialects import postgresql  # or the appropriate dialect for your DB
            # print(orbits.statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
            orbits = orbits.all()

        return orbits

    def get_by_family(self, ids: list):
        with Session(self.engine) as session:
            orbits = session.query(Orbit).with_entities(
                Orbit.id, Orbit.t_period, Orbit.ax, Orbit.ay, Orbit.az, Orbit.dist_primary, Orbit.dist_secondary,
                Orbit.cj, Orbit.l1_r, Orbit.l2_r, Orbit.l3_r, Orbit.l4_r, Orbit.l5_r, Orbit.l6_r, Orbit.l1_im,
                Orbit.l2_im, Orbit.l3_im, Orbit.l4_im, Orbit.l5_im, Orbit.l6_im
            ).filter(Orbit.id.in_(ids)).all()

        return orbits

    def get_ids_by_cj(self, ids: list, cj_min: float, cj_max: float):
        with Session(self.engine) as session:
            orbits = session.query(Orbit).with_entities(Orbit.id).filter(
                Orbit.cj >= cj_min, Orbit.cj <= cj_max, Orbit.id.in_(ids)
            ).all()

        return orbits


# if __name__ == "__main__":
#     from internal.app.app import db
#     o_r = OrbitRepo(db.engine)
#     orbits = o_r.get_all_map(
#         {"t_min": 0, "t_max": 0, "ax_min": None, "ax_max": None, "ay_min": None, "ay_max": None, "az_min": None,
#          "az_max": None, "cj_min": None, "cj_max": None},
#         [8, 9]
#     )
#
#     print(orbits)
