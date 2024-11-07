from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import JSON


class Base(DeclarativeBase):
    pass


class Trajectory(Base):
    __tablename__ = "trajectories"

    orbit_id: Mapped[int] = mapped_column(name="orbit_id", primary_key=True)
    v: Mapped[JSON] = mapped_column(type_=JSON, name="v")
    x: Mapped[int] = mapped_column(name="x")
    y: Mapped[int] = mapped_column(name="y")
    z: Mapped[int] = mapped_column(name="z")
    vx: Mapped[int] = mapped_column(name="vx")
    vy: Mapped[int] = mapped_column(name="vy")
    vz: Mapped[int] = mapped_column(name="vz")
