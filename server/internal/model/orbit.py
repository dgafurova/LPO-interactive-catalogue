from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class Orbit(Base):
    __tablename__ = "orbits"
    # __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(name="id", primary_key=True)
    x: Mapped[float] = mapped_column(name="x")
    z: Mapped[float] = mapped_column(name="z")
    v: Mapped[float] = mapped_column(name="v")
    alpha: Mapped[float] = mapped_column(name="alpha")
    t_period: Mapped[float] = mapped_column(name="t")
    l1_r: Mapped[float] = mapped_column(name="l1_r")
    l2_r: Mapped[float] = mapped_column(name="l2_r")
    l3_r: Mapped[float] = mapped_column(name="l3_r")
    l4_r: Mapped[float] = mapped_column(name="l4_r")
    l5_r: Mapped[float] = mapped_column(name="l5_r")
    l6_r: Mapped[float] = mapped_column(name="l6_r")
    l1_im: Mapped[float] = mapped_column(name="l1_im")
    l2_im: Mapped[float] = mapped_column(name="l2_im")
    l3_im: Mapped[float] = mapped_column(name="l3_im")
    l4_im: Mapped[float] = mapped_column(name="l4_im")
    l5_im: Mapped[float] = mapped_column(name="l5_im")
    l6_im: Mapped[float] = mapped_column(name="l6_im")
    ax: Mapped[float] = mapped_column(name="ax")
    ay: Mapped[float] = mapped_column(name="ay")
    az: Mapped[float] = mapped_column(name="az")
    dist_primary: Mapped[float] = mapped_column(name="dist_primary")
    dist_secondary: Mapped[float] = mapped_column(name="dist_secondary")
    dist_curve: Mapped[float] = mapped_column(name="dist_curve")
    cj: Mapped[float] = mapped_column(name="cj")
    stable: Mapped[bool] = mapped_column(name="stable")
    stability_order: Mapped[int] = mapped_column(name="stability_order")
