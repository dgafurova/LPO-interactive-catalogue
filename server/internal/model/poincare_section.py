from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import JSON
from sqlalchemy.schema import PrimaryKeyConstraint


class Base(DeclarativeBase):
    pass


class PoincareSection(Base):
    __tablename__ = "poincare_sections"

    orbit_id: Mapped[int] = mapped_column(name="orbit_id", primary_key=True)
    plane: Mapped[str] = mapped_column(name="plane", primary_key=True)
    v: Mapped[JSON] = mapped_column(type_=JSON, name="v")

    __table_args__ = (
        PrimaryKeyConstraint('orbit_id', 'plane'),
    )
