from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class OrbitFamily(Base):
    __tablename__ = "orbit_families"

    id: Mapped[int] = mapped_column(name="id", primary_key=True)
    orbit_id: Mapped[int] = mapped_column(name="orbit_id")
    tag: Mapped[str] = mapped_column(name="tag")
