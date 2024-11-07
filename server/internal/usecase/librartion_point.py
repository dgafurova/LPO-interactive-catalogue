from internal.model.libration_point import LibrationPoint
from internal.db.libration_point import LibrationPointRepo


class LibrationPointUnit:
    libration_point_repo: LibrationPointRepo

    def __init__(self, libration_point_repo: LibrationPointRepo):
        self.libration_point_repo = libration_point_repo

    def get_by_id(self, idx: int) -> LibrationPoint:
        return self.libration_point_repo.get_by_id(idx)
