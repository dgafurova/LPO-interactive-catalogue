class NotFoundError(Exception):
    def __init__(self):
        super().__init__(f"not found")


class NotValidError(Exception):
    def __init__(self):
        super().__init__(f"not valid")
