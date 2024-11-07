def NextDxNewton(dx, y0, y1, dxmax, tol):
    """
    Вычисляет следующий шаг изменения dx на основе метода Ньютона.

    Параметры:
        dx (float): Текущий шаг изменения.
        y0 (float): Значение функции в предыдущей точке.
        y1 (float): Значение функции в текущей точке.
        dxmax (float): Максимально допустимый шаг изменения.
        tol (float): Толеранс для прекращения изменений (погрешность).

    Возвращает:
        float: Новое значение шага dx, скорректированное на основе метода Ньютона.
    """
    NewDx = 0
    if abs(y1) >= tol and y0 != y1:
        NewDx = dx
        if y1 != 0:
            # Корректируем шаг изменения с учетом отношения y1 к разности y0 и y1
            NewDx = max(min(dx * y1 / (y0 - y1), dxmax), -dxmax)

    return NewDx


class optimizer:
    """
    Класс optimizer управляет процессом оптимизации, хранит историю значений x и y,
    определяет необходимость следующего шага и вычисляет следующее значение x для оптимизации.
    """

    # Атрибуты класса с начальными значениями
    dx = 0.001
    dxmax = 0.1
    xvals = []
    yvals = []
    step = 0
    tolerance = 0.000001
    output = False

    def __init__(self):
        """
        Инициализирует экземпляр класса optimizer, устанавливая начальные значения списков xvals и yvals,
        а также счетчик шагов step.
        """
        self.xvals = []
        self.yvals = []
        self.step = 0

    def inCycle(self):
        """
        Определяет, находится ли процесс оптимизации в цикле.

        Возвращает:
            bool: True, если обнаружен цикл, иначе False.
        """
        if self.step <= 4:
            return False
        # Проверяем, достаточно ли близки значения x на шагах 2 и 4 с учетом машинной точности
        if abs(self.xvals[2] - self.xvals[4]) < 1e-16:
            return True
        return False

    def needNextStep(self):
        """
        Определяет, необходим ли следующий шаг оптимизации.

        Возвращает:
            bool: True, если необходимо продолжать оптимизацию, иначе False.
        """
        if len(self.yvals) <= 2:
            return True
        # Проверка на циклическое поведение (закомментировано)
        # if self.inCycle():
        #     if self.output:
        #         print('Cycled')
        #     return False
        # Проверяем изменение первых двух значений x с учетом толеранса
        if abs(self.xvals[0] - self.xvals[1]) <= self.tolerance:
            return False
        else:
            return True

    def nextX(self, x, y):
        """
        Вычисляет следующее значение x для оптимизации на основе текущих значений x и y.

        Параметры:
            x (float): Текущее значение x.
            y (float): Текущее значение y.

        Возвращает:
            float: Следующее значение x для использования в процессе оптимизации.
        """
        self.step += 1

        i = 0
        # Находим место для добавления точки, чтобы массив остался отсортированным
        while i < min(2, len(self.yvals)) and y > self.yvals[i]:
            i += 1
        self.xvals.insert(i, x)
        self.yvals.insert(i, y)

        xvals = self.xvals
        yvals = self.yvals

        if self.step == 1:
            return x + self.dx

        # Вычисляем новое значение x1 с использованием функции NextDxNewton
        x1 = xvals[0] + NextDxNewton(
            xvals[1] - xvals[0], yvals[0], yvals[1], self.dx, 0
        )
        if self.step > 2:
            # Квадратичная аппроксимация для улучшения шага
            a = (
                yvals[0] / (xvals[0] - xvals[1]) / (xvals[0] - xvals[2])
                - yvals[1] / (xvals[0] - xvals[1]) / (xvals[1] - xvals[2])
                + yvals[2] / (xvals[1] - xvals[2]) / (xvals[0] - xvals[2])
            )
            if a > 0:
                b = (yvals[0] - yvals[1]) / (xvals[0] - xvals[1]) - a * (
                    xvals[0] + xvals[1]
                )
                xm = -b / (2 * a)
                # Корректируем x1 с учетом dxmax
                x1 = x + max(min(xm - x, self.dxmax), -self.dxmax)

        # Проверка на циклическое поведение (закомментировано)
        # if self.inCycle():
        #     x1 = (x1 + 2 * x) / 3

        return x1

    def getXY(self):
        """
        Возвращает текущее оптимальное значение x и соответствующее ему y.

        Возвращает:
            tuple: Кортеж (x, y) с наилучшими текущими значениями.
        """
        return self.xvals[0], self.yvals[0]

    def getdx(self):
        """
        Возвращает текущий шаг изменения dx как абсолютную разницу между первыми двумя значениями x в списке xvals.

        Возвращает:
            float: Текущий шаг изменения dx.
        """
        return abs(self.xvals[0] - self.xvals[1])


# Пример использования


# Предположим, что у нас есть функция f, которую мы хотим оптимизировать (например, найти корень)
def f(x):
    return x**2 - 4  # Пример: корни при x=2 и x=-2


if __name__ == "__main__":
    # Создание экземпляра оптимизатора
    opt = optimizer()

    # Начальные значения
    current_x = 0.0
    current_y = f(current_x)

    # Включение вывода сообщений о циклах (если необходимо)
    opt.output = True

    # Процесс оптимизации
    while opt.needNextStep():
        next_x = opt.nextX(current_x, current_y)
        current_y = f(next_x)
        current_x = next_x
        if opt.output:
            print(f"Step {opt.step}: x = {current_x}, y = {current_y}")

    # Получение оптимального значения
    optimal_x, optimal_y = opt.getXY()
    print(f"Optimal x: {optimal_x}, y: {optimal_y}")
