# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 00:31:13 2018

@author: stasb

Этот модуль предназначен для техники обнаружения событий.

Событие определяется как момент времени, когда некоторая функция E(t, s), называемая
функцией события, достигает специфического значения E*. Другими словами, обнаружение события
является задачей поиска корня для функции EF(t, s) = E(t, s) - E*.
Обнаружение событий делится на два этапа: разделение корней и уточнение корней.
Разделение корней заключается в нахождении двух последовательных моментов времени ti, tj
и состояний si, sj (рассчитанных через интегрирование) таких, что EF(ti, si) * EF(tj, sj) < 0.
Уточнение корней — это вычисление времени t* и состояния s*, таких что |EF(t*, s*)| < eps,
где eps — заданная точность.

Разделение корней учитывает направление, в котором функция события достигает значения E*:

направление <=> условие
    -1        EF(ti, si) > 0 и EF(tj, sj) < 0
     1        EF(ti, si) < 0 и EF(tj, sj) > 0
     0        EF(ti, si) * EF(tj, sj) < 0

События могут быть терминальными и нетерминальными. Терминальные события сообщают интегратору,
когда необходимо завершить процесс интегрирования, и поэтому могут рассматриваться как граничные условия.
"""

import math
import pandas as pd
import numpy as np
from scipy.optimize import root as scipy_root

# from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from numba import njit, types  # компилятор, типы
import pkg_resources

from orbipy.solout import default_solout, event_solout
from model_integrator import base_model
from orbipy.plotter import plottable


class event_detector:
    """
    Класс event_detector предназначен для обнаружения событий во время
    интегрирования системы ОДУ.
    """

    columns = ["e", "cnt", "trm"]

    def __init__(self, model, events, tol=1e-12, accurate_model=None):
        """
        Инициализирует экземпляр класса event_detector.

        Параметры:
            model (base_model): Модель, которую необходимо интегрировать.
            events (list): Список событий для обнаружения.
            tol (float): Толеранс для поиска корней.
            accurate_model (base_model, optional): Более точная модель для уточнения событий.
        """
        # Проверяем, является ли модель экземпляром base_model или его подкласса
        if not isinstance(model, base_model):
            raise TypeError(
                "model должен быть экземпляром класса base_model или его подкласса"
            )
        self.model = model
        if accurate_model is not None and not isinstance(accurate_model, base_model):
            raise TypeError(
                "accurate_model должен быть экземпляром класса base_model или его подкласса"
            )
        self.accurate_model = model if accurate_model is None else accurate_model
        # Проверяем, что список событий не пуст
        # if not events:
        #     raise ValueError("Список событий не должен быть пустым")
        self.events = events
        self.solout = event_solout(self.events)
        self.tol = tol

    def prop(self, s0, t0, t1, ret_df=True, last_state="none"):
        """
        Пропагирует состояние модели от времени t0 до t1, обнаруживая события.

        Параметры:
            s0 (array): Начальное состояние.
            t0 (float): Начальное время.
            t1 (float): Конечное время.
            ret_df (bool): Возвращать ли результаты в виде DataFrame.
            last_state (str): Как сделать последнее состояние траектории в соответствии с терминальными событиями.
                - 'none'  : Не изменять траекторию;
                - 'last'  : Последнее событие является терминальным с большим индексом строки;
                - 'mint'  : Последнее событие является терминальным, произошедшим раньше других терминальных событий.

        Возвращает:
            tuple: (df, evdf) если ret_df=True, иначе (numpy.ndarray, numpy.ndarray)
        """
        if self.model.polar:
            return self.prop_polar(s0, t0, t1, ret_df=ret_df, last_state=last_state)
        for e in self.events:
            if hasattr(e, "reset"):
                e.reset()
        old_solout, old_solout2 = self.model.solout, self.accurate_model.solout
        self.model.solout = self.solout
        df = self.model.prop(s0, t0, t1, ret_df=False)
        evdf = self.accurate(ret_df=False)
        self.model.solout, self.accurate_model.solout = old_solout, old_solout2

        # Сделать последнее состояние согласованным с терминальными событиями
        last_state = last_state.lower()
        if evdf.shape[0] > 0 and last_state != "none":
            cev, arr = self.split_data(evdf)
            arr_trm = arr[cev[:, 2] == 1]
            if arr_trm.shape[0] > 0:
                if last_state == "last":
                    df[-1] = arr_trm[-1]
                elif last_state == "mint":
                    s = arr_trm[np.argmin(arr_trm[:, 0])]
                    df[-1] = s

        if ret_df:
            df = self.model.to_df(df)
            evdf = self.to_df(evdf)
        return df, evdf

    def accurate(self, ret_df=True):
        """
        Уточняет события с использованием более точной модели.

        Параметры:
            ret_df (bool): Возвращать ли результаты в виде DataFrame.

        Возвращает:
            DataFrame или numpy.ndarray: Уточненные события.
        """
        self.accurate_model.solout = default_solout()

        evout = []
        for e in self.solout.evout:
            event = self.events[e[0]]
            if event.accurate:
                ts = self.solout.out[e[2]]
                # print('ts shape:', len(ts))
                t0 = ts[0]
                s0 = ts[1:]
                t1 = self.solout.out[e[2] + 1][0]
                # print('root call:', 't0', t0, 't1', t1, 's0', s0)
                sa = self.root(event, t0, t1, s0)
                # print('acc shape:', len(sa))
            else:
                sa = self.solout.out[e[2]]
            # print('not acc shape:', len(sa))
            evout.append([e[0], e[1], event.terminal, *sa])

        if ret_df:
            df = self.to_df(evout)
            return df
        return np.array(evout)

    def root(self, event, t0, t1, s0):
        """
        Находит корень функции события в интервале [t0, t1].

        Параметры:
            event (base_event): Событие для поиска корня.
            t0 (float): Начальное время интервала.
            t1 (float): Конечное время интервала.
            s0 (array): Начальное состояние.

        Возвращает:
            array: Состояние в момент корня.
        """
        s_opt = [0]  # для сохранения рассчитанного состояния во время поиска корня

        # print('root finding', type(event))
        def froot(t, s0, t0):
            t = t[0]  # поскольку scipy.root передает [t] вместо t
            if t == t0:
                s = np.array([t, *s0])
            else:
                s = self.accurate_model.prop(s0, t0, t, ret_df=False)[-1]
            s_opt[0] = s
            res = event(t, s[1:])
            # print('t0', t0, 't', t, 's0', s0, 's', s, 'res', res)
            # print(event, t, s[1:], '->', res)
            return res

        scipy_root(froot, args=(s0, t0), x0=t0, tol=self.tol)
        # print('end of root finding')

        # scipy.optimize.solve_ivp использует:
        # brentq(froot, t0, t1, args=(s0, t0), xtol=self.tol, rtol=self.tol)
        return s_opt[0]

    def prop_polar(self, s0, t0, t1, ret_df=True, last_state="none"):
        """
        Пропагирует состояние модели в полярной системе координат.

        Параметры:
            s0 (array): Начальное состояние.
            t0 (float): Начальное время.
            t1 (float): Конечное время.
            ret_df (bool): Возвращать ли результаты в виде DataFrame.
            last_state (str): Как сделать последнее состояние траектории в соответствии с терминальными событиями.
                - 'none'  : Не изменять траекторию;
                - 'last'  : Последнее событие является терминальным с большим индексом строки;
                - 'mint'  : Последнее событие является терминальным, произошедшим раньше других терминальных событий.

        Возвращает:
            tuple: (df, evdf) если ret_df=True, иначе (numpy.ndarray, numpy.ndarray)
        """
        n = self.model._constants.shape[0]
        new_constants = np.empty(n + 1)
        new_constants[:-1] = self.model._constants
        new_constants[-1] = self.model.jacobi(s0)

        for e in self.events:
            if hasattr(e, "reset"):
                e.reset()
        old_solout, old_solout2 = self.model.solout, self.accurate_model.solout
        self.model.solout = self.solout
        df = self.model.prop_polar(s0, t0, t1, ret_df=False)  # t, x, y, z, vx, vy, vz
        evdf = self.accurate_polar(
            new_constants, ret_df=False
        )  # ev, cnt, trm, t, x, y, z, vx, vy, vz
        self.model.solout, self.accurate_model.solout = old_solout, old_solout2

        # Сделать последнее состояние согласованным с терминальными событиями
        last_state = last_state.lower()
        if evdf.shape[0] > 0 and last_state != "none":
            cev, arr = self.split_data(evdf)
            arr_trm = arr[cev[:, 2] == 1]
            if arr_trm.shape[0] > 0:
                if last_state == "last":
                    df[-1] = arr_trm[-1]
                elif last_state == "mint":
                    s = arr_trm[np.argmin(arr_trm[:, 0])]
                    df[-1] = s

        if ret_df:
            df = self.model.to_df(df)
            evdf = self.to_df(evdf)
        return df, evdf

    def accurate_polar(self, constants, ret_df=True):
        """
        Уточняет события в полярной системе координат с использованием более точной модели.

        Параметры:
            constants (array): Массив констант модели.
            ret_df (bool): Возвращать ли результаты в виде DataFrame.

        Возвращает:
            DataFrame или numpy.ndarray: Уточненные события.
        """
        self.accurate_model.solout = default_solout()

        evout = []
        for e in self.solout.evout:
            event = self.events[e[0]]
            if event.accurate:
                ts = self.solout.out[e[2]]
                t0 = ts[0]
                s0 = ts[1:]
                t1 = self.solout.out[e[2] + 1][0]
                sa = self.root_polar(event, t0, t1, s0, constants)
            else:
                sa = self.solout.out[e[2]]
                sa.append(0)
                sa[1:] = self.model.rphi2vxvy(sa[1:], constants)
            evout.append([e[0], e[1], event.terminal, *sa])

        if ret_df:
            df = self.to_df(evout)
            return df
        return np.array(evout)

    def root_polar(self, event, t0, t1, s0, constants):
        """
        Находит корень функции события в полярной системе координат в интервале [t0, t1].

        Параметры:
            event (base_event): Событие для поиска корня.
            t0 (float): Начальное время интервала.
            t1 (float): Конечное время интервала.
            s0 (array): Начальное состояние.
            constants (array): Массив констант модели.

        Возвращает:
            array: Состояние в момент корня.
        """
        s_opt = [0]  # для сохранения рассчитанного состояния во время поиска корня

        def froot(t, s0, t0, constants):
            t = t[0]  # поскольку scipy.root передает [t] вместо t
            if t == t0:
                s = np.array([t, *s0])
            else:
                s = self.accurate_model.prop_polar(
                    s0, t0, t, ret_df=False, new_constants=constants
                )[-1]
            s_opt[0] = s
            res = event(t, s[1:])
            return res

        scipy_root(froot, args=(s0, t0, constants), x0=t0, tol=self.tol)
        return s_opt[0]

    def to_df(self, arr, columns=None):
        """
        Преобразует массив данных в DataFrame.

        Параметры:
            arr (array-like): Массив данных.
            columns (list, optional): Список названий столбцов. Если не указано, используется класс event_detector.columns + model.columns.

        Возвращает:
            DataFrame: Таблица с данными.
        """
        if columns is None:
            columns = event_detector.columns + self.model.columns
        # print(len(columns), arr.shape)
        return self.model.to_df(arr, columns=columns)  # , index_col)

    def split_data(self, data):
        """
        Разделяет данные по столбцам: ['e', 'cnt', 'trm'] и ['t', 'x', 'y', ...].

        Параметры:
            data (DataFrame или array): Входные данные.

        Возвращает:
            tuple: (часть с ['e', 'cnt', 'trm'], часть с ['t', 'x', 'y', ...])
        """
        if isinstance(data, pd.DataFrame):
            # d = data.reset_index()
            return data[event_detector.columns], data[self.model.columns]
        n = len(event_detector.columns)
        return data[:, :n], data[:, n:]


class base_event(plottable):
    """
    Класс base_event является общим интерфейсом для всех классов событий в OrbiPy.
    Событие хранит необходимые данные и вычисляет значение функции события, которую оно представляет.
    Обнаружение события — это поиск корня функции события.
    """

    coord = "e"

    def __init__(self, value=0, direction=0, terminal=True, accurate=True, count=-1):
        """
        Инициализирует экземпляр класса base_event.

        Параметры:
            value (float): Целевое значение функции события E*.
            direction (int): Направление обнаружения события.
                - -1: Функция события убывает;
                - 1: Функция события возрастает;
                - 0: Любое направление.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        self.value = value
        self.terminal = terminal
        self.direction = direction
        self.accurate = accurate
        self.count = count

    def __call__(self, t, s):
        """
        Вычисляет значение функции события.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Значение функции события.
        """
        return 0

    def get_df(self):
        """
        Получает DataFrame с информацией о событии.

        Возвращает:
            DataFrame: Таблица с данными события.
        """
        return pd.DataFrame({self.coord: self.value}, index=[0])

    def plot_df(self, df, ax, projection, **kwargs):
        """
        Отображает событие на графике.

        Параметры:
            df (DataFrame): Данные для отображения.
            ax (Axes): Ось matplotlib для рисования.
            projection (str): Тип проекции ('x-y', 'x-z', 'y-z').
            **kwargs: Дополнительные параметры для рисования.
        """
        p = projection.split("-")
        c = self.coord
        kwargs["label"] = self.__repr__()
        if p[0] == c:
            ax.axvline(df[c].values, **kwargs)
        elif p[1] == c:
            ax.axhline(df[c].values, **kwargs)

    def __repr__(self):
        """
        Возвращает строковое представление события.

        Возвращает:
            str: Описание события.
        """
        return self.__class__.__name__ + ":[val]%r [dir]%r [trm]%r [acc]%r [cnt]%r" % (
            self.value,
            self.direction,
            self.terminal,
            self.accurate,
            self.count,
        )


class event_combine(base_event):
    """
    Класс event_combine объединяет список событий в одно событие, которое выглядит
    как первое событие в списке, но действует как все они одновременно.
    Это событие происходит, когда происходит любое из событий в списке.
    """

    def __init__(self, events):
        """
        Инициализирует экземпляр класса event_combine.

        Параметры:
            events (list): Список событий для объединения.
        """
        if not events:
            raise ValueError("Должно быть указано хотя бы одно событие")
        self.events = events

    def __getattr__(self, attr):
        """
        Делегирует атрибуты первому событию в списке.

        Параметры:
            attr (str): Имя атрибута.

        Возвращает:
            Любое: Значение атрибута из первого события.
        """
        # print('combine_getattr', attr)
        return getattr(self.events[0], attr)

    def __call__(self, t, s):
        """
        Вычисляет значение объединенного события.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Произведение значений всех событий.
        """
        ret = 1.0
        for e in self.events:
            ret *= e(t, s)
        return ret

    def __repr__(self):
        """
        Возвращает строковое представление объединенного события.

        Возвращает:
            str: Описание объединенного события.
        """
        return "event_combine: " + self.events.__repr__()


class event_chain(base_event):
    """
    Класс event_chain выглядит как последнее событие в цепочке, но работает как последовательность
    событий: события должны происходить в заданном порядке, и только последнее событие
    будет вести себя как событие.
    Все события в цепочке должны быть терминальными!
    """

    def __init__(self, events, autoreset=False):
        """
        Инициализирует экземпляр класса event_chain.

        Параметры:
            events (list): Список событий для цепочки.
            autoreset (bool): Автоматически сбрасывать цепочку после последнего события.
        """
        if not events:
            raise ValueError("Должно быть указано хотя бы одно событие")
        self.events = events
        self.autoreset = autoreset
        self.last = len(self.events) - 1
        self.select_event(0)

    def select_event(self, idx):
        """
        Выбирает текущее событие по индексу.

        Параметры:
            idx (int): Индекс события в цепочке.
        """
        # print('event_chain idx:', idx)
        self.idx = idx
        self.event_checker = event_solout([self.events[self.idx]])

    def reset(self):
        """
        Сбрасывает цепочку к первому событию.
        """
        self.select_event(0)

    def __getattr__(self, attr):
        """
        Делегирует атрибуты последнему событию в цепочке.

        Параметры:
            attr (str): Имя атрибута.

        Возвращает:
            Любое: Значение атрибута из последнего события.
        """
        # print('chain_getattr', attr)
        return getattr(self.events[self.last], attr)

    def __call__(self, t, s):
        """
        Вычисляет значение текущего события в цепочке.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Значение текущего события.
        """
        if self.idx == self.last:
            ret = self.events[self.idx](t, s)
            if self.autoreset and self.event_checker(t, s) == -1:
                self.reset()
            return ret
        else:
            if self.event_checker(t, s) == -1:
                self.select_event(self.idx + 1)  # выбираем следующее событие
            # Возвращаем 0, так как строгие неравенства в event_solout
            return 0.0

    def __repr__(self):
        """
        Возвращает строковое представление цепочки событий.

        Возвращает:
            str: Описание цепочки событий.
        """
        return "event_chain: " + self.events.__repr__()


class center_event(base_event):
    """
    Класс center_event является базовым классом для всех событий, которые используют
    центр (точку) в расчетах.
    """

    def __init__(
        self,
        center=np.zeros(3),
        value=0,
        direction=0,
        terminal=True,
        accurate=True,
        count=-1,
    ):
        """
        Инициализирует экземпляр класса center_event.

        Параметры:
            center (array-like): Центр, используемый в расчетах.
            value (float): Целевое значение функции события.
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(value, direction, terminal, accurate, count)
        self.center = center

    def __repr__(self):
        """
        Возвращает строковое представление центра события.

        Возвращает:
            str: Описание центра события.
        """
        return super().__repr__() + " [center]%r" % self.center


class center_angle_event(center_event):
    """
    Класс center_angle_event является базовым классом для всех событий, которые используют
    центр (точку) и угол в расчетах.
    """

    def __init__(
        self,
        center=np.zeros(3),
        flip=False,
        value=0,
        direction=0,
        terminal=True,
        accurate=True,
        count=-1,
    ):
        """
        Инициализирует экземпляр класса center_angle_event.

        Параметры:
            center (array-like): Центр, используемый в расчетах.
            flip (bool): Отражать угол относительно Y-оси.
            value (float): Целевой угол в градусах, принадлежащий сегменту [-pi, pi].
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(center, value, direction, terminal, accurate, count)
        self.flip = flip

    def __repr__(self):
        """
        Возвращает строковое представление углового события.

        Возвращает:
            str: Описание углового события.
        """
        return super().__repr__() + " [flip]%r" % self.flip


class model_event(base_event):
    """
    Класс model_event является базовым классом для всех событий, которые используют
    модель в расчетах.
    """

    def __init__(
        self, model, value=0, direction=0, terminal=True, accurate=True, count=-1
    ):
        """
        Инициализирует экземпляр класса model_event.

        Параметры:
            model (base_model): Модель, используемая в расчетах.
            value (float): Целевое значение функции события.
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(value, direction, terminal, accurate, count)
        self.model = model

    def __repr__(self):
        """
        Возвращает строковое представление события модели.

        Возвращает:
            str: Описание события модели.
        """
        return super().__repr__() + " [model]%r" % self.model


class eventT(base_event):
    coord = "t"

    def __call__(self, t, s):
        """
        Событие по времени: когда время достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: t - value
        """
        return t - self.value


class eventSinT(base_event):
    def __call__(self, t, s):
        """
        Событие, основанное на синусоиде времени.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: sin((t / value) * pi)
        """
        return math.sin((t / self.value) * math.pi)


class eventX(base_event):
    coord = "x"

    def __call__(self, t, s):
        """
        Событие по координате X: когда x достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: s[0] - value
        """
        return s[0] - self.value

    def to_code(self, i):
        """
        Генерирует строку кода для события по координате X.

        Параметры:
            i (int): Индекс события.

        Возвращает:
            str: Строка кода.
        """
        return "val%03d = x - (%.18f)" % (i, self.value)


class eventY(base_event):
    coord = "y"

    def __call__(self, t, s):
        """
        Событие по координате Y: когда y достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: s[1] - value
        """
        return s[1] - self.value

    def to_code(self, i):
        """
        Генерирует строку кода для события по координате Y.

        Параметры:
            i (int): Индекс события.

        Возвращает:
            str: Строка кода.
        """
        return "val%03d = y - (%.18f)" % (i, self.value)


class eventZ(base_event):
    coord = "z"

    def __call__(self, t, s):
        """
        Событие по координате Z: когда z достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: s[2] - value
        """
        return s[2] - self.value

    def to_code(self, i):
        """
        Генерирует строку кода для события по координате Z.

        Параметры:
            i (int): Индекс события.

        Возвращает:
            str: Строка кода.
        """
        return "val%03d = z - (%.18f)" % (i, self.value)


class eventVX(base_event):
    coord = "vx"

    def __call__(self, t, s):
        """
        Событие по скорости VX: когда vx достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: s[3] - value
        """
        return s[3] - self.value


class eventVY(base_event):
    coord = "vy"

    def __call__(self, t, s):
        """
        Событие по скорости VY: когда vy достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: s[4] - value
        """
        return s[4] - self.value


class eventVZ(base_event):
    coord = "vz"

    def __call__(self, t, s):
        """
        Событие по скорости VZ: когда vz достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: s[5] - value
        """
        return s[5] - self.value


class eventAX(model_event):
    coord = "ax"

    def __call__(self, t, s):
        """
        Событие по ускорению AX: когда ax достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: model.right_part(t, s, constants)[3] - value
        """
        return self.model.right_part(t, s, self.model.constants)[3] - self.value


class eventAY(model_event):
    coord = "ay"

    def __call__(self, t, s):
        """
        Событие по ускорению AY: когда ay достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: model.right_part(t, s, constants)[4] - value
        """
        return self.model.right_part(t, s, self.model.constants)[4] - self.value


class eventAZ(model_event):
    coord = "az"

    def __call__(self, t, s):
        """
        Событие по ускорению AZ: когда az достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: model.right_part(t, s, constants)[5] - value
        """
        return self.model.right_part(t, s, self.model.constants)[5] - self.value


class eventR(center_event):
    splits = 64

    def __call__(self, t, s):
        """
        Событие по радиусу R: когда радиус достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 - value^2
        """
        return (
            (s[0] - self.center[0]) ** 2
            + (s[1] - self.center[1]) ** 2
            + (s[2] - self.center[2]) ** 2
        ) - self.value**2

    def get_df(self):
        """
        Получает DataFrame для отображения сферы события.

        Возвращает:
            DataFrame: Таблица с координатами сферы.
        """
        alpha = np.linspace(0, 2 * np.pi, self.splits)
        c01 = self.center[0] + self.value * np.cos(alpha)
        c10 = self.center[1] + self.value * np.sin(alpha)
        c02 = self.center[0] + self.value * np.cos(alpha)
        c20 = self.center[2] + self.value * np.sin(alpha)
        c12 = self.center[1] + self.value * np.cos(alpha)
        c21 = self.center[2] + self.value * np.sin(alpha)
        z = np.zeros(self.splits, dtype=float)
        return pd.DataFrame(
            {
                "x": np.hstack((c01, c02, z)),
                "y": np.hstack((c10, z, c12)),
                "z": np.hstack((z, c20, c21)),
            }
        )

    def plot_df(self, df, ax, projection, **kwargs):
        """
        Отображает сферу события на графике.

        Параметры:
            df (DataFrame): Данные для отображения.
            ax (Axes): Ось matplotlib для рисования.
            projection (str): Тип проекции ('x-y', 'x-z', 'y-z').
            **kwargs: Дополнительные параметры для рисования.
        """
        p = projection.split("-")
        all_prj = ("x-y", "x-z", "y-z")
        for i, prj in enumerate(all_prj):
            if projection in (prj, prj[::-1]):
                s = slice(i * self.splits, (i + 1) * self.splits)
                ax.plot(df[p[0]][s].values, df[p[1]][s].values, **kwargs)
                break


class eventDR(center_event):
    def __call__(self, t, s):
        """
        Событие по производной радиуса по времени (DR): когда производная радиуса достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
        """
        v = s[3:]
        r = s[:3] - self.center
        r = r / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
        return r[0] * v[0] + r[1] * v[1] + r[2] * v[2]


class eventRdotV(center_event):
    def __call__(self, t, s):
        """
        Событие по произведению радиуса на скорость (RdotV): когда произведение радиуса и скорости достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: r · v - value
        """
        r = s[:3] - self.center
        v = s[3:6]
        return r[0] * v[0] + r[1] * v[1] + r[2] * v[2]

    def to_code(self, i):
        """
        Генерирует строку кода для события RdotV.

        Параметры:
            i (int): Индекс события.

        Возвращает:
            str: Строка кода.
        """
        return """val%03d = (x - %.18f)*vx + (y - %.18f)*vy + (z - %.18f)*vz""" % (
            i,
            *self.center,
        )


class eventAlphaX(center_angle_event):
    def __call__(self, t, s):
        """
        Событие по углу AlphaX: когда угол относительно оси X достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: angle - value
        """
        x, y = s[0] - self.center, s[1]
        if self.flip:
            x = -x
        angle = math.degrees(math.atan2(y, x))
        return angle - self.value


class eventOmegaX(center_event):
    def __call__(self, t, s):
        """
        Событие по угловой скорости OmegaX: когда угловая скорость относительно оси X достигает заданного значения.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: omega - value
        """
        v = s[3:5]
        r = (s[0] - self.center, s[1])
        omega = (r[0] * v[1] - r[1] * v[0]) / (r[0] ** 2 + r[1] ** 2)
        return omega - self.value


class eventHyperboloidX(center_event):
    """
    Класс eventHyperboloidX представляет собой гиперболоид, ориентированный вдоль оси X.
    Секция по XY — это гипербола, заданная неявным уравнением:
        x^2/a^2 - y^2/b^2 = 1
    """

    param_t = 1.0
    splits = 64

    def __init__(
        self,
        a,
        b,
        flip=False,
        center=0.0,
        value=0,
        direction=0,
        terminal=True,
        accurate=True,
        count=-1,
    ):
        """
        Инициализирует экземпляр класса eventHyperboloidX.

        Параметры:
            a (float): Параметр a гиперболы.
            b (float): Параметр b гиперболы.
            flip (bool): Отражать гиперболоид относительно оси Y.
            center (float): Центр гиперболоида по оси X.
            value (float): Целевое значение функции события.
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(center, 0, direction, terminal, accurate, count)
        self.flip = flip
        self.a = a
        self.b = b

    def __call__(self, t, s):
        """
        Вычисляет значение функции события для гиперболоида.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: x - a * sqrt(1 + (y / b)^2)
        """
        x, y = s[0] - self.center, (s[1] ** 2 + s[2] ** 2) ** 0.5
        if self.flip:
            x = -x
        return x - self.a * (1 + (y / self.b) ** 2) ** 0.5

    def get_df(self):
        """
        Получает DataFrame для отображения гиперболоида.

        Возвращает:
            DataFrame: Таблица с координатами гиперболоида.
        """
        pt = np.linspace(-self.param_t, self.param_t, self.splits + 1)
        x = self.a * np.cosh(pt) + self.center
        if self.flip:
            x = -x
        y = self.b * np.sinh(pt)
        return pd.DataFrame({"x": x, "y": y})

    def plot_df(self, df, ax, projection, **kwargs):
        """
        Отображает гиперболоид на графике.

        Параметры:
            df (DataFrame): Данные для отображения.
            ax (Axes): Ось matplotlib для рисования.
            projection (str): Тип проекции ('x-y', 'x-z').
            **kwargs: Дополнительные параметры для рисования.
        """
        # p = projection.split('-')
        # all_prj = ('x-y', 'x-z', 'y-z')
        all_prj = ("x-y", "x-z")
        for i, prj in enumerate(all_prj):
            if projection in (prj,):
                ax.plot(df["x"].values, df["y"].values, **kwargs)
                # if projection in (prj[::-1],):
                #     ax.plot(df[p[1]].values,
                #             df[p[0]].values, **kwargs)
                break


class eventParaboloidX(center_event):
    """
    Класс eventParaboloidX представляет собой параболоид, ориентированный вдоль оси X.
    Секция по XY — это парабола, заданная явным уравнением:
        x = y^2/a
    """

    splits = 64

    def __init__(
        self,
        a,
        flip=False,
        center=0.0,
        value=0,
        direction=0,
        terminal=True,
        accurate=True,
        count=-1,
    ):
        """
        Инициализирует экземпляр класса eventParaboloidX.

        Параметры:
            a (float): Параметр a параболы.
            flip (bool): Отражать параболоид относительно оси Y.
            center (float): Центр параболоида по оси X.
            value (float): Целевое значение функции события.
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(center, 0, direction, terminal, accurate, count)
        self.flip = flip
        self.a = a

    def __call__(self, t, s):
        """
        Вычисляет значение функции события для параболоида.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: x - y^2 / a
        """
        x, y = s[0] - self.center, (s[1] ** 2 + s[2] ** 2) ** 0.5
        if self.flip:
            x = -x
        return x - y**2 / self.a


class eventConeX(center_angle_event):
    def __call__(self, t, s):
        """
        Событие по углу конуса относительно оси X.

        Параметры:
            t (float): Текущее время.
            s (array): Текущее состояние системы.

        Возвращает:
            float: angle - value
        """
        x, y = s[0] - self.center, (s[1] ** 2 + s[2] ** 2) ** 0.5
        if self.flip:
            x = -x
        angle = math.degrees(math.atan2(y, x))
        return angle - self.value


class eventInsidePathXY(center_angle_event):
    """
    Класс eventInsidePathXY предназначен для обнаружения события, когда траектория пересекает 3D поверхность,
    образованную вращением плоского пути вокруг оси X.
    """

    def __init__(
        self,
        path,
        center,
        flip=False,
        splits=720,
        direction=0,
        terminal=True,
        accurate=True,
        count=-1,
    ):
        """
        Инициализирует экземпляр класса eventInsidePathXY.

        Параметры:
            path (np.ndarray): Путь в плоскости XY.
            center (array-like): Центр вращения.
            flip (bool): Отражать путь относительно оси X.
            splits (int): Количество разбиений для интерполяции.
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(center, flip, 0, direction, terminal, accurate, count)
        self.splits = splits
        self.set_path(path)

    def set_path(self, path):
        """
        Устанавливает путь для события.

        Параметры:
            path (np.ndarray): Путь в плоскости XY.
        """
        if not isinstance(path, np.ndarray) or path.ndim < 2:
            raise TypeError(
                "path должен быть numpy массивом с 2 измерениями\n%r" % path
            )
        self.path = path
        x = self.path[:, 0] - self.center
        if self.flip:
            x = -x
        y = self.path[:, 1]
        theta = np.arctan2(y, x)
        order = np.argsort(theta)
        theta = theta[order]
        r = x**2 + y**2
        r = r[order]
        # self.rint = interp1d(theta[:-1], r[:-1], fill_value='extrapolate', kind='cubic')
        self.rint = InterpolatedUnivariateSpline(theta[:-1], r[:-1])
        # theta равномерно распределен, строго возрастающий массив
        self.theta = np.linspace(-np.pi, np.pi, self.splits)
        self.r = self.rint(self.theta)

    def theta_r(self, t, s):
        """
        Вычисляет угол theta и радиус r для текущего состояния.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            tuple: (theta, r)
        """
        x, y, z = s[0] - self.center, s[1], s[2]
        if self.flip:
            x = -x
        r = x**2 + y**2 + z**2

        theta = math.atan2((y**2 + z**2) ** 0.5, x)

        return theta, r

    def interp(t, s, center, flip, x_arr, y_arr):
        """
        Быстрая линейная интерполяционная функция.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.
            center (array-like): Центр вращения.
            flip (bool): Отражать путь относительно оси X.
            x_arr (np.ndarray): Массив углов theta.
            y_arr (np.ndarray): Массив радиусов r.

        Возвращает:
            float: Разница между текущим радиусом и интерполированным радиусом.
        """
        # вычисляем r, theta
        x, y, z = s[0] - center, s[1], s[2]
        if flip:
            x = -x

        r = x**2 + y**2 + z**2

        theta = math.atan2((y**2 + z**2) ** 0.5, x)

        # линейная интерполяция
        i = types.int64((theta - x_arr[0]) // (x_arr[1] - x_arr[0]))
        r1 = y_arr[i] + (y_arr[i + 1] - y_arr[i]) / (x_arr[i + 1] - x_arr[i]) * (
            theta - x_arr[i]
        )
        return r - r1

    interp = njit(cache=True)(interp).compile("f8(f8,f8[:],f8,b1,f8[:],f8[:])")

    # interp = \
    # compiler.compile_isolated(interp,
    #                           [types.double, types.double[:], types.double,
    #                            types.boolean, types.double[:], types.double[:]],
    #                           return_type=types.double).entry_point

    def __call__(self, t, s):
        """
        Вычисляет значение функции события для пересечения пути.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Разница между текущим радиусом и интерполированным радиусом.
        """
        return eventInsidePathXY.interp(
            t, s, self.center, self.flip, self.theta, self.r
        )

    def deprecated__call__(self, t, s):
        """
        Устаревшая версия метода __call__.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Разница между текущим радиусом и интерполированным радиусом.
        """
        theta, r = self.theta_r(t, s)
        r1 = self.rint(theta)
        print(r, r1)
        return r - r1  # внутри пути (event < 0)

    def get_df(self):
        """
        Получает DataFrame для отображения пути.

        Возвращает:
            DataFrame: Таблица с координатами пути.
        """
        return pd.DataFrame({"x": self.path[:, 0], "y": self.path[:, 1]})

    def plot_df(self, df, ax, projection, **kwargs):
        """
        Отображает путь на графике.

        Параметры:
            df (DataFrame): Данные для отображения.
            ax (Axes): Ось matplotlib для рисования.
            projection (str): Тип проекции ('x-y', 'y-x').
            **kwargs: Дополнительные параметры для рисования.
        """
        p = projection.split("-")
        if projection in ("x-y", "y-x"):
            ax.plot(df[p[0]].values, df[p[1]].values, **kwargs)


class eventSplitLyapunov(center_angle_event):
    """
    Класс eventSplitLyapunov предназначен для обнаружения события, когда траектория пересекает 3D поверхность,
    образованную вращением плоской орбиты Ляпунова вокруг оси X.
    """

    def __init__(
        self,
        lyapunov_orbit_half,
        center,
        flip=False,
        # fname=pkg_resources.resource_filename(__name__, 'data/hlyapunov_sel1.csv'),
        # orbit_idx=-500,
        split_theta=1.8461392981282345,
        left=True,
        splits=720,
        direction=0,
        terminal=True,
        accurate=True,
        count=-1,
    ):
        """
        Инициализирует экземпляр класса eventSplitLyapunov.

        Параметры:
            lyapunov_orbit_half (np.ndarray): Половина орбиты Ляпунова.
            center (array-like): Центр вращения.
            flip (bool): Отражать орбиту относительно оси X.
            split_theta (float): Угол разделения.
            left (bool): Направление события.
            splits (int): Количество разбиений для интерполяции.
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(center, flip, 0, direction, terminal, accurate, count)
        self.splits = splits
        self.split_theta = split_theta
        self.left = left
        self.set_orbit(lyapunov_orbit_half)

    def set_orbit(self, orbit):
        """
        Устанавливает орбиту Ляпунова для события.

        Параметры:
            orbit (np.ndarray): Орбита Ляпунова.
        """
        if not isinstance(orbit, np.ndarray) or orbit.ndim < 2:
            raise TypeError(
                "orbit должен быть numpy массивом с 2 измерениями\n%r" % orbit
            )
        # orbit должен быть np.array, x->orbit[:,0], y->orbit[:,1]
        self.orbit = orbit.copy()
        mid_idx = self.orbit.shape[0] // 2
        if self.orbit[mid_idx, 1] < 0:  # отражаем орбиту
            self.orbit[:, 1] = -self.orbit[:, 1]
        x = self.orbit[:, 0] - self.center
        if self.flip:
            x = -x
        y = self.orbit[:, 1]
        theta = np.arctan2(y, x)
        order = np.argsort(theta)
        theta = theta[order]
        r = x**2 + y**2
        r = r[order]
        self.rint = InterpolatedUnivariateSpline(theta, r)
        # theta равномерно распределен, строго возрастающий массив
        self.theta = np.linspace(0.0, np.pi, self.splits)
        self.r = self.rint(self.theta)

    def theta_r(self, t, s):
        """
        Вычисляет угол theta и радиус r для текущего состояния.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            tuple: (theta, r)
        """
        x, y, z = s[0] - self.center, s[1], s[2]
        if self.flip:
            x = -x
        r = x**2 + y**2 + z**2

        theta = math.atan2((y**2 + z**2) ** 0.5, x)

        return theta, r

    def get_xy(self, theta):
        """
        Получает координаты (x, y) для заданного угла theta.

        Параметры:
            theta (float): Угол theta.

        Возвращает:
            np.ndarray: Координаты [x, y].
        """
        r = self.rint(theta) ** 0.5
        x = r * math.cos(theta)
        if self.flip:
            x = -x
        x += self.center
        y = r * math.sin(theta)
        return np.array([x, y])

    def interp(t, s, center, flip, x_arr, y_arr, theta_s, left):
        """
        Быстрая линейная интерполяционная функция.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.
            center (array-like): Центр вращения.
            flip (bool): Отражать путь относительно оси X.
            x_arr (np.ndarray): Массив углов theta.
            y_arr (np.ndarray): Массив радиусов r.
            theta_s (float): Угол разделения.
            left (bool): Направление события.

        Возвращает:
            float: Разница между текущим радиусом и интерполированным радиусом, или -1.0.
        """
        # вычисляем r, theta
        x, y, z = s[0] - center, s[1], s[2]
        if flip:
            x = -x

        theta_xy = math.atan2(math.fabs(y), x)
        # print(theta_xy)
        # print('theta_xy, theta_s', theta_xy, theta_s)
        if (not left and (theta_xy > theta_s)) or (left and (theta_xy < theta_s)):
            return -1.0

        theta = math.atan2((y**2 + z**2) ** 0.5, x)

        r = x**2 + y**2 + z**2

        # линейная интерполяция
        i = types.int64((theta - x_arr[0]) // (x_arr[1] - x_arr[0]))
        r1 = y_arr[i] + (y_arr[i + 1] - y_arr[i]) / (x_arr[i + 1] - x_arr[i]) * (
            theta - x_arr[i]
        )
        return r - r1

    interp = njit(cache=True)(interp).compile("f8(f8,f8[:],f8,b1,f8[:],f8[:],f8,b1)")

    # interp = \
    # compiler.compile_isolated(interp,
    #                           [types.double, types.double[:], types.double,
    #                            types.boolean, types.double[:], types.double[:],
    #                            types.double, types.boolean],
    #                           return_type=types.double).entry_point

    def __call__(self, t, s):
        """
        Вычисляет значение функции события для разбиения орбиты Ляпунова.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Разница между текущим радиусом и интерполированным радиусом или -1.0.
        """
        return eventSplitLyapunov.interp(
            t,
            s,
            self.center,
            self.flip,
            self.theta,
            self.r,
            self.split_theta,
            self.left,
        )

    def deprecated__call__(self, t, s):
        """
        Устаревшая версия метода __call__.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Разница между текущим радиусом и интерполированным радиусом.
        """
        theta, r = self.theta_r(t, s)
        theta_xy = math.atan2(math.fabs(s[1]), s[0])
        if (not self.left and (theta_xy > self.split_theta)) or (
            self.left and (theta_xy < self.split_theta)
        ):
            return -1
        r1 = self.rint(theta)
        print(r, r1)
        return r - r1  # внутри пути (event < 0)

    def get_df(self):
        """
        Получает DataFrame для отображения орбиты.

        Возвращает:
            DataFrame: Таблица с координатами орбиты.
        """
        return pd.DataFrame({"x": self.orbit[:, 0], "y": self.orbit[:, 1]})

    def plot_df(self, df, ax, projection, **kwargs):
        """
        Отображает орбиту на графике.

        Параметры:
            df (DataFrame): Данные для отображения.
            ax (Axes): Ось matplotlib для рисования.
            projection (str): Тип проекции ('x-y', 'y-x').
            **kwargs: Дополнительные параметры для рисования.
        """
        p = projection.split("-")
        if projection in ("x-y", "y-x"):
            ax.plot(df[p[0]].values, df[p[1]].values, **kwargs)


class eventSPL(model_event):
    _data = {}
    _compiled_find_index = None
    _compiled_interp = None

    def __init__(
        self,
        model,
        jc,
        point="L1",
        left=True,
        # value=0,
        direction=0,
        terminal=True,
        accurate=True,
        count=-1,
    ):
        """
        Инициализирует экземпляр класса eventSPL.

        Параметры:
            model (base_model): Модель, используемая в расчетах.
            jc (float): Константа Жакоби.
            point (str): Точка, относительно которой определяется событие (например, 'L1').
            left (bool): Направление события.
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(model, 0, direction, terminal, accurate, count)
        self.point = point
        self.left = left
        self.flip = point == "L1"
        self.center = model.__getattribute__(point)
        if eventSPL._compiled_find_index is None:
            eventSPL._compiled_find_index = njit(cache=True)(
                eventSPL.find_index
            ).compile("i8(f8[:],f8)")
            # eventSPL._compiled_find_index = \
            #     compiler.compile_isolated(eventSPL.find_index,
            #                               [types.double[:], types.double],
            #                               return_type=types.int64).entry_point
        if eventSPL._compiled_interp is None:
            eventSPL._compiled_interp = njit(cache=True)(eventSPL.interp).compile(
                "f8(f8,f8[:],f8,b1,f8[:],f8[:],f8,b1)"
            )

            # eventSPL._compiled_interp = \
            #     compiler.compile_isolated(eventSPL.interp,
            #                               [types.double, types.double[:], types.double,
            #                                types.boolean, types.double[:], types.double[:],
            #                                types.double, types.boolean],
            #                               return_type=types.double).entry_point
        self.load_data(jc)

    @staticmethod
    def find_index(arr, val):
        """
        Находит индекс первого элемента, меньшего val.

        Параметры:
            arr (np.ndarray): Массив чисел.
            val (float): Значение для сравнения.

        Возвращает:
            int: Индекс первого элемента, меньшего val, или -1 если такого нет.
        """
        for i in range(len(arr)):
            if arr[i] < val:
                return i
        return -1

    @staticmethod
    def find_index_old(arr, val):
        """
        Устаревшая версия функции find_index.

        Параметры:
            arr (np.ndarray): Массив чисел.
            val (float): Значение для сравнения.

        Возвращает:
            int: Индекс первого элемента, меньшего или равного val.
        """
        mask = arr <= val
        return np.argmax(mask)

    def set_jc(self, jc):
        """
        Устанавливает константу Жакоби и соответствующие параметры.

        Параметры:
            jc (float): Константа Жакоби.

        Raises:
            RuntimeError: Если подходящая орбита не найдена для заданной константы Жакоби.
        """
        sp = eventSPL._data[self.point]["SP"]
        pr = eventSPL._data[self.point]["PR"]
        idx = eventSPL._compiled_find_index(sp[:, 1], jc)
        if idx < 0:
            raise RuntimeError(
                "Не удалось найти подходящую орбиту для константы Жакоби %f" % jc
            )
        self.split_theta = sp[idx, 2]
        self.selected_jc = sp[idx, 1]
        self.r = pr[idx].copy()
        self.jc = jc

    def load_data(self, jc):
        """
        Загружает данные для события SPL.

        Параметры:
            jc (float): Константа Жакоби.
        """
        """
        HLY = Семейство горизонтальных орбит Ляпунова
        PR = полярное представление для HLY
        SP = точка разделения
        """
        if self.point in eventSPL._data:
            pr = eventSPL._data[self.point]["PR"]
            sp = eventSPL._data[self.point]["SP"]
        else:
            fname = "HLY_" + self.point + "_" + self.model.const_set
            sp_fname = "SP_" + fname + ".csv"
            pr_fname = "PR_" + fname + ".npy"
            sp_path = pkg_resources.resource_filename(
                __name__, "data/families/" + sp_fname
            )
            pr_path = pkg_resources.resource_filename(
                __name__, "data/families/" + pr_fname
            )
            sp = np.loadtxt(sp_path)
            pr = np.load(pr_path)
            eventSPL._data[self.point] = {"SP": sp, "PR": pr}

        self.set_jc(jc)
        self.theta = np.linspace(0.0, np.pi, 720)
        self.rint = InterpolatedUnivariateSpline(self.theta, self.r)

    def theta_r(self, t, s):
        """
        Вычисляет угол theta и радиус r для текущего состояния.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            tuple: (theta, r)
        """
        x, y, z = s[0] - self.center, s[1], s[2]
        if self.flip:
            x = -x
        r = x**2 + y**2 + z**2

        theta = math.atan2((y**2 + z**2) ** 0.5, x)

        return theta, r

    def get_xy(self, theta):
        """
        Получает координаты (x, y) для заданного угла theta.

        Параметры:
            theta (float): Угол theta.

        Возвращает:
            np.ndarray: Координаты [x, y].
        """
        r = self.rint(theta) ** 0.5
        x = r * math.cos(theta)
        if self.flip:
            x = -x
        x += self.center
        y = r * math.sin(theta)
        return np.array([x, y])

    def interp(t, s, center, flip, x_arr, y_arr, theta_s, left):
        """
        Быстрая линейная интерполяционная функция.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.
            center (array-like): Центр вращения.
            flip (bool): Отражать путь относительно оси X.
            x_arr (np.ndarray): Массив углов theta.
            y_arr (np.ndarray): Массив радиусов r.
            theta_s (float): Угол разделения.
            left (bool): Направление события.

        Возвращает:
            float: Разница между текущим радиусом и интерполированным радиусом, или -1.0.
        """
        # вычисляем r, theta
        x, y, z = s[0] - center, s[1], s[2]
        if flip:
            x = -x

        theta_xy = math.atan2(math.fabs(y), x)
        # print(theta_xy)
        # print('theta_xy, theta_s', theta_xy, theta_s)
        if (not left and (theta_xy > theta_s)) or (left and (theta_xy < theta_s)):
            return -1.0

        theta = math.atan2((y**2 + z**2) ** 0.5, x)

        r = x**2 + y**2 + z**2

        # линейная интерполяция
        i = types.int64((theta - x_arr[0]) // (x_arr[1] - x_arr[0]))
        r1 = y_arr[i] + (y_arr[i + 1] - y_arr[i]) / (x_arr[i + 1] - x_arr[i]) * (
            theta - x_arr[i]
        )
        return r - r1

    def __call__(self, t, s):
        """
        Вычисляет значение функции события для SPL.

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Разница между текущим радиусом и интерполированным радиусом или -1.0.
        """
        return eventSPL._compiled_interp(
            t,
            s,
            self.center,
            self.flip,
            self.theta,
            self.r,
            self.split_theta,
            self.left,
        )


class eventFOV(center_event):
    def __init__(
        self, orbit, center, r, direction=0, terminal=True, accurate=True, count=-1
    ):
        """
        Инициализирует экземпляр класса eventFOV.

        Параметры:
            orbit (np.ndarray): Орбита для построения конуса.
            center (array-like): Центр конуса.
            r (float): Радиус конуса.
            direction (int): Направление обнаружения события.
            terminal (bool): Является ли событие терминальным.
            accurate (bool): Требуется ли точное определение события.
            count (int): Счетчик событий.
        """
        super().__init__(center, r, direction, terminal, accurate, count)
        self.set_orbit(orbit)

    def set_orbit(self, orbit):
        """
        Устанавливает орбиту для конуса FOV.

        Параметры:
            orbit (np.ndarray): Орбита для построения конуса.
        """
        if not isinstance(orbit, np.ndarray) or orbit.ndim < 2 or orbit.shape[1] < 4:
            raise TypeError(
                "[orbit] должен быть numpy массивом формы (n, 4): (t, x, y, z)\n%r"
                % orbit
            )
        self.orbit = orbit
        self.oint = interp1d(
            self.orbit[:, 0],
            self.orbit[:, 1:4],
            axis=0,
            kind="cubic",
            fill_value="extrapolate",
        )

    def __call__(self, t, s):
        """
        Вычисляет значение функции события для поля зрения (FOV).

        Параметры:
            t (float): Время.
            s (array): Состояние системы.

        Возвращает:
            float: Разница между углом конуса и текущим углом.
        """
        cone = self.oint(t)[0]
        cone_c = self.center - cone
        cone_s = s[:3] - cone
        # print(cone_c, cone_s)
        # try:
        d_c = (cone_c[0] ** 2 + cone_c[1] ** 2 + cone_c[2] ** 2) ** 0.5
        d_s = (cone_s[0] ** 2 + cone_s[1] ** 2 + cone_s[2] ** 2) ** 0.5
        alpha_cone_c = math.atan2(self.value, d_c)
        alpha_cone_s = math.acos(np.dot(cone_c, cone_s) / (d_c * d_s))
        # except:
        #     pass
        return alpha_cone_c - alpha_cone_s
