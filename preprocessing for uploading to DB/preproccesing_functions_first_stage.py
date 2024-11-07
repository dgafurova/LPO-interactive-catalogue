import pandas as pd
import numpy as np
import orbipy as op

# Параметры интегратора
params = {"rtol": 1e-12, "atol": 1e-12, "nsteps": 100000, "max_step": np.inf}
integrator = op.dopri5_integrator(params=params)

# Создание модели CRTBP для системы Земля-Луна с использованием интегратора
model = op.crtbp3_model("Earth-Moon (default)", integrator=integrator)

# Создание объекта для визуализации орбит
plotter = op.plotter.from_model(model, length_units="Mm")

# Масштабировщик для модели
scaler = op.scaler.from_model(model)

# Создание модели с State Transition Matrix (STM)
stmmodel = op.crtbp3_model("Earth-Moon (default)", stm=True)


def get_ind(data, num):
    """
    Получить индексы ближайших значений к равномерно распределенным точкам.

    Параметры:
    data - массив данных
    num - количество точек

    Возвращает:
    indlist - список индексов ближайших значений
    """
    indlist = []
    ideal = np.linspace(data.min(), data.max(), num=num)
    for i in range(num):
        indlist.append((abs(data - ideal[i])).idxmin())
    return indlist


def get_stability_complex(orb, threshold=1e-2):
    """
    Определить стабильность орбиты на основе множителей Флоэда.

    Параметры:
    orb - DataFrame с данными орбиты
    threshold - пороговое значение для определения стабильности

    Добавляет в DataFrame колонки 'stability' и 'unitCirclePairs'.
    """
    # nums - число пар НЕ на единичной окружности
    nums = []
    if "l6" in orb.columns:
        mlt = orb[["l1", "l2", "l3", "l4", "l5", "l6"]]
        fmNum = 6
    else:
        mlt = orb[["l1", "l2", "l3", "l4", "l5"]]
        fmNum = 5

    for i in range(orb.shape[0]):
        num = 0
        for j in range(fmNum):
            if abs(mlt.iloc[i, j]) - 1.0 > threshold:
                num += 1
        nums.append(num)
    nums = np.array(nums)

    stability = np.array(nums == 0, dtype=int)
    orb["stability"] = stability

    # число пар множителей на единичной окружности
    orb["unitCirclePairs"] = 2 - nums


def get_stability(orb, threshold=1e-2):
    """
    Определить стабильность орбиты на основе множителей Флоэда с комплексными значениями.

    Параметры:
    orb - DataFrame с данными орбиты
    threshold - пороговое значение для определения стабильности

    Добавляет в DataFrame колонки 'stability' и 'unitCirclePairs'.
    """
    if "l6" in orb.columns:
        mlt = orb[
            [
                "l1_r",
                "l2_r",
                "l3_r",
                "l4_r",
                "l5_r",
                "l6_r",
                "l1_im",
                "l2_im",
                "l3_im",
                "l4_im",
                "l5_im",
                "l6_im",
            ]
        ]
        numFM = 6
    else:
        mlt = orb[
            [
                "l1_r",
                "l2_r",
                "l3_r",
                "l4_r",
                "l5_r",
                "l1_im",
                "l2_im",
                "l3_im",
                "l4_im",
                "l5_im",
            ]
        ]
        numFM = 5

    # nums - число пар НЕ на единичной окружности
    nums = []
    for i in range(orb.shape[0]):
        num = 0
        for j in range(numFM):
            if abs(mlt.iloc[i, j] + mlt.iloc[i, numFM + j] * 1j) - 1.0 > threshold:
                num += 1
        nums.append(num)

    nums = np.array(nums)
    stability = np.array(nums == 0, dtype=int)
    orb["stability"] = stability

    # число пар множителей на единичной окружности
    orb["unitCirclePairs"] = 2 - nums


def get_dists_planar(orb, LP):
    """
    Рассчитать расстояния от точек орбиты до точки Lagrange (LP).

    Параметры:
    orb - DataFrame с данными орбиты
    LP - координата точки Lagrange

    Добавляет в DataFrame колонку 'dists'.
    """
    orb["dists"] = orb.x - LP


def get_dists(orb):
    """
    Рассчитать накопленные расстояния между последовательными точками орбиты.

    Параметры:
    orb - DataFrame с данными орбиты

    Добавляет в DataFrame колонку 'dists'.
    """
    dists = [0]
    for i in range(1, orb.shape[0]):
        dist = np.linalg.norm(orb[["x", "z"]].iloc[i] - orb[["x", "z"]].iloc[i - 1])
        dists.append(dists[-1] + dist)

    dists = pd.DataFrame(dists, columns=["dists"])
    orb["dists"] = dists


def get_cj(orb):
    """
    Рассчитать постоянную Жакоби для каждой точки орбиты.

    Параметры:
    orb - DataFrame с данными орбиты

    Добавляет в DataFrame колонку 'cj'.
    """
    cjs = []
    for i in range(orb.shape[0]):
        y = model.get_zero_state()
        y[[0, 2, 4]] = np.real(orb.iloc[i, :3])
        cj = model.jacobi(y)
        cjs.append(cj)
    orb["cj"] = cjs


def get_cj_and_periods(orb, nCr):
    """
    Рассчитать постоянную Жакоби и периоды орбиты.

    Параметры:
    orb - DataFrame с данными орбиты
    nCr - количество пересечений с плоскостью y=0 для определения периода

    Добавляет в DataFrame колонки 't' и 'cj'.
    """
    det = op.event_detector(model, events=[op.eventY(count=nCr)])

    periods = []
    cjs = []
    for i in range(orb.shape[0]):
        y = model.get_zero_state()
        y[[0, 2, 4]] = np.real(orb.iloc[i, :3])
        _, evout = det.prop(y, 0.0, 20.0 * np.pi, ret_df=False, last_state="last")
        t = scaler(evout[-1, 3], "nd-d")
        cj = model.jacobi(y)
        periods.append(t)
        cjs.append(cj)

    orb["t"] = periods
    orb["cj"] = cjs
