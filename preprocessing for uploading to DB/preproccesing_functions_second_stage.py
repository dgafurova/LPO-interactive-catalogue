import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
import orbipy as op

# Параметры интегратора (DOPRI5)
params = {"rtol": 1e-10, "atol": 1e-10, "nsteps": 100000, "max_step": np.inf}
integrator = op.dopri5_integrator(params=params)

# Создание модели ЦКТП (Циркулярно-ограниченной трехтельной проблемы) для системы Земля-Луна
model = op.crtbp3_model("Earth-Moon (default)", integrator=integrator)

# Создание объекта для построения графиков орбит
plotter = op.plotter.from_model(model, length_units="Mm")

# Создание масштабировщика для преобразования единиц измерения
scaler = op.scaler.from_model(model)

# Создание модели с матрицей перехода состояний (STM)
stmmodel = op.crtbp3_model("Earth-Moon (default)", stm=True)


def df2complex(data):
    """
    Преобразование колонок множителей Флоэда из вещественных частей в комплексные числа.

    Параметры:
    ----------
    data : pandas.DataFrame
        Данные орбиты с колонками множителей Флоэда.
    """
    if "l6" in data.columns:
        fmNum = 6
    else:
        fmNum = 5
    for i in range(fmNum):
        # Преобразование каждой компоненты множителя Флоэда в комплексное число
        data[f"l{i+1}"] = data[f"l{i+1}"].apply(lambda x: complex(x))


def trajectory2SI(orb):
    """
    Преобразование траектории из безразмерных единиц (DU) в системы единиц SI.

    Параметры:
    ----------
    orb : pandas.DataFrame
        Траектория орбиты в безразмерных единицах.

    Возвращает:
    ----------
    pandas.DataFrame
        Траектория орбиты в системах единиц SI.
    """
    # Масштабирование координат из DU в километры
    orb.x = scaler(orb.x, "nd-km")
    orb.y = scaler(orb.y, "nd-km")
    orb.z = scaler(orb.z, "nd-km")

    # Масштабирование скоростей из DU/DU-км/с в км/с
    orb.vx = scaler(orb.vx, "nd/nd-km/s")
    orb.vy = scaler(orb.vy, "nd/nd-km/s")
    orb.vz = scaler(orb.vz, "nd/nd-km/s")

    # Масштабирование времени из DU-дней в дни
    orb.t = scaler(orb.t, "nd-d")
    return orb


def generate_one_trajectory(s, t):
    """
    Генерация одной траектории орбиты за заданный период.

    Параметры:
    ----------
    s : numpy.ndarray
        Начальное состояние орбиты в безразмерных единицах.
    t : float
        Полный период орбиты в безразмерных единицах.

    Возвращает:
    ----------
    pandas.DataFrame
        Траектория орбиты в системах единиц SI.
    """
    # Интегрирование орбиты вперед на половину периода
    arr1 = model.prop(s, 0, t / 2)
    # Интегрирование орбиты назад на половину периода
    arr2 = model.prop(s, 0, -t / 2)
    # Корректировка времени для обратной траектории
    arr2.t = arr1.t.iloc[-1] - arr2.t
    # Объединение двух траекторий в одну
    arr = pd.concat([arr1, arr2[::-1]]).reset_index(drop=True)
    return trajectory2SI(arr)


def compute_properties(
    orb1, fam_name, saveFiles=False, planarLyapunov=False, index_start=0, request=False
):
    """
    Вычисление и форматирование свойств орбит.

    Форматирует колонки в следующем порядке:
    + x (км), z (км), v (км/с), t (дни),
    + ax (км), ay (км), az (км),
    + dist_primary (км), dist_secondary (км),
    + dist_curve (км),
    + cj,
    + l1_r, l2_r, l3_r, l4_r, l5_r, l6_r, l1_im, l2_im, l3_im, l4_im, l5_im, l6_im,
    + stable (bool), stability_order (int)

    Если request=True, подготавливает данные для отправки как запрос.
    Иначе подготавливает данные для непосредственного добавления в базу данных.

    Параметры:
    ----------
    orb1 : pandas.DataFrame
        Данные орбиты с безразмерными состояниями и периодом в днях.
    fam_name : str
        Тег семейства орбит.
    saveFiles : bool, optional
        Если True, сохраняет файлы в формате CSV.
    planarLyapunov : bool, optional
        Если True, добавляет колонку 'alpha'.
    index_start : int, optional
        Начальный индекс для идентификации орбит.
    request : bool, optional
        Если True, подготавливает данные для запроса.

    Возвращает:
    ----------
    pandas.DataFrame
        Форматированные данные орбиты.
    """
    orb = orb1.copy()
    orbColumns = orb.columns
    # Обработка множителей Флоэда
    if "l6_r" not in orbColumns:
        if "l5_r" in orbColumns:
            # Добавление столбцов l6_r и l6_im со значениями по умолчанию
            orb["l6_r"] = [1] * orb.shape[0]
            orb["l6_im"] = [0] * orb.shape[0]
        elif "l1" in orbColumns:
            if "l6" not in orbColumns:
                orb["l6"] = [1 + 0j] * orb.shape[0]

            # Создание DataFrame для вещественных частей множителей Флоэда
            lr = pd.DataFrame(
                list(
                    [
                        np.real(orb.l1),
                        np.real(orb.l2),
                        np.real(orb.l3),
                        np.real(orb.l4),
                        np.real(orb.l5),
                        np.real(orb.l6),
                    ]
                ),
                index=["l1_r", "l2_r", "l3_r", "l4_r", "l5_r", "l6_r"],
            ).T
            # Создание DataFrame для мнимых частей множителей Флоэда
            lim = pd.DataFrame(
                list(
                    [
                        np.imag(orb.l1),
                        np.imag(orb.l2),
                        np.imag(orb.l3),
                        np.imag(orb.l4),
                        np.imag(orb.l5),
                        np.imag(orb.l6),
                    ]
                ),
                index=["l1_im", "l2_im", "l3_im", "l4_im", "l5_im", "l6_im"],
            ).T
            # Объединение исходного DataFrame с новыми столбцами
            orb = pd.concat([orb, lr, lim], axis=1)

    # Проверка наличия информации о стабильности
    if "unitCirclePairs" not in orbColumns:
        print("I have no information on stability, it is sad")

    # Расчет свойств орбит
    ax = []
    ay = []
    az = []
    distPrim = []
    distSec = []
    distCurve = [0]
    stable = []
    stabilityOrder = []
    arr_all = []
    coordinates = []

    cjNotIn = "cj" not in orbColumns
    if cjNotIn:
        cj = []

    for i in range(orb.shape[0]):
        # Получение начального состояния орбиты
        s = model.get_zero_state()
        s[[0, 2, 4]] = np.real(orb.iloc[i][["x", "z", "v"]])
        # Масштабирование периода
        t = np.real(scaler(orb.iloc[i]["t"], "d-nd"))
        # Генерация траектории орбиты
        arr = generate_one_trajectory(s, t)
        if saveFiles:
            # Сохранение траектории в формате JSON
            arr_all.append(arr.to_json(orient="records"))
            # Сохранение координат начальной точки траектории
            coordinates.append(arr.iloc[0, 1:])
        #         plotter.plot_proj(arr)

        # Вычисление разницы максимумов и минимумов координат для определения размеров орбиты
        ax.append(arr.x.max() - arr.x.min())
        ay.append(arr.y.max() - arr.y.min())
        az.append(arr.z.max() - arr.z.min())

        # Вычисление минимальных расстояний до первичной и вторичной точек
        distPrim.append(
            min(
                np.linalg.norm(
                    arr.iloc[:, 1:4] - [-scaler(model.mu, "nd-km"), 0, 0], axis=1
                )
            )
        )
        distSec.append(
            min(
                np.linalg.norm(
                    arr.iloc[:, 1:4] - [scaler(model.mu1, "nd-km"), 0, 0], axis=1
                )
            )
        )

        if i > 0:
            # Накопленное расстояние по траектории
            distCurve.append(
                distCurve[-1] + np.linalg.norm(s[[0, 2]] - orb.iloc[i - 1][["x", "z"]])
            )
        if cjNotIn:
            # Вычисление постоянной Жакоби
            cj.append(model.jacobi(s))
        # Определение стабильности орбиты
        stable.append(True if orb.unitCirclePairs[i] == 2 else False)
        stabilityOrder.append(orb.unitCirclePairs[i])

    if planarLyapunov:
        # Добавление колонки 'alpha' для планарных направлений Ляпунова
        orb["alpha"] = [0] * orb.shape[0]

    # Добавление рассчитанных свойств в DataFrame
    orb["ax"] = ax
    orb["ay"] = ay
    orb["az"] = az
    orb["dist_primary"] = distPrim
    orb["dist_secondary"] = distSec

    orb["dist_curve"] = distCurve
    # Масштабирование координат и скоростей в системе единиц SI
    orb.x = scaler(orb.x, "nd-km")
    orb.z = scaler(orb.z, "nd-km")
    orb.v = scaler(orb.v, "nd/nd-km/s")

    if cjNotIn:
        # Добавление колонки 'cj' в DataFrame
        orb["cj"] = cj

    orb["stable"] = stable
    orb["stability_order"] = stabilityOrder
    orb["t_period"] = orb["t"]

    if request:
        # Выбор определенных колонок для запроса
        orb = orb[
            [
                "x",
                "z",
                "v",
                "alpha",
                "t_period",
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
                "ax",
                "ay",
                "az",
                "dist_primary",
                "dist_secondary",
                "dist_curve",
                "cj",
                "stable",
                "stability_order",
            ]
        ]
    else:
        # Добавление идентификаторов орбит
        orb["id"] = pd.DataFrame(range(index_start, index_start + orb.shape[0]))

        # Выбор определенных колонок для базы данных
        orb = orb[
            [
                "id",
                "x",
                "z",
                "v",
                "alpha",
                "t",
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
                "ax",
                "ay",
                "az",
                "dist_primary",
                "dist_secondary",
                "dist_curve",
                "cj",
                "stable",
                "stability_order",
            ]
        ]

    if saveFiles:
        # Сохранение данных орбиты в CSV файл
        file_name = f"formatted files\\{fam_name}.csv"
        orb.to_csv(file_name, index=False)
    return orb


def compute_trajectories(orb, fam_name, saveFiles=False, index_start=0, request=False):
    """
    Вычисление траекторий орбит и их сохранение.

    Параметры:
    ----------
    orb : pandas.DataFrame
        Данные орбиты с безразмерными состояниями и периодом в днях.
    fam_name : str
        Тег семейства орбит.
    saveFiles : bool, optional
        Если True, сохраняет траектории в формате CSV.
    index_start : int, optional
        Начальный индекс для идентификации орбит.
    request : bool, optional
        Если True, подготавливает данные для запроса.

    Возвращает:
    ----------
    pandas.DataFrame
        Данные траекторий орбит.
    """
    arr_all = []
    coordinates = []

    for i in range(orb.shape[0]):
        # Получение начального состояния орбиты
        s = model.get_zero_state()
        s[[0, 2, 4]] = np.real(orb.iloc[i][["x", "z", "v"]])
        # Масштабирование периода
        t = np.real(scaler(orb.iloc[i]["t"], "d-nd"))
        # Генерация траектории орбиты
        arr = generate_one_trajectory(s, t)
        if request:
            # Добавление идентификатора орбиты
            orbit_id = pd.DataFrame(
                [index_start + i] * arr.shape[0], columns=["orbit_id"]
            )
            arr_all.append(pd.concat([orbit_id, arr], axis=1))
        else:
            # Сохранение траектории в формате JSON
            arr_all.append(arr.to_json(orient="records"))
            # Сохранение координат начальной точки траектории
            coordinates.append(
                [
                    scaler(s[0], "nd-km"),
                    scaler(s[1], "nd-km"),
                    scaler(s[2], "nd-km"),
                    scaler(s[3], "nd/nd-km/s"),
                    scaler(s[4], "nd/nd-km/s"),
                    scaler(s[5], "nd/nd-km/s"),
                ]
            )

    if request:
        # Объединение всех траекторий в один DataFrame
        arr_all = pd.concat(arr_all).reset_index(drop=True)

    else:
        # Создание идентификаторов орбит
        orb_id = pd.DataFrame(
            range(index_start, index_start + orb.shape[0]), columns=["orbit_id"]
        )
        # Создание DataFrame для траекторий
        arr_all = pd.DataFrame(np.array(arr_all, dtype=list), columns=["v"])

        # Создание DataFrame для координат
        coordinates = pd.DataFrame(
            coordinates, columns=["x", "y", "z", "vx", "vy", "vz"]
        )
        # Объединение идентификаторов, траекторий и координат
        arr_all = pd.concat([orb_id, arr_all["v"], coordinates], axis=1)

    if saveFiles:
        # Сохранение траекторий орбит в CSV файл
        file_name_traj = f"formatted files//{fam_name}_trajectories.csv"
        arr_all.to_csv(file_name_traj, index=False)

    return arr_all


def compute_poincare_sections(
    orb, fam_name, saveFiles=False, index_start=0, request=False
):
    """
    Вычисление сечений Пуанкаре для орбит и их сохранение.

    Параметры:
    ----------
    orb : pandas.DataFrame
        Данные орбиты с безразмерными состояниями и периодом в днях.
    fam_name : str
        Тег семейства орбит.
    saveFiles : bool, optional
        Если True, сохраняет сечения Пуанкаре в формате CSV.
    index_start : int, optional
        Начальный индекс для идентификации орбит.
    request : bool, optional
        Если True, подготавливает данные для запроса.

    Возвращает:
    ----------
    pandas.DataFrame
        Данные сечений Пуанкаре орбит.
    """
    # Создание детектора событий для различных плоскостей
    det = op.event_detector(
        model,
        events=[
            op.eventX(terminal=False),
            op.eventY(terminal=False),
            op.eventZ(terminal=False),
            op.eventVX(terminal=False),
            op.eventVY(terminal=False),
            op.eventVZ(terminal=False),
        ],
    )

    # Определение названий плоскостей сечений
    planes = ["x = 0", "y = 0", "z = 0", "vx = 0", "vy = 0", "vz = 0"]
    sectionList = []
    planeList = []
    orbit_id = []

    for i in range(orb.shape[0]):
        # Получение начального состояния орбиты
        s = model.get_zero_state()
        s[[0, 2, 4]] = orb[["x", "z", "v"]].iloc[i]
        # Масштабирование периода
        period = scaler(orb["t"].iloc[i], "d-nd")
        # Интегрирование орбиты на заданный период
        _, ev = det.prop(s, 0, period + 1e-10)

        # Масштабирование результатов интегрирования
        ev["t"] = scaler(ev["t"], "nd-d")
        ev["x"] = scaler(ev["x"], "nd-km")
        ev["y"] = scaler(ev["y"], "nd-km")
        ev["z"] = scaler(ev["z"], "nd-km")
        ev["vx"] = scaler(ev["vx"], "nd/nd-km/s")
        ev["vy"] = scaler(ev["vy"], "nd/nd-km/s")
        ev["vz"] = scaler(ev["vz"], "nd/nd-km/s")

        # Поиск точек пересечения орбиты с плоскостями
        for j in range(6):
            section = ev[ev.e == j].copy()
            if len(section) > 0:
                # Обнуление соответствующей координаты при пересечении
                section.iloc[:, j + 1] = [0] * section.shape[0]
                plane = planes[j]
                if request:
                    # Подготовка данных для запроса
                    section = section.iloc[:, 3:]
                    for k in range(section.shape[0]):
                        planeList.append(plane)
                        orbit_id.append(i + index_start)
                else:
                    # Преобразование данных в формат JSON
                    section = section.iloc[:, 3:].to_json(orient="records")
                    planeList.append(plane)
                    orbit_id.append(i + index_start)

                sectionList.append(section)

    # Создание DataFrame для плоскостей и идентификаторов орбит
    planeList = pd.DataFrame(planeList, columns=["plane"])
    orbit_id = pd.DataFrame(orbit_id, columns=["orbit_id"])

    if request:
        # Объединение всех сечений в один DataFrame
        sectionList = pd.concat(sectionList).reset_index(drop=True)
        result = pd.concat([orbit_id, planeList, sectionList], axis=1)

    else:
        # Создание DataFrame для сечений Пуанкаре
        sectionList = pd.DataFrame(sectionList, columns=["v"])
        result = pd.concat([orbit_id, planeList, sectionList], axis=1)

    if saveFiles:
        # Сохранение сечений Пуанкаре в CSV файл
        file_name_traj = f"formatted files//{fam_name}_poincare_sections.csv"
        result.to_csv(file_name_traj, index=False)

    return result
