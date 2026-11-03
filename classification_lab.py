"""Classification Lab — учебный симулятор задачи классификации.

Девятый проект серии образовательных симуляторов:
    1. One Artificial Neuron   2. Perceptron           3. Mini Neural Network
    4. Gradient Descent        5. Backpropagation      6. Neural Network Training Lab
    7. Activation Functions    8. Loss Functions Lab   9. Classification Lab

Приложение показывает всю цепочку классификации целиком:
    признаки -> числовая оценка z -> граница решения -> предсказанный класс
    -> сравнение с истинным классом -> ошибки и accuracy.

Модель: линейный бинарный классификатор
    z  = w1*x1 + w2*x2 + b
    правило:  z >= 0  -> класс 1,   z < 0  -> класс 0
    граница:  w1*x1 + w2*x2 + b = 0

Граница, зоны, предсказания, матрица ошибок и accuracy считаются по одним и
тем же w1, w2 и b. Никаких декоративных линий и никаких заранее заготовленных
ответов.

Только стандартная библиотека: Python 3, tkinter, tkinter.ttk, math.
Внешние пакеты (NumPy, matplotlib и другие) не используются.

Автор: SteveWorks001

Куда смотреть
-------------
Раздел 1   Области ввода, палитра, обозначения
Раздел 2   Датасеты: точки и их истинные классы
Раздел 3   Модель: оценка, правило, граница, метрики
Раздел 4   Форматирование чисел и разбор ввода
Раздел 5   Локализация: весь текст живёт в TEXTS (русский по умолчанию)
Раздел 6   Учебные эксперименты
Раздел 7   Справка: заголовки и тексты разделов
Раздел 8   Виджеты: слайдер с полем ввода и плоскость классификации
Раздел 9   Главное окно
Раздел 10  Точка входа
"""

import math
import tkinter as tk
from tkinter import ttk, font as tkfont

# ----------------------------------------------------------------------
# Раздел 1. Области ввода, палитра, обозначения
# ----------------------------------------------------------------------

WORLD_MIN, WORLD_MAX = -4.0, 4.0            # все точки данных лежат здесь
PAD_LEFT, PAD_TOP, PAD_RIGHT, PAD_BOTTOM = 54, 16, 16, 42

W_MIN, W_MAX = -3.0, 3.0                    # область весов
B_MIN, B_MAX = -4.0, 4.0                     # область смещения
NUDGE = 0.1                                  # шаг кнопок +/-
SLIDER_STEP = 0.05                           # шаг слайдера

DEFAULT_W1, DEFAULT_W2, DEFAULT_B = 1.0, 1.0, 0.0
DEFAULT_DATASET = "separable"

MARGIN = 1.0                                 # уровень изолиний z = +/-1
TEXT_WIDTH = 38                              # ширина панелей текста, символы

# Обе таблицы показывают одни и те же столбцы, поэтому ширина задаётся один
# раз и для таблицы точек, и для списка ошибок.
TABLE_KEYS = ("n", "x1", "x2", "true", "z", "pred", "ok")
TABLE_WIDTHS = (30, 44, 44, 80, 52, 72, 62)

# Истинный класс задаётся формой маркера, предсказанный — какой зоной он
# накрыт. Красное кольцо — ошибка. Та же логика, что и в Perceptron.
COL_CLASS0 = "#2f6f9f"       # синий — класс 0
COL_CLASS1 = "#c8651b"       # жёлто-красный — класс 1
COL_REGION0 = "#e8f1f7"      # зона, где предсказывается класс 0
COL_REGION1 = "#fbeee0"      # зона, где предсказывается класс 1
COL_BOUNDARY = "#1b3a55"     # граница решения (z = 0)
COL_MARGIN = "#8fa8bd"       # изолинии z = +/-1
COL_WRONG = "#b3261e"        # кольцо вокруг ошибочной точки
COL_SELECTED = "#5b3a86"     # рамка выбранной точки
COL_GRID = "#dfe3e8"
COL_AXIS = "#4a4a4a"
COL_TEXT = "#242424"
COL_BG = "#fbfbf7"
COL_PAPER = "#fdfdf9"
COL_FRAME = "#c8c8c0"
COL_DIM = "#8a8a82"
COL_HEAD = "#31566f"
COL_KEY = "#b07d10"
COL_BAD_BG = "#fde8e6"
COL_BAD_FG = "#7a1a13"

SUBS = "₀₁₂₃₄₅₆₇₈₉"


def sub(number):
    """x -> x с цифрой в виде нижнего индекса (для x₁, w₂ и так далее)."""
    return "".join(SUBS[int(c)] if c.isdigit() else c for c in str(number))


# ----------------------------------------------------------------------
# Раздел 2. Датасеты
# ----------------------------------------------------------------------
# Каждая точка — это (x1, x2, истинный_класс). Координаты выбраны вручную:
# набор не меняется между запусками, и каждую точку видно отдельно.
# tuned — параметры, подобранные для этого набора руками; их показывает
# кнопка «параметры для набора» и учебные эксперименты. Подбора машинным
# обучением здесь нет.

DATASETS = (
    {
        "key": "separable",
        "tuned": (1.0, 1.0, 0.0),
        "points": (
            (1.0, 1.0, 1), (2.0, 1.0, 1), (1.0, 2.0, 1), (2.0, 2.0, 1),
            (0.5, 2.0, 1), (1.5, 1.5, 1), (3.0, 0.5, 1),
            (-1.0, -1.0, 0), (-2.0, -1.0, 0), (-1.0, -2.0, 0),
            (-2.0, -2.0, 0), (-0.5, -2.0, 0), (-1.5, -1.5, 0),
            (-3.0, -0.5, 0),
        ),
    },
    {
        "key": "overlap",
        "tuned": (1.0, -0.3, 0.8),
        "points": (
            (1.5, 1.0, 1), (2.0, 0.0, 1), (1.0, -1.5, 1), (2.5, 1.5, 1),
            (0.5, -0.5, 1), (1.5, -2.5, 1), (-0.5, 1.0, 1), (-1.0, -1.0, 1),
            (-1.5, 0.5, 0), (-2.0, -0.5, 0), (-1.0, 1.5, 0), (-2.5, -1.0, 0),
            (-1.5, -2.0, 0), (0.5, 1.5, 0), (1.0, -0.5, 0),
        ),
    },
    {
        "key": "xor",
        "tuned": (-1.0, 1.0, 0.0),
        "points": (
            (1.0, 1.0, 1), (-1.0, -1.0, 1), (-1.0, 1.0, 0), (1.0, -1.0, 0),
            (1.5, 1.5, 1), (-1.5, -1.5, 1), (-1.5, 1.5, 0), (1.5, -1.5, 0),
        ),
    },
    {
        "key": "ring",
        "tuned": (2.0, 0.8, -2.0),
        "points": (
            (0.0, 0.0, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0), (0.5, -0.5, 0),
            (-0.5, -0.5, 0),
            (2.5, 0.0, 1), (-2.5, 0.0, 1), (0.0, 2.5, 1), (0.0, -2.5, 1),
            (1.8, 1.8, 1), (-1.8, 1.8, 1), (1.8, -1.8, 1), (-1.8, -1.8, 1),
        ),
    },
    {
        "key": "stripes",
        "tuned": (-1.0, -2.0, -3.0),
        "points": (
            (-3.0, 2.0, 0), (-2.5, 0.0, 0), (-3.0, -2.0, 0), (-2.0, 1.5, 0),
            (0.0, 2.0, 1), (0.3, 0.0, 1), (-0.3, -1.5, 1), (0.2, -2.5, 1),
            (3.0, 2.0, 0), (2.5, 0.0, 0), (3.0, -2.0, 0), (2.0, 1.5, 0),
        ),
    },
)

DATASET_BY_KEY = {entry["key"]: entry for entry in DATASETS}


# ----------------------------------------------------------------------
# Раздел 3. Модель
# ----------------------------------------------------------------------

def score(w1, w2, b, x1, x2):
    """Числовая оценка классификатора: z = w1*x1 + w2*x2 + b."""
    return w1 * x1 + w2 * x2 + b


def predict(z):
    """Правило решения: z >= 0 -> класс 1, иначе класс 0."""
    return 1 if z >= 0 else 0


def has_boundary(w1, w2):
    """Есть ли у модели граница: она есть, только хотя бы один вес не ноль."""
    return not (w1 == 0.0 and w2 == 0.0)


def classify(point, w1, w2, b):
    """Полный ответ по одной точке: оценка, класс, верно ли."""
    x1, x2, true_class = point
    z = score(w1, w2, b, x1, x2)
    predicted = predict(z)
    return {
        "x1": x1, "x2": x2,
        "true": true_class,
        "z": z,
        "pred": predicted,
        "correct": predicted == true_class,
    }


def evaluate(points, w1, w2, b):
    """Предсказания по всему набору: верно/неверно, accuracy и матрица ошибок."""
    rows = [classify(point, w1, w2, b) for point in points]
    correct = sum(1 for row in rows if row["correct"])
    total = len(rows)
    accuracy = correct / total if total else 0.0
    matrix = {
        "tp": sum(1 for r in rows if r["true"] == 1 and r["pred"] == 1),
        "fn": sum(1 for r in rows if r["true"] == 1 and r["pred"] == 0),
        "fp": sum(1 for r in rows if r["true"] == 0 and r["pred"] == 1),
        "tn": sum(1 for r in rows if r["true"] == 0 and r["pred"] == 0),
    }
    return {"rows": rows, "correct": correct, "wrong": total - correct,
            "total": total, "accuracy": accuracy, "matrix": matrix}


def norm_w(w1, w2):
    """Длина вектора весов: |w| = sqrt(w1^2 + w2^2)."""
    return math.hypot(w1, w2)


def margin_distance(z, w1, w2):
    """Расстояние от точки до границы: |z| / |w|. При |w| = 0 не определено."""
    length = norm_w(w1, w2)
    if length == 0.0:
        return None
    return abs(z) / length


def boundary_equation_text(w1, w2, b):
    """Уравнение границы z = 0, разрешённое относительно x2."""
    if not has_boundary(w1, w2):
        return None
    if w2 != 0.0:
        return "x{q} = -({a}·x{p} {o} {v}) / {e}".format(
            q=sub(2), a=fmt(w1), p=sub(1), o=signed_op(b), v=fmt(abs(b)),
            e=fmt(w2))
    return "x{p} = {v}".format(p=sub(1), v=fmt(-b / w1))


def segment_on_plane(w1, w2, b, xmin, xmax, ymin, ymax):
    """Отрезок границы z = 0 внутри видимой части плоскости.

    Прямая ищется пересечением с краями окна, поэтому линия всегда именно
    там, где её задают w1, w2 и b, и никогда не выходит за рамки.
    """
    if not has_boundary(w1, w2):
        return None
    found = []
    if w2 != 0.0:
        for x in (xmin, xmax):
            y = -(w1 * x + b) / w2
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                found.append((x, y))
    if w1 != 0.0:
        for y in (ymin, ymax):
            x = -(w2 * y + b) / w1
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                found.append((x, y))
    points = []
    for point in found:
        # углы окна попадают в список дважды: сравниваем с допуском, но сами
        # координаты не округляем, чтобы z на границе оставался нулём
        duplicate = False
        for other in points:
            if abs(point[0] - other[0]) < 1e-9 and abs(point[1] - other[1]) < 1e-9:
                duplicate = True
                break
        if not duplicate:
            points.append(point)
    if len(points) < 2:
        return None
    points.sort()
    return points[0], points[-1]


def clip_half_plane(polygon, w1, w2, b, keep_positive):
    """Отсекает многоугольник полуплоскостью z >= 0 (или z < 0).

    Нужен, чтобы закрасить ровно ту часть плоскости, которой классификатор
    ставит класс 1 (или 0). Значение z считается теми же w1, w2 и b.
    """
    def value(point):
        z = score(w1, w2, b, point[0], point[1])
        return z if keep_positive else -z

    output = []
    previous = polygon[-1]
    previous_value = value(previous)
    for current in polygon:
        current_value = value(current)
        if current_value >= 0:
            if previous_value < 0:
                output.append(interpolate(previous, current, value))
            output.append(current)
        elif previous_value >= 0:
            output.append(interpolate(previous, current, value))
        previous, previous_value = current, current_value
    return [point for point in output if point is not None]


def interpolate(start, end, value):
    """Точка пересечения ребра start->end с линией, где value = 0."""
    start_value = value(start)
    end_value = value(end)
    if start_value == end_value:
        return None
    ratio = start_value / (start_value - end_value)
    return (start[0] + ratio * (end[0] - start[0]),
            start[1] + ratio * (end[1] - start[1]))


# ----------------------------------------------------------------------
# Раздел 4. Форматирование чисел и разбор ввода
# ----------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def fmt(value, digits=2):
    """Короткая запись числа: без хвостовых нулей и без -0.00."""
    if value is None:
        return "—"
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text


def fmt_score(value, digits=3):
    """Оценка z: всегда с десятичным знаком, чтобы не читалась как класс."""
    if value is None:
        return "—"
    text = f"{value:.{digits}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def fmt_percent(value, digits=1):
    return f"{value * 100:.{digits}f}%"


def signed_op(value):
    """Знак слагаемого для читаемой подстановки: + или -."""
    return "+" if value >= 0 else "-"


def parse_number(text):
    """Разбирает ввод: принимает и точку, и запятую. Иначе — None."""
    cleaned = (text or "").strip().replace(",", ".")
    if cleaned == "":
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def substitute_sum(w1, w2, b, x1, x2):
    """Подстановка чисел в оценку: одна строка с уже подставленными числами.

    Знак выносится перед слагаемым, поэтому строка не превращается в «+ -1».
    """
    def term(weight, feature):
        return fmt(abs(weight)) + "·" + fmt(feature)

    head = ("-" if w1 < 0 else "") + term(w1, x1)
    return "{h} {o1} {t2} {o2} {v}".format(
        h=head, o1=signed_op(w2), t2=term(w2, x2), o2=signed_op(b),
        v=fmt(abs(b)))


# ----------------------------------------------------------------------
# Раздел 5. Локализация
# ----------------------------------------------------------------------

LANG = "ru"


def set_language(code):
    global LANG
    LANG = code


def get_language():
    return LANG


# Весь переводимый текст живёт здесь. Формулы нейтральны по языку и вынесены
# в виджет плоскости: числа и символы одинаковы в обоих языках.

TEXTS = {
    "ru": {
        "app_title": "Classification Lab — как классификатор делит плоскость",
        "subtitle": ("Признаки x₁, x₂ → оценка z = w₁x₁ + w₂x₂ + b → граница "
                     "решения → предсказанный класс → сравнение с истинным"),
        # Левая колонка
        "box_dataset": "Датасет",
        "btn_tuned": "Подобрать параметры под набор",
        "tuned_hint": "Параметры набраны вручную: машина ничего не обучает.",
        "box_params": "Параметры классификатора",
        "lbl_w1": "Вес w₁",
        "lbl_w2": "Вес w₂",
        "lbl_b": "Смещение b",
        "hint_step": "Слайдер — шаг 0.05, кнопки — шаг 0.1. Запятая и точка "
                     "оба принимаются.",
        "btn_reset": "Сбросить всё",
        "box_rule": "Оценка и правило",
        "rule_formula": "z = w₁x₁ + w₂x₂ + b",
        "rule_text": "Если z ≥ 0 → класс 1. Если z < 0 → класс 0. "
                     "Точка ровно на границе (z = 0) получает класс 1.",
        "box_boundary": "Граница решения",
        "bd_formula": "Граница — это множество точек с z = 0:",
        "bd_equation": "Уравнение: {eq}",
        "bd_normal": "Нормаль границы — вектор весов (w₁, w₂) = ({a}, {b}). "
                     "Он перпендикулярен прямой.",
        "bd_degenerate": "w₁ = 0 и w₂ = 0: границей служить может что угодно, "
                         "сама прямая не определена. Оценка одинакова во всех "
                         "точках и равна b = {b}, поэтому вся плоскость "
                         "получает один класс.",
        "box_watch": "Что произошло",
        "box_language": "Язык интерфейса",
        "lang_ru": "Русский",
        "lang_en": "English",
        # Обозначения на плоскости
        "box_legend": "Обозначения",
        "legend_true0": "Истинный класс 0 — квадрат",
        "legend_true1": "Истинный класс 1 — кружок",
        "legend_region0": "Зона, где классификатор ставит 0",
        "legend_region1": "Зона, где классификатор ставит 1",
        "legend_boundary": "Граница решения (z = 0)",
        "legend_margin": "Изолинии z = +1 и z = −1",
        "legend_wrong": "Красное кольцо — точка классифицирована неверно",
        "legend_selected": "Пунктирная рамка — выбранная точка",
        # Вкладки
        "tab_point": "Точка",
        "tab_dataset": "Датасет",
        "tab_exp": "Эксперименты",
        "tab_help": "Справка",
        # Вкладка «Точка»
        "pt_none": "Кликните любую точку на плоскости — здесь появится её "
                   "полный расчёт.",
        "pt_box_point": "Выбранная точка",
        "pt_box_calc": "Расчёт оценки по шагам",
        "pt_box_verdict": "Итог по точке",
        "pt_x1": "Признак x₁ = {v}",
        "pt_x2": "Признак x₂ = {v}",
        "pt_true": "Истинный класс: {v}",
        "pt_pred": "Предсказанный класс: {v}",
        "pt_side": "Точка лежит {side} границы",
        "pt_side_on": "Точка лежит ровно на границе",
        "pt_dist": "Расстояние до границы = |z| / |w| = {d}",
        "pt_dist_none": "Вектор весов нулевой — расстояние до границы "
                        "не определено.",
        "calc_1_formula": "1. Формула:  z = w₁x₁ + w₂x₂ + b",
        "calc_2_substitute": "2. Подставляем:  z = {s}",
        "calc_3_value": "3. Складываем:  z = {v}",
        "calc_4_rule": "4. Правило:  {r}",
        "calc_5_class": "5. Значит, классификатор отвечает:  класс {c}",
        "rule_ge": "{z} ≥ 0 → класс 1",
        "rule_lt": "{z} < 0 → класс 0",
        "verdict_ok": "Верно: предсказанный класс совпал с истинным.",
        "verdict_bad": "Ошибка: предсказали {pred}, а истинный класс {true}.",
        "side_left": "слева от",
        "side_right": "справа от",
        "pt_box_table": "Все точки набора",
        "ok_yes": "верно",
        "ok_no": "ошибка",
        "n": "№",
        "x1": "x₁",
        "x2": "x₂",
        "z": "z",
        "true": "истинный",
        "pred": "предсказ.",
        "ok": "верно?",
        # Вкладка «Датасет»
        "ds_box_stats": "Итоги по набору",
        "ds_name_line": "Набор: {name}",
        "ds_total": "Всего точек: {n}",
        "ds_correct": "Верных предсказаний: {n}",
        "ds_wrong": "Ошибок: {n}",
        "ds_accuracy": "Accuracy: {p}",
        "acc_formula": "accuracy = верные / всего",
        "acc_substitute": "accuracy = {c} / {t} = {v} = {p}",
        "ds_box_matrix": "Матрица ошибок",
        "cm_caption": "Строка — истинный класс, столбец — предсказанный.",
        "cm_true0": "истинный 0",
        "cm_true1": "истинный 1",
        "cm_pred0": "предсказ. 0",
        "cm_pred1": "предсказ. 1",
        "cm_note": "Диагональ (TN, TP) — верные ответы. Вне диагонали — "
                   "ошибки: FP (ложная тревога, класс 0 приняли за 1) и FN "
                   "(пропуск, класс 1 приняли за 0).",
        "ds_box_points": "Точки, где классификатор ошибся",
        "ds_errors_hint": "Здесь только неверные предсказания: истинный класс "
                          "не совпал с предсказанным.",
        "ds_errors_none": "Ошибок нет: все точки набора классифицированы "
                          "верно, поэтому список пуст.",
        # Названия и смысл наборов
        "ds_separable_name": "Разделимые классы",
        "ds_separable_desc": "Два облака по разные стороны прямой x₁ + x₂ = 0. "
                             "Одной прямой хватает: accuracy 100%.",
        "ds_overlap_name": "Классы с перекрытием",
        "ds_overlap_desc": "Два облака, но несколько точек уже оказались по "
                           "чужую сторону: даже в лучшем положении прямой "
                           "остаются две ошибки из 15 — accuracy 86.7%.",
        "ds_xor_name": "XOR",
        "ds_xor_desc": "Класс зависит от сочетания признаков. Одной прямой не "
                       "разделить: лучше, чем 6 из 8, не выйдет.",
        "ds_ring_name": "Кольцо",
        "ds_ring_desc": "Класс 0 в центре, класс 1 — вокруг него. Прямая "
                        "всегда разрезает кольцо: перебор всех направлений "
                        "даёт максимум 9 из 13, то есть 69.2%.",
        "ds_stripes_name": "Полосы",
        "ds_stripes_desc": "Класс 0 занимает две отдельные полосы по краям. "
                           "Одна прямая не удержит обе: максимум 9 из 12, то "
                           "есть 75%.",
        # Вкладка «Эксперименты»
        "exp_intro": "Каждый эксперимент ставит набор и параметры и объясняет, "
                     "что смотреть. Нажмите — приложение само переключит всё, "
                     "кроме языка.",
        "exp_box_list": "Учебные сценарии",
        "exp_box_goal": "Что посмотреть",
        "exp_run": "Запустить",
        "exp_done": "Эксперимент «{name}»: набор и параметры установлены.",
        "exp_result": "Результат: верных {c} из {t}, accuracy {p}.",
        # Вкладка «Справка»
        "help_intro": "Кратко о задаче классификации. Разделы слева, текст — "
                      "справа. Стрелки и клавиши Page Up / Page Down листают.",
        "help_nav": "Разделы справки",
        # Строки состояния
        "status_start": "Готово: набор «{name}», параметры w₁ = {w1}, "
                        "w₂ = {w2}, b = {b}. Верных {c} из {t}.",
        "status_params": "Параметры: w₁ = {w1}, w₂ = {w2}, b = {b}. Верных "
                         "{c} из {t}, ошибок {e}.",
        "status_dataset": "Набор «{name}» — параметры не менялись: граница "
                          "та же самая. Верных {c} из {t}.",
        "status_reset": "Сброс: набор «{name}», w₁ = {w1}, w₂ = {w2}, "
                        "b = {b}, выбор точки снят.",
        "status_selected": "Выбрана точка №{n}: истинный класс — {true}, "
                           "оценка z = {z}, ответ — {pred}.",
        "status_invalid": "Поле «{field}»: введите число (запятая или точка).",
        "status_clamped": "Поле «{field}»: значение {raw} вне диапазона "
                          "[{lo}; {hi}], взято {value}.",
        "status_tuned": "Параметры набора «{name}»: w₁ = {w1}, w₂ = {w2}, "
                        "b = {b}.",
        "status_lang": "Язык интерфейса переключён на {name}; состояние "
                       "приложения не изменилось.",
        "status_no_boundary": "w₁ = 0 и w₂ = 0: граница не определена, вся "
                              "плоскость относится к классу {c}.",
        "class0_name": "класс 0",
        "class1_name": "класс 1",
        # Короткие подписи прямо на плоскости
        "legend_c0": "класс 0",
        "legend_c1": "класс 1",
        "legend_boundary": "z = 0",
        "legend_margin": "z = ±1",
        "legend_wrong": "ошибка",
        "legend_selected": "выбрана",
        # Живые объяснения состояния
        "watch_all_ok": "Верных ответов {c} из {t}: весь набор раскладывается "
                        "по двум половинам плоскости, ошибок нет.",
        "watch_errors": "Ошибок {e} из {t}: {fp} — это точки класса 0 в зоне "
                        "класса 1, {fn} — точки класса 1 в зоне класса 0.",
        "watch_vertical": "w₂ = 0, поэтому граница вертикальна: её ставит "
                          "только x₁ = {v}.",
        "watch_horizontal": "w₁ = 0, поэтому граница горизонтальна: "
                            "x₂ = {v}.",
        "watch_ties": "Точек ровно на границе (z = 0): {n}. По правилу они "
                      "получают класс 1.",
        "watch_slope": "Наклон границы равен {r}: его задаёт отношение -w₁ к "
                       "w₂, а b на него не влияет.",
    },
    "en": {
        "app_title": "Classification Lab — how a classifier splits the plane",
        "subtitle": ("Features x₁, x₂ → score z = w₁x₁ + w₂x₂ + b → decision "
                     "boundary → predicted class → comparison with the true "
                     "class"),
        "box_dataset": "Dataset",
        "btn_tuned": "Use the parameters tuned for this set",
        "tuned_hint": "These parameters were typed by hand: nothing here "
                      "trains a model.",
        "box_params": "Classifier parameters",
        "lbl_w1": "Weight w₁",
        "lbl_w2": "Weight w₂",
        "lbl_b": "Bias b",
        "hint_step": "The slider steps by 0.05, the buttons by 0.1. Both a "
                     "comma and a dot are accepted.",
        "btn_reset": "Reset everything",
        "box_rule": "Score and decision rule",
        "rule_formula": "z = w₁x₁ + w₂x₂ + b",
        "rule_text": "If z ≥ 0 → class 1. If z < 0 → class 0. A point exactly "
                     "on the boundary (z = 0) is given class 1.",
        "box_boundary": "Decision boundary",
        "bd_formula": "The boundary is the set of points where z = 0:",
        "bd_equation": "Equation: {eq}",
        "bd_normal": "The boundary's normal is the weight vector "
                     "(w₁, w₂) = ({a}, {b}); it is perpendicular to the line.",
        "bd_degenerate": "w₁ = 0 and w₂ = 0: no line can serve as the "
                         "boundary. The score is the same everywhere and "
                         "equals b = {b}, so the whole plane receives one "
                         "class.",
        "box_watch": "What happened",
        "box_language": "Interface language",
        "lang_ru": "Русский",
        "lang_en": "English",
        "box_legend": "Legend",
        "legend_true0": "True class 0 — square",
        "legend_true1": "True class 1 — circle",
        "legend_region0": "Region the classifier labels 0",
        "legend_region1": "Region the classifier labels 1",
        "legend_boundary": "Decision boundary (z = 0)",
        "legend_margin": "Contour lines z = +1 and z = −1",
        "legend_wrong": "Red ring — the point is misclassified",
        "legend_selected": "Dashed frame — the selected point",
        "tab_point": "Point",
        "tab_dataset": "Dataset",
        "tab_exp": "Experiments",
        "tab_help": "Help",
        "pt_none": "Click any point on the plane — its full calculation "
                   "appears here.",
        "pt_box_point": "Selected point",
        "pt_box_calc": "The score, step by step",
        "pt_box_verdict": "Outcome for the point",
        "pt_x1": "Feature x₁ = {v}",
        "pt_x2": "Feature x₂ = {v}",
        "pt_true": "True class: {v}",
        "pt_pred": "Predicted class: {v}",
        "pt_side": "The point lies {side} the boundary",
        "pt_side_on": "The point lies exactly on the boundary",
        "pt_dist": "Distance to the boundary = |z| / |w| = {d}",
        "pt_dist_none": "Zero weight vector — the distance to the boundary is "
                        "undefined.",
        "calc_1_formula": "1. Formula:  z = w₁x₁ + w₂x₂ + b",
        "calc_2_substitute": "2. Substitute:  z = {s}",
        "calc_3_value": "3. Add up:  z = {v}",
        "calc_4_rule": "4. Rule:  {r}",
        "calc_5_class": "5. So the classifier answers:  class {c}",
        "rule_ge": "{z} ≥ 0 → class 1",
        "rule_lt": "{z} < 0 → class 0",
        "verdict_ok": "Correct: the predicted class matches the true class.",
        "verdict_bad": "Wrong: we predicted {pred}, the true class is {true}.",
        "side_left": "to the left of",
        "side_right": "to the right of",
        "pt_box_table": "Every point of the set",
        "ok_yes": "correct",
        "ok_no": "wrong",
        "n": "#",
        "x1": "x₁",
        "x2": "x₂",
        "z": "z",
        "true": "true",
        "pred": "pred",
        "ok": "correct?",
        "ds_box_stats": "Results over the set",
        "ds_name_line": "Dataset: {name}",
        "ds_total": "Points: {n}",
        "ds_correct": "Correct predictions: {n}",
        "ds_wrong": "Wrong predictions: {n}",
        "ds_accuracy": "Accuracy: {p}",
        "acc_formula": "accuracy = correct / total",
        "acc_substitute": "accuracy = {c} / {t} = {v} = {p}",
        "ds_box_matrix": "Confusion matrix",
        "cm_caption": "Rows are the true class, columns are the predicted one.",
        "cm_true0": "true 0",
        "cm_true1": "true 1",
        "cm_pred0": "predicted 0",
        "cm_pred1": "predicted 1",
        "cm_note": "The diagonal (TN, TP) holds correct answers. Off the "
                   "diagonal are the errors: FP (false alarm — a class 0 "
                   "point counted as 1) and FN (a miss — a class 1 point "
                   "counted as 0).",
        "ds_box_points": "Points the classifier got wrong",
        "ds_errors_hint": "Only the misclassified points: the true class does "
                          "not match the predicted one.",
        "ds_errors_none": "No errors: every point of the set is classified "
                          "correctly, so the list is empty.",
        "ds_separable_name": "Separable classes",
        "ds_separable_desc": "Two clouds on opposite sides of the line "
                             "x₁ + x₂ = 0. One straight line is enough: "
                             "accuracy 100%.",
        "ds_overlap_name": "Overlapping classes",
        "ds_overlap_desc": "Two clouds, but a few points already landed on "
                           "the wrong side: even in the best position of the "
                           "line two of the 15 points stay wrong — accuracy "
                           "86.7%.",
        "ds_xor_name": "XOR",
        "ds_xor_desc": "The class depends on the combination of features. One "
                       "line cannot split it: better than 6 out of 8 is "
                       "impossible.",
        "ds_ring_name": "Ring",
        "ds_ring_desc": "Class 0 in the centre, class 1 around it. A line "
                        "always cuts the ring: checking every direction gives "
                        "at most 9 out of 13, that is 69.2%.",
        "ds_stripes_name": "Stripes",
        "ds_stripes_desc": "Class 0 occupies two separate bands at the edges. "
                           "One line cannot hold both: the best it can do is 9 "
                           "out of 12, that is 75%.",
        "exp_intro": "Each experiment sets a dataset and parameters, then says "
                     "what to look at. Press it — the application switches "
                     "everything except the language.",
        "exp_box_list": "Teaching scenarios",
        "exp_box_goal": "What to look at",
        "exp_run": "Run",
        "exp_done": "Experiment “{name}”: dataset and parameters applied.",
        "exp_result": "Result: {c} correct out of {t}, accuracy {p}.",
        "help_intro": "A short tour of the classification task. Sections on "
                      "the left, text on the right. Arrows and Page Up / "
                      "Page Down scroll.",
        "help_nav": "Help sections",
        "status_start": "Ready: dataset “{name}”, w₁ = {w1}, w₂ = {w2}, "
                        "b = {b}. {c} correct out of {t}.",
        "status_params": "Parameters: w₁ = {w1}, w₂ = {w2}, b = {b}. {c} "
                         "correct out of {t}, {e} wrong.",
        "status_dataset": "Dataset “{name}” — parameters unchanged, the "
                          "boundary is the same line. {c} correct out of {t}.",
        "status_reset": "Reset: dataset “{name}”, w₁ = {w1}, w₂ = {w2}, "
                        "b = {b}, point selection cleared.",
        "status_selected": "Selected point №{n}: true class — {true}, score "
                           "z = {z}, answer — {pred}.",
        "status_invalid": "Field “{field}”: enter a number (comma or dot).",
        "status_clamped": "Field “{field}”: {raw} is outside [{lo}; {hi}], "
                          "using {value}.",
        "status_tuned": "Parameters for dataset “{name}”: w₁ = {w1}, "
                        "w₂ = {w2}, b = {b}.",
        "status_lang": "Interface language switched to {name}; the application "
                       "state did not change.",
        "status_no_boundary": "w₁ = 0 and w₂ = 0: no boundary exists, the "
                              "whole plane is class {c}.",
        "class0_name": "class 0",
        "class1_name": "class 1",
        "legend_c0": "class 0",
        "legend_c1": "class 1",
        "legend_boundary": "z = 0",
        "legend_margin": "z = ±1",
        "legend_wrong": "wrong",
        "legend_selected": "selected",
        "watch_all_ok": "{c} correct out of {t}: the whole set falls neatly "
                        "into the two half-planes, no mistakes.",
        "watch_errors": "Mistakes: {e} out of {t}. Class 0 points inside the "
                        "class 1 region — {fp}; class 1 points inside the "
                        "class 0 region — {fn}.",
        "watch_vertical": "w₂ = 0, so the boundary is vertical: only "
                          "x₁ = {v} decides where it stands.",
        "watch_horizontal": "w₁ = 0, so the boundary is horizontal: "
                            "x₂ = {v}.",
        "watch_ties": "Points standing exactly on the boundary (z = 0): {n}. "
                      "By the rule they receive class 1.",
        "watch_slope": "The boundary's slope is {r}: -w₁ over w₂ sets it, and "
                       "b does not touch it.",
    },
}


# ----------------------------------------------------------------------
# Раздел 6. Учебные эксперименты
# ----------------------------------------------------------------------
# Эксперимент — это состояние приложения: набор, три параметра и выбранная
# точка. Числа подобраны так, чтобы каждый сценарий учил чему-то одному.

EXPERIMENTS = (
    {"key": "perfect", "dataset": "separable", "w1": 1.0, "w2": 1.0,
     "b": 0.0, "point": None},
    {"key": "rotate", "dataset": "separable", "w1": 1.0, "w2": -1.0,
     "b": 0.0, "point": None},
    {"key": "bias_in", "dataset": "separable", "w1": 1.0, "w2": 1.0,
     "b": 2.5, "point": 11},
    {"key": "bias_out", "dataset": "separable", "w1": 1.0, "w2": 1.0,
     "b": -2.5, "point": 0},
    {"key": "no_weights", "dataset": "separable", "w1": 0.0, "w2": 0.0,
     "b": 0.0, "point": None},
    {"key": "overlap", "dataset": "overlap", "w1": 1.0, "w2": -0.3,
     "b": 0.8, "point": 13},
    {"key": "xor_naive", "dataset": "xor", "w1": 1.0, "w2": 0.0,
     "b": 0.0, "point": None},
    {"key": "xor_limit", "dataset": "xor", "w1": -1.0, "w2": 1.0,
     "b": 0.0, "point": None},
    {"key": "ring", "dataset": "ring", "w1": 2.0, "w2": 0.8,
     "b": -2.0, "point": 6},
    {"key": "stripes", "dataset": "stripes", "w1": -1.0, "w2": -2.0,
     "b": -3.0, "point": 2},
)

EXPERIMENT_BY_KEY = {entry["key"]: entry for entry in EXPERIMENTS}

EXP_TEXTS = {
    "ru": {
        "perfect": (
            "Идеально разделимые классы",
            "Граница x₁ + x₂ = 0 проходит между двумя облаками. Ни одна "
            "точка не перешла на чужую сторону: верных 14 из 14, accuracy "
            "100%. Это тот случай, когда одной прямой действительно хватает.",
        ),
        "rotate": (
            "Поворот: прямая прошла сквозь данные",
            "Веса стали (1, -1), и граница повернулась на 90 градусов — теперь "
            "она режет оба облака. Заметьте точки с z = 0: по правилу они "
            "получают класс 1, и истинный класс 0 для них становится ошибкой.",
        ),
        "bias_in": (
            "Смещение сдвигает границу параллельно",
            "Веса прежние, а b выросло до 2.5: прямая не повернулась, а "
            "переехала и отрезала кусок от облака класса 0. Выбрана точка "
            "(-0.5, -2): её z ровно 0, то есть точка стоит на границе.",
        ),
        "bias_out": (
            "Почти идеальная прямая и одна ошибка",
            "Тот же набор, но b = -2.5: граница уехала в другую сторону. "
            "Верных 13 из 14 — accuracy 92.9%, а ошибка спрятана в точке "
            "(1, 1). Одно число по всему набору её не показывает, а красное "
            "кольцо показывает.",
        ),
        "no_weights": (
            "Оба веса равны нулю: границы нет",
            "При w₁ = 0 и w₂ = 0 оценка везде равна b, поэтому делить "
            "плоскость не на что. Прямая не нарисована, и весь экран "
            "окрашен в один класс. Это честное поведение модели, а не "
            "поломка интерфейса.",
        ),
        "overlap": (
            "Перекрытие: ошибки остаются",
            "Прямая с весами (1, -0.3) и b = 0.8 стоит в лучшем из найденных "
            "положений: верных 13 из 15, accuracy 86.7%. Две точки всё равно "
            "лежат не с той стороны — их не спасёт ни поворот, ни сдвиг, "
            "потому что мешает сама геометрия набора. Выбрана точка (0.5, "
            "1.5): истинный класс 0, а предсказан 1.",
        ),
        "xor_naive": (
            "XOR и наивная вертикальная прямая",
            "Прямая x₁ = 0 делит плоскость пополам и угадывает ровно половину "
            "точек: 4 из 8. Класс здесь зависит от сочетания знаков обоих "
            "признаков, а вертикальная прямая про знак x₂ ничего не знает.",
        ),
        "xor_limit": (
            "XOR: предел возможностей одной прямой",
            "Наклон (−1, 1) — лучшая из прямых для этого набора: 6 верных из "
            "8, accuracy 75%. Больше одной прямой дать не может: два класса 1 "
            "лежат в противоположных углах, и любая прямая либо объединяет их "
            "с чужим углом, либо разводит их по разным половинам.",
        ),
        "ring": (
            "Кольцо: прямая не обводит класс вокруг",
            "Прямая повёрнута и сдвинута в лучшее положение: 9 верных из 13, "
            "accuracy 69.2%. Это предел — перебор всех направлений и всех "
            "смещений не даёт лучше. Четыре ошибки — внешние точки класса 1: "
            "прямая всегда разрезает кольцо, и часть круга остаётся в чужой "
            "половине.",
        ),
        "stripes": (
            "Полосы: две области одной прямой нельзя",
            "Класс 0 занимает две несвязные полосы по краям, класс 1 — "
            "среднюю. Лучшая прямая удерживает правую полосу целиком, но "
            "теряет одну точку левой полосы и две точки среднего класса: 9 "
            "верных из 12, accuracy 75%. Прямая умеет отдать классу только "
            "одну половину плоскости.",
        ),
    },
    "en": {
        "perfect": (
            "Perfectly separable classes",
            "The boundary x₁ + x₂ = 0 runs between the two clouds. No point "
            "crossed to the other side: 14 correct out of 14, accuracy 100%. "
            "This is the case where one straight line really is enough.",
        ),
        "rotate": (
            "Rotated: the line cuts through the data",
            "The weights became (1, -1), so the boundary turned by 90 degrees "
            "and now slices both clouds. Look at the points with z = 0: by the "
            "rule they receive class 1, which turns their true class 0 into a "
            "mistake.",
        ),
        "bias_in": (
            "The bias slides the boundary parallel to itself",
            "The weights are the same, but b grew to 2.5: the line did not "
            "rotate, it moved and cut a piece off the class 0 cloud. The "
            "selected point is (-0.5, -2), whose z is exactly 0 — it stands on "
            "the boundary.",
        ),
        "bias_out": (
            "An almost perfect line and one mistake",
            "Same dataset, b = -2.5: the boundary travelled the other way. 13 "
            "correct out of 14 — accuracy 92.9% — while the single error hides "
            "at (1, 1). One number over the whole set does not show it; the "
            "red ring does.",
        ),
        "no_weights": (
            "Both weights zero: no boundary exists",
            "With w₁ = 0 and w₂ = 0 the score equals b everywhere, so there is "
            "nothing to split the plane with. No line is drawn and the whole "
            "canvas is tinted one class. That is the model behaving honestly, "
            "not a broken interface.",
        ),
        "overlap": (
            "Overlap: mistakes stay",
            "The line with weights (1, -0.3) and b = 0.8 stands in the best "
            "position found: 13 correct out of 15, accuracy 86.7%. Two points "
            "still lie on the wrong side — no rotation or shift saves them, "
            "because the geometry of the set itself is in the way. The "
            "selected point is (0.5, 1.5): its true class is 0, and it is "
            "predicted as 1.",
        ),
        "xor_naive": (
            "XOR and a naive vertical line",
            "The line x₁ = 0 halves the plane and guesses exactly half of the "
            "points: 4 out of 8. Here the class depends on the combination of "
            "both signs, while a vertical line knows nothing about the sign of "
            "x₂.",
        ),
        "xor_limit": (
            "XOR: the limit of a single straight line",
            "The slope (-1, 1) is the best line for this set: 6 correct out of "
            "8, accuracy 75%. One line cannot do better: the two class 1 "
            "points sit in opposite corners, so any line either merges them "
            "with a foreign corner or pulls them apart.",
        ),
        "ring": (
            "Ring: a line cannot encircle a class",
            "The line is rotated and shifted into its best position: 9 "
            "correct out of 13, accuracy 69.2%. That is the limit — checking "
            "every direction and every offset does not do better. The four "
            "mistakes are outer class 1 points: a line always cuts the ring "
            "and leaves part of the circle in the wrong half.",
        ),
        "stripes": (
            "Stripes: two regions do not fit one line",
            "Class 0 takes two disconnected bands at the edges, class 1 fills "
            "the middle one. The best line keeps the whole right band but "
            "loses one point of the left band and two points of the middle "
            "class: 9 correct out of 12, accuracy 75%. A line can give a class "
            "only one half of the plane.",
        ),
    },
}


# ----------------------------------------------------------------------
# Раздел 7. Справка
# ----------------------------------------------------------------------
# Формат раздела: (заголовок, строки). Строка, начинающаяся с "»", — ключевая
# формула; с "!" — важная оговорка. Остальные строки — обычный текст.

HELP_KEYS = (
    "task", "features", "binary", "score", "rule", "boundary", "weights",
    "bias", "noline", "correct", "accuracy", "matrix", "separability", "xor",
)

HELP = {
    "ru": {
        "task": (
            "1. Что такое классификация",
            (
                "Классификация — задача угадать метку по признакам: не число, "
                "а название класса. Здесь классов два, а признаков ровно "
                "два, поэтому всю задачу видно на плоскости.",
                "Цепочка на каждом экране одна и та же: признаки x₁ и x₂ → "
                "оценка z → граница решения → предсказанный класс → сравнение "
                "с истинным классом.",
                "! Вопрос приложения не «как обучить модель», а «что именно "
                "происходит, когда модель относит точку к классу».",
            ),
        ),
        "features": (
            "2. Признаки и метки",
            (
                "Признак — это измеримое число про объект. Их два: x₁ по "
                "горизонтали и x₂ по вертикали. Каждая точка плоскости — "
                "объект с двумя такими числами.",
                "Метка (истинный класс) — заранее известный ответ, 0 или 1. "
                "Метка показана формой маркера: квадрат — класс 0, кружок — "
                "класс 1.",
                "Предсказанный класс — ответ классификатора. Он совпадает с "
                "меткой не всегда; разница и есть ошибка.",
                "! Классификатор видит признаки, но метку — нет. Метка нужна "
                "только чтобы проверить его ответ.",
            ),
        ),
        "binary": (
            "3. Бинарная классификация",
            (
                "Классов ровно два, поэтому ответ всегда один из двух, а "
                "плоскость делится на две части.",
                "Область, где классификатор ставит 0, подкрашена голубым, "
                "область с ответом 1 — оранжевым. Цвет области — это "
                "предсказание, форма точки — истина.",
                "Отсюда простое правило чтения картинки: точка лежит в области "
                "своего класса — ответ верный; легла в чужую — ошибка.",
            ),
        ),
        "score": (
            "4. Что значит числовая оценка",
            (
                "Классификатор сначала считает не класс, а число — оценку z.",
                "» z = w₁x₁ + w₂x₂ + b",
                "Знак z говорит, на какой стороне границы лежит точка, а "
                "|z| / |w| — как далеко от неё. Нулевая оценка значит, что "
                "точка стоит ровно на границе.",
                "На вкладке «Точка» видно всю подстановку: формула, значения, "
                "сумма, правило, ответ.",
            ),
        ),
        "rule": (
            "5. Правило решения",
            (
                "Оценка превращается в класс по порогу.",
                "» z ≥ 0 → класс 1,   z < 0 → класс 0",
                "Порог выбран сознательно и одинаков для всех точек. Точка "
                "ровно на границе, где z = 0, получает класс 1 — это хорошо "
                "видно в эксперименте про поворот границы.",
            ),
        ),
        "boundary": (
            "6. Граница решения",
            (
                "Граница — это все точки с z = 0. Это обычная прямая, и она "
                "взята не для красоты: те же w₁, w₂ и b, что и в оценке.",
                "» w₁x₁ + w₂x₂ + b = 0",
                "По одну сторону от прямой z положителен — там предсказывается "
                "класс 1; по другую z отрицателен — там класс 0.",
                "Пунктирные изолинии z = +1 и z = −1 показывают, насколько "
                "далеко точка ушла от порога, то есть насколько уверенным "
                "получается решение.",
            ),
        ),
        "weights": (
            "7. Как веса влияют на границу",
            (
                "Вектор (w₁, w₂) перпендикулярен границе, поэтому веса задают "
                "направление прямой, а не размер области.",
                "w₂ = 0 — граница вертикальна, x₁ = -b / w₁. w₁ = 0 — "
                "горизонтальна. Равные по модулю веса дают наклон ±1.",
                "Двигайте w₁ и w₂: прямая вращается. Если при этом ошибок "
                "становится больше, значит наклон перестал совпадать с "
                "геометрией набора.",
            ),
        ),
        "bias": (
            "8. Как смещение влияет на границу",
            (
                "b не меняет наклон: он переставляет прямую параллельно ей "
                "самой.",
                "» x₂ = -(w₁x₁ + b) / w₂",
                "Держите веса и двигайте только b: в какой-то момент прямая "
                "начнёт отрезать кусок от нужного облака, и появятся ошибки. "
                "Два эксперимента про смещение показывают это с обеих сторон.",
            ),
        ),
        "noline": (
            "9. Когда границы вовсе нет",
            (
                "Если w₁ = 0 и w₂ = 0, то z = b одинаково в любой точке "
                "плоскости. Делить нечего: граница не определена.",
                "Весь экран тогда окрашен в один класс — тот, который "
                "задаёт знак b по правилу z ≥ 0. Приложение пишет прямо, что "
                "границы нет, вместо того чтобы рисовать несуществующую "
                "прямую.",
            ),
        ),
        "correct": (
            "10. Верные и неверные ответы",
            (
                "Точка классифицирована верно, если предсказанный класс "
                "совпал с истинным.",
                "Ошибочные точки обведены красным кольцом и помечены словом "
                "«ошибка» в таблицах. Их видно сразу, даже когда их одна "
                "штук на весь набор.",
                "! Точка ровно на границе засчитывается как класс 1. Если её "
                "истинный класс 0, это ошибка, и кольцо обязательно покажет.",
            ),
        ),
        "accuracy": (
            "11. Accuracy",
            (
                "» accuracy = верные предсказания / все предсказания",
                "На вкладке «Датасет» видно число верных, число ошибок и саму "
                "подстановку, например 11 / 15 = 0.733 = 73.3%.",
                "! Accuracy — одно число на весь набор: оно не говорит, кто "
                "именно ошибся и в какую сторону. Сравните 92.9% при одной "
                "ошибке в эксперименте про смещение.",
            ),
        ),
        "matrix": (
            "12. Матрица ошибок",
            (
                "Строки — истинный класс, столбцы — предсказанный.",
                "» TN — класс 0 и предсказан 0;  TP — класс 1 и предсказан 1",
                "» FP — ложная тревога: истинный 0, а предсказан 1;  FN — "
                "пропуск: истинный 1, а предсказан 0",
                "Диагональ — верные ответы, вне диагонали — ошибки. TN + TP "
                "всегда даёт число верных, FP + FN — число ошибок, поэтому "
                "матрица и accuracy никогда не расходятся.",
                "Матрица пересчитывается на каждое изменение w₁, w₂, b и на "
                "каждую смену набора.",
            ),
        ),
        "separability": (
            "13. Линейная разделимость",
            (
                "Набор линейно разделим, если существует прямая, по одну "
                "сторону которой лежат только точки класса 0, а по другую — "
                "только класса 1.",
                "«Разделимые классы» разделимы. «Классы с перекрытием» при "
                "данной геометрии уже содержат точки одного класса в чужой "
                "области, и прямая этого не исправит.",
                "«Кольцо» и «Полосы» не разделимы: классу нужна либо замкнутая "
                "область, либо две несвязные, а прямая умеет давать только "
                "две полуплоскости.",
            ),
        ),
        "xor": (
            "14. Почему XOR не решается одной прямой",
            (
                "В наборе XOR точки класса 1 стоят в противоположных углах "
                "(знаки x₁ и x₂ совпадают), а точки класса 0 — в двух "
                "остальных углах (знаки разные).",
                "Прямая делит плоскость на две половины. Чтобы удержать оба "
                "угла класса 1 вместе, одна половина должна быть сразу двух "
                "противоположных углов — а тогда в неё попадает и чужой угол.",
                "Проверка перебором для этого набора: любая прямая оставляет "
                "неверными как минимум две точки из восьми. Значит 6 из 8, "
                "или 75% — это предел одной линейной границы.",
                "! Именно поэтому в нейронных сетях появляются слои: одна "
                "линейная граница — только первый шаг, а не решение.",
            ),
        ),
    },
    "en": {
        "task": (
            "1. What classification is",
            (
                "Classification is the task of guessing a label from features: "
                "not a number, but a class name. There are two classes and "
                "exactly two features here, so the whole task fits on a plane.",
                "The chain is the same on every screen: features x₁ and x₂ → "
                "score z → decision boundary → predicted class → comparison "
                "with the true class.",
                "! The question this application asks is not “how do we train a "
                "model” but “what exactly happens when a model puts a point "
                "into a class”.",
            ),
        ),
        "features": (
            "2. Features and labels",
            (
                "A feature is a measurable number about an object. There are "
                "two: x₁ along the horizontal axis and x₂ along the vertical. "
                "Each point of the plane is one object with those two numbers.",
                "The label (true class) is the answer known in advance: 0 or 1. "
                "It is drawn as the shape of the marker — a square is class 0, "
                "a circle is class 1.",
                "The predicted class is the classifier's answer. It does not "
                "always match the label, and the difference is the error.",
                "! The classifier sees the features but never the label. The "
                "label is only there to check its answer.",
            ),
        ),
        "binary": (
            "3. Binary classification",
            (
                "There are exactly two classes, so the answer is always one of "
                "two and the plane splits into two parts.",
                "The region where the classifier answers 0 is tinted blue, the "
                "region where it answers 1 is tinted orange. The colour of the "
                "region is the prediction; the shape of the point is the truth.",
                "That gives a simple way to read the picture: a point inside "
                "the region of its own class is correct, a point that landed in "
                "the other region is wrong.",
            ),
        ),
        "score": (
            "4. What the score means",
            (
                "The classifier computes a number before it computes a class — "
                "the score z.",
                "» z = w₁x₁ + w₂x₂ + b",
                "The sign of z says which side of the boundary the point lies "
                "on, and |z| / |w| says how far from it. A zero score means "
                "the point stands exactly on the boundary.",
                "The Point tab shows the whole substitution: formula, numbers, "
                "sum, rule, answer.",
            ),
        ),
        "rule": (
            "5. The decision rule",
            (
                "The score becomes a class through a threshold.",
                "» z ≥ 0 → class 1,   z < 0 → class 0",
                "The threshold is chosen deliberately and is the same for every "
                "point. A point exactly on the boundary, where z = 0, receives "
                "class 1 — the rotation experiment shows that clearly.",
            ),
        ),
        "boundary": (
            "6. The decision boundary",
            (
                "The boundary is the set of all points with z = 0. It is an "
                "ordinary straight line, and it is not decoration: the same "
                "w₁, w₂ and b that compute the score.",
                "» w₁x₁ + w₂x₂ + b = 0",
                "On one side of the line z is positive — class 1 is predicted "
                "there; on the other z is negative — class 0.",
                "The dashed contours z = +1 and z = −1 show how far a point has "
                "moved away from the threshold, that is how confident the "
                "decision is.",
            ),
        ),
        "weights": (
            "7. How the weights move the boundary",
            (
                "The vector (w₁, w₂) is perpendicular to the boundary, so the "
                "weights set the direction of the line, not the size of a "
                "region.",
                "w₂ = 0 makes the boundary vertical, x₁ = -b / w₁. w₁ = 0 "
                "makes it horizontal. Equal weights in absolute value give a "
                "slope of ±1.",
                "Drag w₁ and w₂: the line rotates. If the number of mistakes "
                "grows, the slope has stopped matching the geometry of the set.",
            ),
        ),
        "bias": (
            "8. How the bias moves the boundary",
            (
                "b does not change the slope: it slides the line parallel to "
                "itself.",
                "» x₂ = -(w₁x₁ + b) / w₂",
                "Keep the weights and move only b: at some moment the line "
                "starts cutting a piece off the cloud it needs, and mistakes "
                "appear. The two bias experiments show it from both sides.",
            ),
        ),
        "noline": (
            "9. When there is no boundary at all",
            (
                "If w₁ = 0 and w₂ = 0 then z = b is the same in every point of "
                "the plane. There is nothing to divide.",
                "The whole canvas is then tinted one class — the one the sign "
                "of b gives under the rule z ≥ 0. The application says plainly "
                "that no boundary exists instead of drawing a line that is not "
                "there.",
            ),
        ),
        "correct": (
            "10. Correct and incorrect answers",
            (
                "A point is classified correctly when the predicted class "
                "equals the true class.",
                "Wrong points are drawn with a red ring and marked “wrong” in "
                "the tables. They are visible even when there is a single one "
                "in the whole set.",
                "! A point exactly on the boundary counts as class 1. If its "
                "true class is 0, that is a mistake, and the ring will show it.",
            ),
        ),
        "accuracy": (
            "11. Accuracy",
            (
                "» accuracy = correct predictions / all predictions",
                "The Dataset tab shows the number of correct answers, the "
                "number of mistakes and the substitution itself, for example "
                "11 / 15 = 0.733 = 73.3%.",
                "! Accuracy is one number for the whole set: it does not say "
                "who made the mistake or in which direction. Compare 92.9% "
                "with a single error in the bias experiment.",
            ),
        ),
        "matrix": (
            "12. The confusion matrix",
            (
                "Rows are the true class, columns are the predicted one.",
                "» TN — true 0, predicted 0;  TP — true 1, predicted 1",
                "» FP — false alarm: true 0, predicted 1;  FN — a miss: true "
                "1, predicted 0",
                "The diagonal holds the correct answers, off it are the "
                "mistakes. TN + TP always equals the number of correct "
                "predictions and FP + FN the number of mistakes, so the matrix "
                "and the accuracy can never disagree.",
                "The matrix is recomputed on every change of w₁, w₂, b and on "
                "every switch of the dataset.",
            ),
        ),
        "separability": (
            "13. Linear separability",
            (
                "A set is linearly separable when some straight line has only "
                "class 0 points on one side and only class 1 points on the "
                "other.",
                "“Separable classes” is separable. “Overlapping classes” "
                "already contains points of one class in the other class's "
                "region, and no line fixes that.",
                "“Ring” and “Stripes” are not separable: the class would "
                "need either a closed region or two disconnected ones, while a "
                "line only ever gives two half-planes.",
            ),
        ),
        "xor": (
            "14. Why XOR cannot be solved by one line",
            (
                "In the XOR set the class 1 points stand in opposite corners "
                "(the signs of x₁ and x₂ agree), while the class 0 points fill "
                "the other two corners (the signs differ).",
                "A line splits the plane into two halves. To keep both class 1 "
                "corners together, one half would have to cover two opposite "
                "corners — and then a foreign corner lands in it too.",
                "Checking all cases for this set: any line leaves at least two "
                "of the eight points wrong. So 6 out of 8, that is 75%, is the "
                "limit of a single linear boundary.",
                "! This is why neural networks have layers: one linear "
                "boundary is only the first step, not the answer.",
            ),
        ),
    },
}


def T(key, **kwargs):
    text = TEXTS[LANG].get(key)
    if text is None:
        return "[" + key + "]"
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


# ----------------------------------------------------------------------
# Раздел 8. Виджеты
# ----------------------------------------------------------------------

def to_text(value):
    """Число в поле ввода: коротко, но без потери смысла."""
    return fmt(value, 3)


def fill_text(widget, lines):
    """Заполняет read-only текстовое окно парами (тег, строка)."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    for tag, text in lines:
        widget.insert("end", text + "\n", tag)
    widget.configure(state="disabled")


class ScrollFrame(ttk.Frame):
    """Вертикально прокручиваемая колонка: левый столбец выше экрана."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=COL_BG,
                                borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient="vertical",
                                 command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.inner,
                                                anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.bind("<Enter>", lambda e: self.canvas.bind(
            "<MouseWheel>", self._on_wheel))
        self.bind("<Leave>", lambda e: self.canvas.unbind("<MouseWheel>"))

    def _on_inner(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 90), "units")


class ParamControl(ttk.Frame):
    """Один параметр: подпись, слайдер, поле ввода и кнопки шага."""

    def __init__(self, master, app, label_key, low, high, value):
        super().__init__(master)
        self.app = app
        self.label_key = label_key
        self.low = low
        self.high = high
        self.value = value
        self._echo = False

        self.label = ttk.Label(self, text=T(label_key),
                               font=app.fonts["smallb"])
        self.label.grid(row=0, column=0, columnspan=4, sticky="w")

        self.var_num = tk.DoubleVar(value=value)
        self.var_text = tk.StringVar(value=to_text(value))
        # Цвет бегунка tk.Scale берётся из bg: на светлом фоне такой ползунок
        # не виден, поэтому bg — акцентный, а дорожка — «бумажная».
        self.scale = tk.Scale(
            self, from_=low, to=high, resolution=SLIDER_STEP,
            orient="horizontal", showvalue=False, length=140,
            sliderrelief="flat", sliderlength=22, bd=0,
            highlightthickness=1, highlightbackground=COL_FRAME,
            troughcolor=COL_PAPER, bg=COL_HEAD, fg=COL_TEXT,
            variable=self.var_num, command=self._from_slider)
        self.scale.grid(row=1, column=0, sticky="w")

        self.entry = ttk.Entry(self, textvariable=self.var_text, width=7,
                               justify="right", font=app.fonts["mono"])
        self.entry.grid(row=1, column=1, padx=(6, 0), sticky="w")
        self.entry.bind("<Return>", self._commit)
        self.entry.bind("<FocusOut>", self._commit)

        self.down = ttk.Button(self, text="-0.1", width=4,
                               command=lambda: self._nudge(-NUDGE))
        self.down.grid(row=1, column=2, padx=(5, 0), sticky="w")
        self.up = ttk.Button(self, text="+0.1", width=4,
                             command=lambda: self._nudge(NUDGE))
        self.up.grid(row=1, column=3, padx=(3, 0), sticky="w")

    def _from_slider(self, _raw):
        if self._echo:
            return
        value = self.var_num.get()
        self.value = clamp(value, self.low, self.high)
        self._echo = True
        self.var_text.set(to_text(self.value))
        self._echo = False
        self._mark(False)
        self.app.refresh()
        self.app.announce_params()

    def _commit(self, _event=None):
        raw = self.var_text.get()
        parsed = parse_number(raw)
        if parsed is None:
            self._mark(True)
            self.app.set_status("status_invalid", field=T(self.label_key))
            return
        value = clamp(parsed, self.low, self.high)
        clamped = value != parsed
        if clamped:
            self.app.set_status("status_clamped", field=T(self.label_key),
                                raw=fmt(parsed, 3), lo=fmt(self.low),
                                hi=fmt(self.high), value=fmt(value, 3))
        self.value = value
        self._mark(False)
        self.var_text.set(to_text(value))
        self._echo = True
        self.var_num.set(round(value / SLIDER_STEP) * SLIDER_STEP)
        self._echo = False
        self.app.refresh()
        if not clamped:
            self.app.announce_params()

    def _nudge(self, step):
        self.var_text.set(to_text(clamp(self.value + step, self.low,
                                         self.high)))
        self._commit()

    def set_value(self, value):
        self.value = clamp(value, self.low, self.high)
        self.var_text.set(to_text(self.value))
        self._echo = True
        self.var_num.set(round(self.value / SLIDER_STEP) * SLIDER_STEP)
        self._echo = False
        self._mark(False)

    def _mark(self, bad):
        self.entry.configure(style="BadEntry.TEntry" if bad else
                             "GoodEntry.TEntry")

    def retranslate(self):
        self.label.configure(text=T(self.label_key))


class PlaneCanvas(tk.Canvas):
    """Плоскость признаков: зоны, граница, изолинии, точки, легенда."""

    def __init__(self, master, app):
        super().__init__(master, bg="white", highlightthickness=2,
                         highlightbackground=COL_FRAME)
        self.app = app
        self.unit = 40.0
        self.x_min = self.y_min = WORLD_MIN
        self.x_max = self.y_max = WORLD_MAX
        self.left = PAD_LEFT
        self.top = PAD_TOP
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._on_click)

    # --- преобразования координат -------------------------------------
    def to_screen(self, x, y):
        return (self.left + (x - self.x_min) * self.unit,
                self.top + (self.y_max - y) * self.unit)

    def _plan(self):
        width = max(self.winfo_width(), 200)
        height = max(self.winfo_height(), 200)
        avail_w = width - PAD_LEFT - PAD_RIGHT
        avail_h = height - PAD_TOP - PAD_BOTTOM
        span = WORLD_MAX - WORLD_MIN
        # Единственный масштаб по обеим осям: перпендикуляр к границе должен
        # выглядеть перпендикуляром, а не наклонной линией.
        self.unit = max(6.0, min(avail_w / span, avail_h / span))
        # Короткая сторона задаёт масштаб, длинная показывает больше мира:
        # иначе на узком окне вокруг квадрата остаются пустые полосы.
        self.x_max = max(WORLD_MAX, avail_w / self.unit / 2.0)
        self.y_max = max(WORLD_MAX, avail_h / self.unit / 2.0)
        self.x_min, self.y_min = -self.x_max, -self.y_max
        self.left = PAD_LEFT + (avail_w - (self.x_max - self.x_min)
                                * self.unit) / 2.0
        self.top = PAD_TOP + (avail_h - (self.y_max - self.y_min)
                              * self.unit) / 2.0

    def _view_rect(self):
        return [(self.x_min, self.y_min), (self.x_max, self.y_min),
                (self.x_max, self.y_max), (self.x_min, self.y_max)]

    def _ticks(self, low, high):
        """Целые отметки оси внутри видимого отрезка (ноль не подписываем)."""
        value = math.ceil(low - 1e-9)
        while value <= math.floor(high + 1e-9):
            if value != 0:
                yield float(value)
            value += 1

    def _flat(self, polygon):
        out = []
        for x, y in polygon:
            sx, sy = self.to_screen(x, y)
            out.extend((sx, sy))
        return out

    # --- отрисовка ----------------------------------------------------
    def redraw(self):
        self.delete("all")
        self._plan()
        state = self.app.state_values()
        w1, w2, b = state["w1"], state["w2"], state["b"]
        rows = self.app.last_rows
        selected = self.app.selected

        self._draw_regions(w1, w2, b)
        self._draw_grid()
        self._draw_axes()
        if has_boundary(w1, w2):
            self._draw_line(level=MARGIN, dash=(3, 4), width=1,
                           color=COL_MARGIN, w1=w1, w2=w2, b=b)
            self._draw_line(level=-MARGIN, dash=(3, 4), width=1,
                           color=COL_MARGIN, w1=w1, w2=w2, b=b)
            self._draw_line(level=0.0, dash=None, width=4,
                           color=COL_BOUNDARY, w1=w1, w2=w2, b=b)
        for index, row in enumerate(rows):
            self._draw_point(index, row, selected, w1, w2, b)
        self._draw_legend()

    def _draw_regions(self, w1, w2, b):
        rect = self._view_rect()
        if not has_boundary(w1, w2):
            color = COL_REGION1 if predict(b) == 1 else COL_REGION0
            self.create_polygon(self._flat(rect), fill=color,
                                outline=color)
            return
        one = clip_half_plane(rect, w1, w2, b, True)
        zero = clip_half_plane(rect, w1, w2, b, False)
        for polygon, color in ((zero, COL_REGION0), (one, COL_REGION1)):
            if len(polygon) >= 3:
                self.create_polygon(self._flat(polygon), fill=color,
                                    outline=color)

    def _draw_grid(self):
        for x in self._ticks(self.x_min, self.x_max):
            sx, sy0 = self.to_screen(x, self.y_min)
            _s, sy1 = self.to_screen(x, self.y_max)
            self.create_line(sx, sy1, sx, sy0, fill=COL_GRID,
                             width=2 if abs(x) < 1e-9 else 1)
            self.create_text(sx, sy0 + 14, text=fmt(x, 0),
                             fill=COL_DIM, font=self.app.fonts["tiny"])
        for y in self._ticks(self.y_min, self.y_max):
            sx0, sy = self.to_screen(self.x_min, y)
            sx1, _s = self.to_screen(self.x_max, y)
            self.create_line(sx0, sy, sx1, sy, fill=COL_GRID,
                             width=2 if abs(y) < 1e-9 else 1)
            self.create_text(sx0 - 12, sy, text=fmt(y, 0), fill=COL_DIM,
                             font=self.app.fonts["tiny"])

    def _draw_axes(self):
        ax, ay = self.to_screen(self.x_min, 0.0)
        bx, by = self.to_screen(self.x_max, 0.0)
        self.create_line(ax, ay, bx - 4, by, fill=COL_AXIS, width=2,
                         arrow="last", arrowshape=(11, 15, 7))
        self.create_text(bx - 10, by - 14, text="x" + sub(1), fill=COL_AXIS,
                         font=self.app.fonts["midb"])
        cx, cy = self.to_screen(0.0, self.y_min)
        dx, dy = self.to_screen(0.0, self.y_max)
        self.create_line(cx, cy, dx, dy + 4, fill=COL_AXIS, width=2,
                         arrow="last", arrowshape=(11, 15, 7))
        self.create_text(dx + 14, dy + 12, text="x" + sub(2), fill=COL_AXIS,
                         font=self.app.fonts["midb"])

    def _draw_line(self, w1, w2, b, level, color, width, dash):
        seg = segment_on_plane(w1, w2, b - level, self.x_min, self.x_max,
                               self.y_min, self.y_max)
        if seg is None:
            return
        (x0, y0), (x1, y1) = seg
        sx0, sy0 = self.to_screen(x0, y0)
        sx1, sy1 = self.to_screen(x1, y1)
        self.create_line(sx0, sy0, sx1, sy1, fill=color, width=width,
                         dash=dash, capstyle="round")

    def _draw_point(self, index, row, selected, w1, w2, b):
        sx, sy = self.to_screen(row["x1"], row["x2"])
        color = COL_CLASS1 if row["true"] == 1 else COL_CLASS0
        if row["true"] == 1:
            self.create_oval(sx - 8, sy - 8, sx + 8, sy + 8, fill=color,
                             outline="white", width=2)
        else:
            self.create_rectangle(sx - 7, sy - 7, sx + 7, sy + 7, fill=color,
                                  outline="white", width=2)
        if not row["correct"]:
            self.create_oval(sx - 13, sy - 13, sx + 13, sy + 13,
                             outline=COL_WRONG, width=3)
        if index == selected:
            self.create_oval(sx - 18, sy - 18, sx + 18, sy + 18,
                             outline=COL_SELECTED, width=2, dash=(4, 3))
            self.create_text(sx + 22, sy - 20, anchor="w",
                             text="z = " + fmt_score(row["z"]),
                             fill=COL_SELECTED, font=self.app.fonts["smallb"])
        self.create_text(sx - 12, sy + 12, anchor="e", text=str(index + 1),
                         fill=COL_TEXT, font=self.app.fonts["tiny"])

    def _draw_legend(self):
        items = (
            (COL_CLASS1, "oval", T("legend_c1")),
            (COL_CLASS0, "rect", T("legend_c0")),
            (COL_BOUNDARY, "line", T("legend_boundary")),
            (COL_MARGIN, "dash", T("legend_margin")),
            (COL_WRONG, "ring", T("legend_wrong")),
            (COL_SELECTED, "ring", T("legend_selected")),
        )
        font = self.app.fonts["tiny"]
        widest = 0
        for _color, _kind, text in items:
            widest = max(widest, font.measure(text))
        box_w = widest + 44
        box_h = len(items) * 17 + 14
        x0 = self.left + 8
        y0 = self.top + 8
        self.create_rectangle(x0, y0, x0 + box_w, y0 + box_h,
                              fill="white", outline=COL_FRAME)
        for position, (color, kind, text) in enumerate(items):
            cy = y0 + 14 + position * 17
            glyph_x = x0 + 16
            if kind == "oval":
                self.create_oval(glyph_x - 5, cy - 5, glyph_x + 5, cy + 5,
                                 fill=color, outline="white")
            elif kind == "rect":
                self.create_rectangle(glyph_x - 5, cy - 5, glyph_x + 5,
                                      cy + 5, fill=color, outline="white")
            elif kind == "ring":
                self.create_oval(glyph_x - 6, cy - 6, glyph_x + 6, cy + 6,
                                 outline=color, width=2)
            elif kind == "dash":
                self.create_line(glyph_x - 9, cy, glyph_x + 9, cy, fill=color,
                                 width=2, dash=(3, 3))
            else:
                self.create_line(glyph_x - 9, cy, glyph_x + 9, cy, fill=color,
                                 width=3)
            self.create_text(glyph_x + 15, cy, anchor="w", text=text,
                             fill=COL_TEXT, font=font)

    def _on_click(self, event):
        rows = self.app.last_rows
        best, best_distance = None, 18.0
        for index, row in enumerate(rows):
            sx, sy = self.to_screen(row["x1"], row["x2"])
            distance = math.hypot(sx - event.x, sy - event.y)
            if distance < best_distance:
                best, best_distance = index, distance
        if best is not None:
            self.app.select_point(best)


# ----------------------------------------------------------------------
# Раздел 9. Главное окно
# ----------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title(T("app_title"))
        self.configure(bg=COL_BG)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.minsize(min(1260, screen_w - 40), min(740, screen_h - 60))
        self.geometry("%dx%d" % (min(max(1280, int(screen_w * 0.9)), screen_w - 20),
                                 min(max(800, int(screen_h * 0.88)), screen_h - 20)))
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        base = tkfont.nametofont("TkDefaultFont").actual("family")
        mono = tkfont.nametofont("TkFixedFont").actual("family")

        def make_font(family, size, weight="normal"):
            return tkfont.Font(family=family, size=size, weight=weight)

        # Шрифты — объекты Font: их можно передавать в widget font= и
        # измерять линейкой canvas, чтобы ничего не обрезалось.
        self.fonts = {
            "title": make_font(base, 16, "bold"),
            "sub": make_font(base, 13),
            "midb": make_font(base, 13, "bold"),
            "small": make_font(base, 12),
            "smallb": make_font(base, 12, "bold"),
            "tiny": make_font(base, 10),
            "mono": make_font(mono, 12),
            "monob": make_font(mono, 12, "bold"),
        }
        self.option_add("*Font", (base, 13))
        self._setup_styles()

        # Состояние: язык его не трогает.
        self.dataset_key = DEFAULT_DATASET
        self.w1, self.w2, self.b = DEFAULT_W1, DEFAULT_W2, DEFAULT_B
        self.selected = None
        self.experiment_key = None
        self.help_key = HELP_KEYS[0]
        self.last_rows = []
        self._table_heads = []
        self.last_result = evaluate(self.points(), self.w1, self.w2, self.b)

        self._text_refs = []
        self.var_status = tk.StringVar()
        self.var_lang = tk.StringVar(value=LANG)

        self._build()
        self._fill_exp_list()
        self._fill_help_list()
        self.exp_list.selection_set(0)
        self.refresh()
        self._show_help()
        self._show_exp_goal()
        result = self.last_result
        self.set_status("status_start", name=self.dataset_name(),
                        w1=fmt(self.w1), w2=fmt(self.w2), b=fmt(self.b),
                        c=result["correct"], t=result["total"])

    # --- оформление ---------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook.Tab", padding=(9, 5))
        style.configure("TLabelframe", background=COL_BG)
        style.configure("TLabelframe.Label", background=COL_BG,
                        foreground=COL_HEAD, font=self.fonts["smallb"])
        style.configure("TFrame", background=COL_BG)
        style.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        style.configure("Title.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["title"])
        style.configure("Sub.TLabel", background=COL_BG, foreground="#5a5a54",
                        font=self.fonts["small"])
        style.configure("Key.TLabel", background=COL_BG, foreground=COL_KEY,
                        font=self.fonts["monob"])
        style.configure("Head.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["smallb"])
        style.configure("Muted.TLabel", background=COL_BG, foreground=COL_DIM,
                        font=self.fonts["tiny"])
        style.configure("Bad.TLabel", background=COL_BG, foreground=COL_BAD_FG,
                        font=self.fonts["smallb"])
        style.configure("Cell.TLabel", background=COL_PAPER,
                        foreground=COL_TEXT, relief="solid", borderwidth=1)
        style.configure("Go.TButton", font=self.fonts["smallb"])
        style.map("Go.TButton", foreground=[("active", COL_HEAD)])
        style.configure("GoodEntry.TEntry", fieldbackground=COL_PAPER)
        style.configure("BadEntry.TEntry", fieldbackground=COL_BAD_BG,
                        foreground=COL_BAD_FG)

    def _reg(self, widget, key, attr="text", **kwargs):
        self._text_refs.append((widget, key, attr, kwargs))
        widget.configure(**{attr: T(key, **kwargs)})

    def _box(self, parent, title_key):
        box = ttk.LabelFrame(parent, text=T(title_key), padding=(9, 5))
        box.pack(fill="x", pady=(0, 7))
        self._text_refs.append((box, title_key, "text", {}))
        return box

    def _label(self, parent, key, font=None, style=None, **kwargs):
        widget = ttk.Label(parent, font=font, style=style, **kwargs)
        self._reg(widget, key)
        return widget

    # --- сборка -------------------------------------------------------
    def _build(self):
        header = ttk.Frame(self, padding=(12, 8, 12, 4))
        header.grid(row=0, column=0, columnspan=3, sticky="we")
        self._label(header, "app_title", style="Title.TLabel").pack(
            anchor="w")
        self._label(header, "subtitle", style="Sub.TLabel",
                    wraplength=1150, justify="left").pack(anchor="w")

        body = ttk.Frame(self, padding=(10, 4, 10, 4))
        body.grid(row=1, column=0, columnspan=3, sticky="nsew")
        # Строка с тулбаром обязана оставаться внизу: без веса средняя строка
        # растягивается по содержимому и нижняя панель уезжает за окно.
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.columnconfigure(1, weight=1)
        body.columnconfigure(0, minsize=340, weight=0)
        body.columnconfigure(1, minsize=400, weight=1)
        body.columnconfigure(2, minsize=432, weight=0)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_center(body)
        self._build_right(body)
        self._build_bottom()

    def _build_left(self, parent):
        scroll = ScrollFrame(parent)
        scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left = scroll.inner

        box = self._box(left, "box_dataset")
        self.combo_dataset = ttk.Combobox(box, state="readonly", width=25)
        self.combo_dataset.pack(fill="x")
        self.combo_dataset.bind("<<ComboboxSelected>>", self._on_dataset)
        self._fill_dataset_combo()
        self.combo_dataset.current(
            [entry["key"] for entry in DATASETS].index(self.dataset_key))
        self.lbl_ds_desc = ttk.Label(box, wraplength=296, justify="left",
                                     font=self.fonts["small"])
        self.lbl_ds_desc.pack(anchor="w", pady=(5, 0))
        self.btn_tuned = ttk.Button(box, text=T("btn_tuned"),
                                    command=self.apply_tuned)
        self._reg(self.btn_tuned, "btn_tuned")
        self.btn_tuned.pack(anchor="w", pady=(6, 2))
        self._label(box, "tuned_hint", style="Muted.TLabel",
                    wraplength=296, justify="left").pack(anchor="w")

        box = self._box(left, "box_params")
        self.controls = {}
        for key, low, high, value in (
                ("w1", W_MIN, W_MAX, DEFAULT_W1),
                ("w2", W_MIN, W_MAX, DEFAULT_W2),
                ("b", B_MIN, B_MAX, DEFAULT_B)):
            label_key = {"w1": "lbl_w1", "w2": "lbl_w2", "b": "lbl_b"}[key]
            control = ParamControl(box, self, label_key, low, high, value)
            control.pack(fill="x", pady=(0, 6))
            self.controls[key] = control
        self.btn_reset = ttk.Button(box, text=T("btn_reset"),
                                    command=self.reset_all)
        self._reg(self.btn_reset, "btn_reset")
        self.btn_reset.pack(anchor="w", pady=(2, 4))
        self._label(box, "hint_step", style="Muted.TLabel", wraplength=296,
                    justify="left").pack(anchor="w")

        box = self._box(left, "box_rule")
        self._label(box, "rule_formula", style="Key.TLabel").pack(anchor="w")
        self._label(box, "rule_text", wraplength=296, justify="left",
                    font=self.fonts["small"]).pack(anchor="w", pady=(3, 0))

        box = self._box(left, "box_boundary")
        self._label(box, "bd_formula", font=self.fonts["small"]).pack(
            anchor="w")
        self.lbl_equation = ttk.Label(box, style="Key.TLabel", wraplength=296,
                                      justify="left")
        self.lbl_equation.pack(anchor="w", pady=(2, 0))
        self.lbl_normal = ttk.Label(box, wraplength=296, justify="left",
                                    font=self.fonts["small"])
        self.lbl_normal.pack(anchor="w", pady=(3, 0))

        box = self._box(left, "box_watch")
        self.lbl_watch = ttk.Label(box, wraplength=296, justify="left",
                                   font=self.fonts["small"])
        self.lbl_watch.pack(anchor="w")

        box = self._box(left, "box_legend")
        for key in ("legend_true0", "legend_true1", "legend_region0",
                    "legend_region1", "legend_boundary", "legend_margin",
                    "legend_wrong", "legend_selected"):
            self._label(box, key, font=self.fonts["tiny"], wraplength=296,
                        justify="left").pack(anchor="w")

        box = self._box(left, "box_language")
        self._lang_buttons = []
        for code, key in (("ru", "lang_ru"), ("en", "lang_en")):
            button = ttk.Radiobutton(box, text=T(key), value=code,
                                     variable=self.var_lang,
                                     command=self.switch_language)
            button.pack(anchor="w")
            self._reg(button, key)
            self._lang_buttons.append(button)

    def _build_center(self, parent):
        center = ttk.Frame(parent)
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.plane = PlaneCanvas(center, self)
        self.plane.grid(row=0, column=0, sticky="nsew")

    def _build_right(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.grid(row=0, column=2, sticky="nsew")
        self.tab_point = ttk.Frame(self.nb, padding=8)
        self.tab_dataset = ttk.Frame(self.nb, padding=8)
        self.tab_exp = ttk.Frame(self.nb, padding=8)
        self.tab_help = ttk.Frame(self.nb, padding=8)
        self._tabs = ((self.tab_point, "tab_point"),
                      (self.tab_dataset, "tab_dataset"),
                      (self.tab_exp, "tab_exp"),
                      (self.tab_help, "tab_help"))
        for frame, key in self._tabs:
            self.nb.add(frame, text=T(key))
        self._build_tab_point()
        self._build_tab_dataset()
        self._build_tab_exp()
        self._build_tab_help()

    def _build_tab_point(self):
        tab = self.tab_point
        box = self._box(tab, "pt_box_point")
        self.lbl_point = ttk.Label(box, wraplength=336, justify="left",
                                   font=self.fonts["small"])
        self.lbl_point.pack(anchor="w")

        box = self._box(tab, "pt_box_calc")
        self.lbl_calc = ttk.Label(box, wraplength=336, justify="left",
                                  font=self.fonts["mono"])
        self.lbl_calc.pack(anchor="w")

        box = self._box(tab, "pt_box_verdict")
        self.lbl_verdict = ttk.Label(box, wraplength=336, justify="left",
                                     font=self.fonts["smallb"])
        self.lbl_verdict.pack(anchor="w")
        self.lbl_side = ttk.Label(box, wraplength=336, justify="left",
                                  font=self.fonts["small"])
        self.lbl_side.pack(anchor="w", pady=(3, 0))

        box = self._box(tab, "pt_box_table")
        self.tree_points = self._make_table(box, TABLE_KEYS, TABLE_WIDTHS)
        self.tree_points.bind("<<TreeviewSelect>>", self._on_table_select)

    def _build_tab_dataset(self):
        tab = self.tab_dataset
        box = self._box(tab, "ds_box_stats")
        self.lbl_ds_line = ttk.Label(box, wraplength=336, justify="left",
                                     font=self.fonts["smallb"])
        self.lbl_ds_line.pack(anchor="w")
        self.lbl_ds_counts = ttk.Label(box, wraplength=336, justify="left",
                                       font=self.fonts["small"])
        self.lbl_ds_counts.pack(anchor="w", pady=(3, 0))
        self._label(box, "acc_formula", style="Key.TLabel").pack(anchor="w",
                                                                 pady=(4, 0))
        self.lbl_acc = ttk.Label(box, style="Key.TLabel", wraplength=336,
                                 justify="left")
        self.lbl_acc.pack(anchor="w", pady=(2, 0))

        box = self._box(tab, "ds_box_matrix")
        self._label(box, "cm_caption", font=self.fonts["tiny"],
                    wraplength=336, justify="left").pack(anchor="w")
        grid = ttk.Frame(box)
        grid.pack(pady=(6, 2))
        ttk.Label(grid).grid(row=0, column=0, padx=4)   # пустой угол сетки
        for column, key in ((1, "cm_pred0"), (2, "cm_pred1")):
            head = ttk.Label(grid, style="Head.TLabel", text=T(key),
                             wraplength=96)
            self._reg(head, key)
            head.grid(row=0, column=column, padx=6, pady=2)
        self.matrix_cells = {}
        layout = ((1, "cm_true0", (("tn", "TN"), ("fp", "FP"))),
                  (2, "cm_true1", (("fn", "FN"), ("tp", "TP"))))
        for row, row_key, cells in layout:
            name = ttk.Label(grid, style="Head.TLabel", text=T(row_key),
                             font=self.fonts["small"])
            self._reg(name, row_key)
            name.grid(row=row, column=0, padx=4, sticky="e")
            for column, (cell_key, letter) in enumerate(cells, start=1):
                cell = ttk.Label(grid, style="Cell.TLabel",
                                 font=self.fonts["monob"], width=7,
                                 anchor="center")
                cell.grid(row=row, column=column, padx=4, pady=3)
                self.matrix_cells[cell_key] = (cell, letter)
        self._label(box, "cm_note", font=self.fonts["tiny"], wraplength=336,
                    justify="left").pack(anchor="w", pady=(4, 0))

        box = self._box(tab, "ds_box_points")
        self.lbl_errors_hint = ttk.Label(box, style="Muted.TLabel",
                                         wraplength=336, justify="left",
                                         font=self.fonts["tiny"])
        self.lbl_errors_hint.pack(anchor="w", pady=(0, 4))
        self.tree_errors = self._make_table(box, TABLE_KEYS, TABLE_WIDTHS)
        self.tree_errors.bind("<<TreeviewSelect>>", self._on_table_select)

    def _make_table(self, parent, keys, widths):
        frame = ttk.Frame(parent)
        frame.pack(fill="x")
        tree = ttk.Treeview(frame, columns=keys, show="headings", height=6)
        for key, width in zip(keys, widths):
            tree.heading(key, text=T(key))
            tree.column(key, width=width, anchor="center", stretch=True)
        tree.tag_configure("wrong", foreground=COL_WRONG)
        tree.tag_configure("sel", font=self.fonts["smallb"])
        tree.tag_configure("na", foreground=COL_DIM)
        bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=bar.set)
        tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self._table_heads.append(tree)
        return tree

    def _build_tab_exp(self):
        tab = self.tab_exp
        self._label(tab, "exp_intro", wraplength=336, justify="left",
                    font=self.fonts["small"]).pack(anchor="w", pady=(0, 6))
        box = self._box(tab, "exp_box_list")
        frame = ttk.Frame(box)
        frame.pack(fill="x")
        self.exp_list = tk.Listbox(frame, height=7, activestyle="dotbox",
                                   exportselection=False, font=self.fonts["small"],
                                   background=COL_PAPER, selectbackground="#d8e4ee",
                                   selectforeground=COL_TEXT, relief="solid",
                                   borderwidth=1)
        self.exp_list.pack(side="left", fill="both", expand=True)
        self.exp_list.bind("<<ListboxSelect>>", self._on_exp_select)
        self.btn_exp = ttk.Button(box, text=T("exp_run"), style="Go.TButton",
                                  command=self.run_selected_experiment)
        self._reg(self.btn_exp, "exp_run")
        self.btn_exp.pack(anchor="w", pady=(6, 0))

        box = self._box(tab, "exp_box_goal")
        self.text_exp = self._make_text(box, height=5)
        self.lbl_exp_result = ttk.Label(box, style="Head.TLabel",
                                        wraplength=336, justify="left")
        self.lbl_exp_result.pack(anchor="w", pady=(5, 0))

    def _build_tab_help(self):
        tab = self.tab_help
        self._label(tab, "help_intro", wraplength=336, justify="left",
                    font=self.fonts["small"]).pack(anchor="w", pady=(0, 6))
        box = self._box(tab, "help_nav")
        frame = ttk.Frame(box)
        frame.pack(fill="both", expand=True)
        self.help_list = tk.Listbox(frame, height=7, activestyle="none",
                                    exportselection=False,
                                    font=self.fonts["small"],
                                    background=COL_PAPER,
                                    selectbackground="#d8e4ee",
                                    selectforeground=COL_TEXT, relief="solid",
                                    borderwidth=1)
        self.help_list.pack(side="left", fill="both", expand=True)
        self.help_list.bind("<<ListboxSelect>>", self._on_help_select)
        nav = ttk.Frame(box)
        nav.pack(anchor="w", pady=(5, 0))
        self.btn_help_prev = ttk.Button(nav, text="<", width=4,
                                        command=lambda: self.help_move(-1))
        self.btn_help_prev.pack(side="left")
        self.btn_help_next = ttk.Button(nav, text=">", width=4,
                                        command=lambda: self.help_move(1))
        self.btn_help_next.pack(side="left", padx=(4, 0))
        self.text_help = self._make_text(tab, height=9)

    def _make_text(self, parent, height):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        # width обязателен: без него tk.Text просит 80 символов и правая
        # колонка перестаёт помещаться в окно на маленьких размерах.
        widget = tk.Text(frame, width=TEXT_WIDTH, height=height, wrap="word",
                         relief="flat",
                         background=COL_PAPER, foreground=COL_TEXT,
                         borderwidth=0, highlightthickness=1,
                         highlightbackground=COL_FRAME, font=self.fonts["small"],
                         padx=8, pady=6)
        bar = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=bar.set, state="disabled")
        widget.tag_configure("h", font=self.fonts["smallb"],
                             foreground=COL_HEAD)
        widget.tag_configure("t", font=self.fonts["small"])
        widget.tag_configure("k", font=self.fonts["monob"],
                             foreground=COL_KEY)
        widget.tag_configure("b", font=self.fonts["small"],
                             foreground=COL_WRONG)
        widget.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        return widget

    # --- нижняя строка ------------------------------------------------
    def _build_bottom(self):
        bar = ttk.Frame(self, padding=(12, 3, 16, 3))
        bar.grid(row=2, column=0, columnspan=3, sticky="we")
        status = ttk.Label(bar, textvariable=self.var_status, anchor="w",
                           font=self.fonts["small"], foreground="#3a3a35")
        status.pack(side="left", fill="x", expand=True)
        # Имя автора постоянно, внизу справа, не переводится и не учебный текст.
        credit = ttk.Label(bar, text="SteveWorks001",
                           font=self.fonts["smallb"], foreground=COL_DIM)
        credit.pack(side="right", padx=(10, 0))
        credit.tkraise()

    # --- служебное ----------------------------------------------------
    def set_status(self, key, **kwargs):
        self.var_status.set(T(key, **kwargs))

    def announce_params(self):
        """Строка состояния после изменения параметров: числа и итог."""
        result = self.last_result
        self.set_status("status_params", w1=fmt(self.w1), w2=fmt(self.w2),
                        b=fmt(self.b), c=result["correct"],
                        t=result["total"], e=result["wrong"])

    def state_values(self):
        return {"w1": self.w1, "w2": self.w2, "b": self.b}

    def points(self):
        return DATASET_BY_KEY[self.dataset_key]["points"]

    def dataset_name(self):
        return T("ds_" + self.dataset_key + "_name")

    def current_state(self):
        """Состояние для проверок: смена языка ничего в нём не трогает."""
        return {"lang": get_language(), "dataset": self.dataset_key,
                "w1": self.w1, "w2": self.w2, "b": self.b,
                "selected": self.selected, "tab": self._current_tab_key(),
                "help": self.help_key, "experiment": self.experiment_key,
                "correct": self.last_result["correct"],
                "total": self.last_result["total"]}

    def _current_tab_key(self):
        for frame, key in self._tabs:
            if self.nb.select() == str(frame):
                return key
        return None

    # --- обработчики --------------------------------------------------
    def _on_dataset(self, _event=None):
        index = self.combo_dataset.current()
        self.dataset_key = DATASETS[index]["key"]
        self.experiment_key = None
        if self.selected is not None and self.selected >= len(self.points()):
            self.selected = None
        self.refresh()
        result = self.last_result
        self.set_status("status_dataset", name=self.dataset_name(),
                        c=result["correct"], t=result["total"])

    def apply_tuned(self):
        w1, w2, b = DATASET_BY_KEY[self.dataset_key]["tuned"]
        self.controls["w1"].set_value(w1)
        self.controls["w2"].set_value(w2)
        self.controls["b"].set_value(b)
        self.refresh()
        self.set_status("status_tuned", name=self.dataset_name(), w1=fmt(w1),
                        w2=fmt(w2), b=fmt(b))

    def reset_all(self):
        self.dataset_key = DEFAULT_DATASET
        index = [entry["key"] for entry in DATASETS].index(DEFAULT_DATASET)
        self.combo_dataset.current(index)
        self.controls["w1"].set_value(DEFAULT_W1)
        self.controls["w2"].set_value(DEFAULT_W2)
        self.controls["b"].set_value(DEFAULT_B)
        self.selected = None
        self.experiment_key = None
        self._sync_exp_list(None)
        self.refresh()
        self.set_status("status_reset", name=self.dataset_name(),
                        w1=fmt(DEFAULT_W1), w2=fmt(DEFAULT_W2),
                        b=fmt(DEFAULT_B))

    def select_point(self, index):
        self.selected = index
        self.refresh()
        row = self.last_rows[index]
        self.set_status("status_selected", n=index + 1,
                        true=T("class%d_name" % row["true"]),
                        z=fmt_score(row["z"]),
                        pred=T("class%d_name" % row["pred"]))

    def _on_table_select(self, event):
        tree = event.widget
        selected_ids = tree.selection()
        if not selected_ids:
            return
        number = int(tree.item(selected_ids[0], "values")[0])
        if number - 1 != self.selected:
            self.select_point(number - 1)

    def _on_exp_select(self, _event=None):
        self._show_exp_goal()

    def run_selected_experiment(self):
        index = self.exp_list.curselection()
        key = EXPERIMENTS[index[0]]["key"] if index else self.experiment_key
        if key is None:
            key = EXPERIMENTS[0]["key"]
        self.run_experiment(key)

    def run_experiment(self, key):
        entry = EXPERIMENT_BY_KEY[key]
        self.dataset_key = entry["dataset"]
        index = [d["key"] for d in DATASETS].index(entry["dataset"])
        self.combo_dataset.current(index)
        self.controls["w1"].set_value(entry["w1"])
        self.controls["w2"].set_value(entry["w2"])
        self.controls["b"].set_value(entry["b"])
        self.selected = entry["point"]
        self.experiment_key = key
        self._sync_exp_list(key)
        self._show_exp_goal()
        self.refresh()
        name = EXP_TEXTS[get_language()][key][0]
        result = self.last_result
        self.var_status.set(
            T("exp_done", name=name) + "  " +
            T("exp_result", c=result["correct"], t=result["total"],
              p=fmt_percent(result["accuracy"])))

    def _sync_exp_list(self, key):
        self.exp_list.selection_clear(0, "end")
        if key is None:
            return
        for position, entry in enumerate(EXPERIMENTS):
            if entry["key"] == key:
                self.exp_list.selection_set(position)
                self.exp_list.see(position)
                break

    def _show_exp_goal(self):
        index = self.exp_list.curselection()
        if not index:
            fill_text(self.text_exp, [])
            self.lbl_exp_result.configure(text="")
            return
        entry = EXPERIMENTS[index[0]]
        name, goal = EXP_TEXTS[get_language()][entry["key"]]
        points = DATASET_BY_KEY[entry["dataset"]]["points"]
        result = evaluate(points, entry["w1"], entry["w2"], entry["b"])
        fill_text(self.text_exp, [("h", name), ("t", goal)])
        self.lbl_exp_result.configure(
            text=T("exp_result", c=result["correct"], t=result["total"],
                   p=fmt_percent(result["accuracy"])))

    def _on_help_select(self, _event=None):
        index = self.help_list.curselection()
        if index:
            self.help_key = HELP_KEYS[index[0]]
            self._show_help()

    def help_move(self, step):
        position = HELP_KEYS.index(self.help_key) + step
        position = clamp(position, 0, len(HELP_KEYS) - 1)
        self.help_key = HELP_KEYS[position]
        self._sync_help_list()
        self._show_help()

    def _sync_help_list(self):
        self.help_list.selection_clear(0, "end")
        position = HELP_KEYS.index(self.help_key)
        self.help_list.selection_set(position)
        self.help_list.see(position)

    def _show_help(self):
        title, lines = HELP[get_language()][self.help_key]
        out = []
        for line in lines:
            if line.startswith("»"):
                out.append(("k", line))
            elif line.startswith("!"):
                out.append(("b", "!" + line[1:]))
            else:
                out.append(("t", line))
        fill_text(self.text_help, out)

    # --- обновление ---------------------------------------------------
    def refresh(self):
        self.w1 = self.controls["w1"].value
        self.w2 = self.controls["w2"].value
        self.b = self.controls["b"].value
        result = evaluate(self.points(), self.w1, self.w2, self.b)
        self.last_result = result
        self.last_rows = result["rows"]
        self.plane.redraw()
        self._refresh_left(result)
        self._refresh_point_tab(result)
        self._refresh_dataset_tab(result)
        self._refresh_tables(result)

    def _refresh_left(self, result):
        self.lbl_ds_desc.configure(text=T("ds_" + self.dataset_key + "_desc"))
        if not has_boundary(self.w1, self.w2):
            self.lbl_equation.configure(
                text=T("bd_degenerate", b=fmt_score(self.b)))
            self.lbl_normal.configure(text=T("status_no_boundary",
                                             c=T("class%d_name"
                                                 % predict(self.b))))
        else:
            self.lbl_equation.configure(
                text=T("bd_equation",
                       eq=boundary_equation_text(self.w1, self.w2, self.b)))
            self.lbl_normal.configure(
                text=T("bd_normal", a=fmt(self.w1), b=fmt(self.w2)))
        lines = [T("status_params", w1=fmt(self.w1), w2=fmt(self.w2),
                   b=fmt(self.b), c=result["correct"], t=result["total"],
                   e=result["wrong"])]
        if has_boundary(self.w1, self.w2):
            if result["wrong"] == 0:
                lines.append(T("watch_all_ok", c=result["correct"],
                               t=result["total"]))
            else:
                matrix = result["matrix"]
                lines.append(T("watch_errors", e=result["wrong"],
                               t=result["total"], fp=matrix["fp"],
                               fn=matrix["fn"]))
            if self.w2 == 0.0:
                lines.append(T("watch_vertical",
                               v=fmt_score(-self.b / self.w1)))
            elif self.w1 == 0.0:
                lines.append(T("watch_horizontal",
                               v=fmt_score(-self.b / self.w2)))
            else:
                lines.append(T("watch_slope", r=fmt(-self.w1 / self.w2)))
            ties = sum(1 for row in result["rows"] if row["z"] == 0.0)
            if ties:
                lines.append(T("watch_ties", n=ties))
        self.lbl_watch.configure(text="\n".join(lines))

    def _refresh_point_tab(self, result):
        index = self.selected
        if index is None or index >= len(result["rows"]):
            self.lbl_point.configure(text=T("pt_none"))
            self.lbl_calc.configure(text="")
            self.lbl_verdict.configure(text="")
            self.lbl_side.configure(text="")
            return
        row = result["rows"][index]
        w1, w2, b = self.w1, self.w2, self.b
        self.lbl_point.configure(text="\n".join((
            "№ %d" % (index + 1),
            T("pt_x1", v=fmt(row["x1"])),
            T("pt_x2", v=fmt(row["x2"])),
            T("pt_true", v=T("class%d_name" % row["true"])),
            T("pt_pred", v=T("class%d_name" % row["pred"])),
        )))
        steps = (
            T("calc_1_formula"),
            T("calc_2_substitute",
              s=substitute_sum(w1, w2, b, row["x1"], row["x2"])),
            T("calc_3_value", v=fmt_score(row["z"])),
            T("calc_4_rule", r=T("rule_ge" if row["z"] >= 0 else "rule_lt",
                                 z=fmt_score(row["z"]))),
            T("calc_5_class", c=row["pred"]),
        )
        self.lbl_calc.configure(text="\n".join(steps))
        self.lbl_verdict.configure(
            text=T("verdict_ok") if row["correct"] else
            T("verdict_bad", pred=T("class%d_name" % row["pred"]),
              true=T("class%d_name" % row["true"])),
            style="Bad.TLabel" if not row["correct"] else "Head.TLabel")
        if not has_boundary(w1, w2):
            self.lbl_side.configure(text=T("pt_dist_none"))
        else:
            distance = margin_distance(row["z"], w1, w2)
            if row["z"] == 0.0:
                side_line = T("pt_side_on")
            else:
                side = T("side_right") if row["z"] > 0 else T("side_left")
                side_line = T("pt_side", side=side)
            self.lbl_side.configure(text="\n".join((
                side_line,
                T("pt_dist", d=fmt_score(distance)),
            )))

    def _refresh_dataset_tab(self, result):
        matrix = result["matrix"]
        self.lbl_ds_line.configure(text=T("ds_name_line",
                                          name=self.dataset_name()))
        self.lbl_ds_counts.configure(text="\n".join((
            T("ds_total", n=result["total"]),
            T("ds_correct", n=result["correct"]),
            T("ds_wrong", n=result["wrong"]),
            T("ds_accuracy", p=fmt_percent(result["accuracy"])),
        )))
        self.lbl_acc.configure(
            text=T("acc_substitute", c=result["correct"], t=result["total"],
                   v=fmt_score(result["accuracy"], 3),
                   p=fmt_percent(result["accuracy"])))
        values = {"tn": matrix["tn"], "fp": matrix["fp"],
                  "fn": matrix["fn"], "tp": matrix["tp"]}
        for key, (cell, letter) in self.matrix_cells.items():
            cell.configure(text="%s = %d" % (letter, values[key]))

    def _refresh_tables(self, result):
        mistakes = [(index, row) for index, row in enumerate(result["rows"])
                    if not row["correct"]]
        self.lbl_errors_hint.configure(
            text=T("ds_errors_none") if not mistakes
            else T("ds_errors_hint"))
        tables = (
            (self.tree_points, list(enumerate(result["rows"]))),
            (self.tree_errors, mistakes),
        )
        for tree, rows in tables:
            tree.delete(*tree.get_children())
            for index, row in rows:
                tags = []
                if not row["correct"]:
                    tags.append("wrong")
                if index == self.selected:
                    tags.append("sel")
                tree.insert("", "end", tags=tuple(tags), values=(
                    index + 1, fmt(row["x1"]), fmt(row["x2"]), row["true"],
                    fmt_score(row["z"]), row["pred"],
                    T("ok_yes") if row["correct"] else T("ok_no")))

    # --- язык ---------------------------------------------------------
    def switch_language(self):
        code = self.var_lang.get()
        if code == get_language():
            return
        set_language(code)
        self.title(T("app_title"))
        for widget, key, attr, kwargs in self._text_refs:
            try:
                widget.configure(**{attr: T(key, **kwargs)})
            except tk.TclError:
                pass
        for frame, key in self._tabs:
            self.nb.tab(frame, text=T(key))
        for tree in self._table_heads:
            self._retable(tree)
        for control in self.controls.values():
            control.retranslate()
        self._fill_dataset_combo()
        self._fill_exp_list()
        self._fill_help_list()
        self.refresh()
        self._show_help()
        self._show_exp_goal()
        name = T("lang_ru") if code == "ru" else T("lang_en")
        self.set_status("status_lang", name=name)

    def _retable(self, tree):
        for column in tree["columns"]:
            tree.heading(column, text=T(column))

    def _fill_dataset_combo(self):
        values = [T("ds_" + entry["key"] + "_name") for entry in DATASETS]
        index = self.combo_dataset.current()
        self.combo_dataset.configure(values=values)
        if index >= 0:
            self.combo_dataset.current(index)

    def _fill_exp_list(self):
        keep = self.exp_list.curselection()
        self.exp_list.delete(0, "end")
        for entry in EXPERIMENTS:
            self.exp_list.insert("end",
                                 EXP_TEXTS[get_language()][entry["key"]][0])
        if keep:
            self.exp_list.selection_set(keep[0])

    def _fill_help_list(self):
        self.help_list.delete(0, "end")
        for key in HELP_KEYS:
            self.help_list.insert("end", HELP[get_language()][key][0])
        self._sync_help_list()


# ----------------------------------------------------------------------
# Раздел 10. Точка входа
# ----------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
