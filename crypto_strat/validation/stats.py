"""
Статистика для поправки на multiple testing (р.4, барьер 5).

Задача, которую здесь решаем. Перебрали 500 конфигов — лучший покажет высокий
Sharpe ПРОСТО ПОТОМУ, что он лучший из 500, даже если эджа нет ни у одного.
Максимум из N случайных величин смещён вверх, и смещение растёт с N.
Значит планка значимости обязана расти вместе с числом проб.

Реализовано по Bailey & Lopez de Prado:
  * PSR  — Probabilistic Sharpe Ratio: вероятность, что истинный SR выше
           порога, с поправкой на асимметрию и толстые хвосты выборки;
  * E[max SR] — ожидаемый максимум Sharpe по N независимым пустышкам;
  * DSR  — Deflated Sharpe Ratio = PSR относительно этого максимума.

DSR отвечает ровно на нужный вопрос: «учитывая, что мы перебрали N штук,
какова вероятность, что у победителя эдж настоящий?»

scipy не используется намеренно — обратная функция нормального распределения
реализована здесь (алгоритм Acklam), чтобы модуль не тянул зависимостей.
"""

from __future__ import annotations

import math

import numpy as np

EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Обратная функция нормального распределения (Acklam), точность ~1e-9."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q, r = p - 0.5, (p - 0.5) ** 2
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    # один шаг уточнения Галлея
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def probabilistic_sharpe_ratio(sr: float, n_obs: int, skew: float, kurtosis: float,
                               sr_benchmark: float = 0.0) -> float:
    """P(истинный SR > sr_benchmark). sr — НЕ годовой, на одно наблюдение (сделку).

    Знаменатель учитывает форму распределения: отрицательная асимметрия и
    толстые хвосты (типично для стратегий с фиксированным стопом и редкими
    крупными выигрышами — или наоборот) снижают доверие к тому же Sharpe.
    """
    if n_obs < 3:
        return 0.0
    var = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr ** 2
    if var <= 0:
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(var)
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] по N независимым пробам с нулевым истинным эджем.

    Это и есть та планка, которую победитель обязан ПЕРЕПРЫГНУТЬ, чтобы его
    результат не объяснялся одним лишь фактом перебора.
    """
    n = max(int(n_trials), 1)
    if n == 1 or sr_variance <= 0:
        return 0.0
    sd = math.sqrt(sr_variance)
    a = norm_ppf(1.0 - 1.0 / n)
    b = norm_ppf(1.0 - 1.0 / (n * math.e))
    return sd * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_ratio(sr: float, n_obs: int, skew: float, kurtosis: float,
                          n_trials: int, sr_variance: float) -> dict:
    """DSR: вероятность, что эдж настоящий, С УЧЁТОМ числа перебранных гипотез."""
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    dsr = probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, sr_benchmark=sr0)
    return {"dsr": dsr, "sr": sr, "sr_threshold": sr0, "n_trials": int(n_trials),
            "n_obs": int(n_obs), "sr_variance": float(sr_variance)}


def min_track_record_length(sr: float, skew: float, kurtosis: float,
                            sr_benchmark: float = 0.0, confidence: float = 0.95) -> float:
    """Сколько сделок НУЖНО, чтобы Sharpe такого размера был значим.

    Полезно как честный ответ на «а хватает ли данных»: если требуется 900
    сделок, а есть 120 — метрика ничего не доказывает, сколько бы красиво
    она ни выглядела.
    """
    if sr <= sr_benchmark:
        return float("inf")
    var = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr ** 2
    if var <= 0:
        return float("inf")
    z = norm_ppf(confidence)
    return 1.0 + var * (z / (sr - sr_benchmark)) ** 2


def sr_variance_across_trials(sharpes: list[float]) -> float:
    """Дисперсия Sharpe по перебранным конфигам — вход для E[max SR].

    Берём именно наблюдённый разброс: он отражает, насколько «широко»
    разбросан перебор, а значит насколько высоко мог залететь лучший по
    чистой случайности.
    """
    arr = np.asarray([s for s in sharpes if np.isfinite(s)], dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(arr.var(ddof=1))
