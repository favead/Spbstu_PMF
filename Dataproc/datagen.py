from typing import Tuple, List, Union
import numpy as np


def generate_data(mesh_info: List[Tuple[Union[int, float], Union[int, float], Union[int, float]]]) -> np.ndarray:
    """
	Генерация данных из гиперкуба с помощью pyDOE lhs
    input: mesh_info - Лист кортежей вида (min, max, ndots).
    Первый индекс массива - ось x, второй - ось y, третий - ось z.
    Следовательно, нужна обработка всех случаев длинны массива
    """
    pass