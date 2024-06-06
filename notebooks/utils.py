import  numpy as np, os, pyfoil  
from scipy.spatial.distance import cdist 
 
import pandas  as pd 

def load_training_data(
    numInternal: int = 100,
    numXmax:int=100,
    numXmin:int=100,
    numYmax:int=100,
    numYmin:int=100,
    path2case: str = os.path.join(os.getcwd(),'NACA0012', 'case.csv.gzip')
):
    # Загрузка данных
    data = pd.read_csv(path2case, compression='gzip').dropna().apply(pd.to_numeric, errors='coerce').dropna()
    data = data.astype('float64')
    # Определение минимальных и максимальных значений
    x_min, y_min, UMin, VMin, pMin = data[['x', 'y', 'U', 'V', 'p']].min()
    x_max, y_max, UMax, VMax, pMax = data[['x', 'y', 'U', 'V', 'p']].max()

    # Генерация случайных индексов для внутренних точек
    selected_indices = np.random.choice(len(data), min(numInternal, len(data)), replace=True)
    internal_data = data.iloc[selected_indices]

    # Генерация случайных индексов для Xmax, Xmin, Ymax, Ymin
    xmax_data = data[data['x'] == x_max].sample(n=numXmax, replace=True)
    xmin_data = data[data['x'] == x_min].sample(n=numXmin, replace=True)
    ymax_data = data[data['y'] == y_max].sample(n=numYmax, replace=True)
    ymin_data = data[data['y'] == y_min].sample(n=numYmin, replace=True)
    print(f'xmax_data.shape {xmax_data.shape}')
    print(f'xmin_data.shape {xmin_data.shape}')
    print(f'ymax_data.shape {ymax_data.shape}')
    print(f'ymin_data.shape {ymin_data.shape}')

    # Сборка вывода
    U_selected = np.concatenate((internal_data['U'], xmax_data['U'], xmin_data['U'], ymax_data['U'], ymin_data['U'])).reshape(-1,1)
    V_selected = np.concatenate((internal_data['V'], xmax_data['V'], xmin_data['V'], ymax_data['V'], ymin_data['V'])).reshape(-1,1)
    P_selected = np.concatenate((internal_data['p'], xmax_data['p'], xmin_data['p'], ymax_data['p'], ymin_data['p'])).reshape(-1,1)
    x_selected = np.concatenate((internal_data['x'], [x_max] * numXmax,[x_min] * numXmin,   ymax_data['x'], ymin_data['x'])).reshape(-1,1)
    y_selected = np.concatenate((internal_data['y'], xmax_data['y'], xmin_data['y'],[y_max] * numYmax,[y_min] * numYmin)).reshape(-1,1)
    timeSteps = np.concatenate((internal_data['Time'].to_numpy(), xmax_data['Time'].to_numpy(), xmin_data['Time'].to_numpy(), ymax_data['Time'].to_numpy(), ymin_data['Time'].to_numpy())).reshape(-1,1)

    # Возврат данных
    return U_selected, V_selected, P_selected, x_selected, y_selected, x_min, y_min, x_max, y_max, UMax, VMax, pMax, timeSteps

def rotate_points(points, center, angle_deg):
    """
    Поворачивает массив точек относительно центра на угол angle_deg.

    Args:
        points: массив точек в формате numpy array, каждая точка представлена координатами (x, y)
        center: координаты центра в формате (x, y)
        angle_deg: угол поворота в градусах
    Returns:
        rotated_points: повернутые точки в формате numpy array
    """
    # Переводим угол в радианы
    angle_rad = np.deg2rad(angle_deg)

    # Вычисляем матрицу поворота
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Переносим центр в начало координат
    translated_points = points - center

    # Поворачиваем точки
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # Возвращаем точки снова в исходное положение
    return rotated_points + center

def point_on_contour(contour_points, point, threshold=1e-6):
    """
    Check if a point lies on the contour.

    Args:
        contour_points (np.array): Array of shape (N, 2) representing the contour points.
        point (np.array): Array of shape (2,) representing the point in 2D space.
        threshold (float): Threshold to consider the point lies on the contour.

    Returns:
        bool: True if the point lies on the contour, False otherwise.
    """
    distances = cdist([point], contour_points)
    return np.any(np.abs(distances) < threshold)

def boundaryNACA(
    naca_code:str='23012',  # Код профиля крыла
    c:float=0.2,            # Коэффициент масштабирования
    offset_x:float=0.0,     # Смещение профиля вдоль оси 0X
    offset_y:float=0.0,     # Смещение профиля вдоль оси 0Y
    angle_deg:float=10,     # Угол атаки, в градусах
    center=np.array([0,0])  # Точка вращения контура профиля
):

    foil = pyfoil.Airfoil.compute_naca(int(naca_code)).normalized()
    a = np.array(foil.curve.tolist())
    _points = np.array(
        pyfoil.Airfoil.compute_naca(
            int(naca_code)
            ).normalized().curve.scale(c).tolist()
        )
    _points = rotate_points(
        _points,
        center=center,
        angle_deg=angle_deg
    )

    return _points
