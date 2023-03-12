import numpy as np
import h5py


def generate_data(n_points: int, mesh_dim: int) -> np.ndarray:
    lower_limits = np.arange(0, n_points) / n_points
    upper_limits = np.arange(1, 1 + n_points) / n_points
    points = np.random.uniform(low=lower_limits, high=upper_limits, size=[mesh_dim, n_points]).T
    np.random.shuffle(points[:, 1])
    return points


def save_mesh_to_file(point_vectors: np.ndarray, filename: str = 'mesh.h5') -> None:
    f = h5py.File(filename)
    f.create_dataset("x", data=point_vectors)
    f.close()
    return None
