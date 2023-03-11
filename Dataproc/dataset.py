import h5py
from torch import Tensor
from torch.utils.data import Dataset


class PinnFileDataset(Dataset):
    def __init__(self, filename: str) -> None:
        super(PinnFileDataset, self).__init__()
        self.filename = filename

    def __getitem__(self, index: int) -> Tensor:
        with h5py.File(self.filename, 'r') as f:
            x = f['x'][index]
            return x

    def __len__(self) -> int:
        with h5py.File(self.filename, 'r') as f:
            return len(f['x'])
