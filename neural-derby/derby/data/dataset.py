"""Put backup and live photos together into the tensors a race trains and tests on."""
from dataclasses import dataclass

import torch

from derby.data.backup import Loaded, split_by_group, split_indices


@dataclass
class RaceData:
    train_x: torch.Tensor      # uint8 N x 3 x 96 x 96
    train_y: torch.Tensor      # long N
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_is_live: torch.Tensor  # bool, True for rows that came from the room's phones
    n_live_train: int
    n_live_test: int

    @property
    def n_train(self) -> int:
        return int(self.train_y.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.test_y.shape[0])


def assemble(backup: Loaded, live_train, live_test, seed: int, device="cpu") -> RaceData:
    """The backup split is by source photo when the loader provides groups, so twin crops never straddle train and test."""
    if backup.groups is not None:
        tr_idx, te_idx = split_by_group(backup.groups, seed)
    else:
        tr_idx, te_idx = split_indices(int(backup.y.shape[0]), seed)
    train_x = [backup.x[tr_idx]]
    train_y = [backup.y[tr_idx]]
    test_x = [backup.x[te_idx]]
    test_y = [backup.y[te_idx]]
    is_live = [torch.zeros(len(te_idx), dtype=torch.bool)]
    n_live_train = n_live_test = 0
    if live_train is not None and live_train[0].shape[0] > 0:
        train_x.append(live_train[0]); train_y.append(live_train[1])
        n_live_train = int(live_train[0].shape[0])
    if live_test is not None and live_test[0].shape[0] > 0:
        test_x.append(live_test[0]); test_y.append(live_test[1])
        is_live.append(torch.ones(live_test[0].shape[0], dtype=torch.bool))
        n_live_test = int(live_test[0].shape[0])
    return RaceData(
        torch.cat(train_x).to(device), torch.cat(train_y).to(device),
        torch.cat(test_x).to(device), torch.cat(test_y).to(device),
        torch.cat(is_live).to(device), n_live_train, n_live_test,
    )
