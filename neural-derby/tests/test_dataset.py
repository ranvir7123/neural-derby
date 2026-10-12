import torch

from derby.data import backup, dataset


def fake_loaded(n=40):
    x = torch.randint(0, 256, (n, 3, 96, 96), dtype=torch.uint8)
    y = torch.arange(n) % 6
    return backup.Loaded(x, y, [None] * n)


def test_assemble_backup_only():
    d = dataset.assemble(fake_loaded(40), None, None, seed=1)
    assert d.train_x.shape[0] == 34 and d.test_x.shape[0] == 6
    assert d.test_is_live.dtype == torch.bool and not d.test_is_live.any()
    assert d.n_live_train == 0 and d.n_live_test == 0


def test_assemble_with_live_rows_appended_and_flagged():
    live_tr = (torch.zeros(5, 3, 96, 96, dtype=torch.uint8), torch.zeros(5, dtype=torch.long))
    live_te = (torch.zeros(3, 3, 96, 96, dtype=torch.uint8), torch.ones(3, dtype=torch.long))
    d = dataset.assemble(fake_loaded(40), live_tr, live_te, seed=1)
    assert d.train_x.shape[0] == 39 and d.test_x.shape[0] == 9
    assert d.test_is_live.tolist() == [False] * 6 + [True] * 3
    assert d.n_live_train == 5 and d.n_live_test == 3
    assert d.train_y.dtype == torch.long and d.test_x.dtype == torch.uint8


def test_assemble_splits_backup_by_photo_when_groups_given():
    n = 40
    loaded = fake_loaded(n)
    for i in range(n):
        loaded.x[i] = i                      # every row is one constant value, so its id can be read back
    loaded.groups = [f"photo_{i // 2}" for i in range(n)]   # 20 photos, 2 crops each
    d = dataset.assemble(loaded, None, None, seed=1)
    test_rows = set(d.test_x[:, 0, 0, 0].tolist())
    train_rows = set(d.train_x[:, 0, 0, 0].tolist())
    assert not test_rows & train_rows
    assert test_rows | train_rows == set(range(n))
    assert len(test_rows) == d.test_x.shape[0] and len(train_rows) == d.train_x.shape[0]
    for photo in range(n // 2):
        crops = {2 * photo, 2 * photo + 1}
        assert crops <= test_rows or crops <= train_rows
