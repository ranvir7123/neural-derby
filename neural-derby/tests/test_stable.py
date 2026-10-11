import torch
from derby.horses import stable


def test_stable_order_and_keys():
    assert [h.key for h in stable.STABLE] == [
        "textbook", "stairs", "elevator", "featherweight", "spotlight", "trifocals", "influencer",
    ]


def test_every_horse_maps_image_to_six_logits():
    x = torch.randn(1, 3, 96, 96)
    for h in stable.STABLE:
        m = h.build()
        m.eval()
        assert m(x).shape == (1, 6), h.key


def test_enabled_stable_respects_switches():
    chosen = stable.enabled_stable({"textbook": True, "influencer": False})
    keys = [h.key for h in chosen]
    assert "influencer" not in keys
    assert "textbook" in keys and "stairs" in keys  # missing keys default to enabled


def test_stairs_and_elevator_differ_only_by_shortcuts():
    from derby.horses.stats import count_params
    s = count_params(stable.build_stairs())
    e = count_params(stable.build_elevator())
    assert s == e  # the shortcut adds no parameters


def test_every_horse_has_card_text():
    for h in stable.STABLE:
        assert h.name.isupper() or h.name.startswith("THE ")
        assert len(h.tagline) > 20
        assert h.colour.startswith("#") and len(h.colour) == 7
