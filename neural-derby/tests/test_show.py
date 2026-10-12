import pytest

from derby.server.show import Act, GateError, Show


def test_acts_in_order_with_gates():
    s = Show()
    assert s.act == Act.LOBBY and not s.photos_open and not s.bets_open
    assert s.advance() == Act.PADDOCK and s.photos_open
    s.close_gate()
    assert not s.photos_open
    assert s.advance() == Act.MULTIPLY
    assert s.advance() == Act.HORSES and s.bets_open
    with pytest.raises(GateError):
        s.advance()                     # DERBY needs data_loaded
    s.data_loaded = True
    assert s.advance() == Act.DERBY and not s.bets_open
    assert s.advance() == Act.SPRINT
    assert s.advance() == Act.VERDICT
    assert s.advance() == Act.SHIP
    with pytest.raises(GateError):
        s.advance()


def test_to_dict():
    d = Show().to_dict()
    assert d == {"act": "LOBBY", "photos_open": False, "bets_open": False, "data_loaded": False}
