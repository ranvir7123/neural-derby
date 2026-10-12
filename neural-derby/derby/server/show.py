"""The six acts as a state machine. Space on the stadium page calls advance()."""
from enum import Enum


class Act(str, Enum):
    LOBBY = "LOBBY"
    PADDOCK = "PADDOCK"
    MULTIPLY = "MULTIPLY"
    HORSES = "HORSES"
    DERBY = "DERBY"
    SPRINT = "SPRINT"
    VERDICT = "VERDICT"
    SHIP = "SHIP"


ORDER = list(Act)


class GateError(Exception):
    pass


class Show:
    def __init__(self):
        self.act = Act.LOBBY
        self.photos_open = False
        self.bets_open = False
        self.data_loaded = False

    def advance(self) -> Act:
        i = ORDER.index(self.act)
        if i + 1 >= len(ORDER):
            raise GateError("The show is over.")
        nxt = ORDER[i + 1]
        if nxt == Act.DERBY and not self.data_loaded:
            raise GateError("Photos are still loading onto the GPU. Try again in a moment.")
        self.act = nxt
        self.photos_open = nxt == Act.PADDOCK
        self.bets_open = nxt == Act.HORSES
        return nxt

    def close_gate(self) -> None:
        self.photos_open = False
        self.bets_open = False

    def to_dict(self) -> dict:
        return {"act": self.act.value, "photos_open": self.photos_open,
                "bets_open": self.bets_open, "data_loaded": self.data_loaded}
