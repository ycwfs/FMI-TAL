from dataclasses import dataclass


@dataclass
class params:
    train: dict
    test: dict

@dataclass
class paths:
    log: str