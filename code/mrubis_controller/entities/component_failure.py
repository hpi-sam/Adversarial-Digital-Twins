import enum
from typing import List

class ComponentFailure(enum.Enum):
    GOOD = "good"
    CF1 = "CF1"
    CF2 = "CF2"
    CF3 = "CF3"
    CF5 = "CF5"

    @classmethod
    def list(self) -> List[str]:
        return [status.value for status in ComponentFailure]