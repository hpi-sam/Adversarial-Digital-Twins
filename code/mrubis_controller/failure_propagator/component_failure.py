import enum
from typing import List

class ComponentFailure(enum.Enum):
    GOOD = "good"
    CF1 = "cf1"
    CF2 = "cf2"
    CF3 = "cf3"
    CF5 = "cf5"

    @classmethod
    def list() -> List[str]:
        return [status.value for status in ComponentFailure]