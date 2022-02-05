import enum
from typing import List

class Fixes(enum.Enum):
    HW_REDEPLOY_COMPONENT = "HwRedeployComponent"
    #REPLACE_COMPONENT = "ReplaceComponent"
    RESTART_COMPONENT = "RestartComponent"
    LW_REDEPLOY_COMPONENT = "LwRedeployComponent"

    @classmethod
    def list(self) -> List[str]:
        return [fix.value for fix in Fixes]