"""Host-owned contracts and orchestration for approved real V2 profiles."""

from .contracts import (
    PROFILE_SCHEDULES,
    RealEvolutionCheckpointV2,
    RealEvolutionManifestV2,
    RealInputFileV2,
    RealModelBindingV2,
    RealRunResultV2,
    RealRuntimeLocationV2,
    RealScheduleV2,
    RealStageRecordV2,
)
from .runner import (
    RealRunnerError,
    RealStageContextV2,
    RealStagePorts,
    SealedStageV2,
    run_real_evolution,
)

__all__ = [
    "PROFILE_SCHEDULES", "RealScheduleV2", "RealModelBindingV2",
    "RealInputFileV2", "RealRuntimeLocationV2", "RealEvolutionManifestV2",
    "RealEvolutionCheckpointV2", "RealStageRecordV2", "RealRunResultV2",
    "RealRunnerError", "RealStageContextV2", "RealStagePorts", "SealedStageV2",
    "run_real_evolution",
]
