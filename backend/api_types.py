# backend/api_types.py
# Version: V3.0 / deepseek edit - 2026-07-15

"""
Pydantic request/response models and typed aliases for ltx2_server.
این فایل شامل تعاریف نوع‌های داده برای API می‌باشد.
"""

from __future__ import annotations

from typing import Annotated, Literal, NamedTuple, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


# ============================================================
# Type Aliases پایه
# ============================================================
NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
JsonObject: TypeAlias = dict[str, object]


# ============================================================
# Enums و Literals
# ============================================================
ModelCheckpointID = Literal[
    "ltx-2.3-22b-distilled",
    "ltx-2.3-22b-dev",
    "ltx-2.3-spatial-upscaler-x2-1.0",
    "ltx-2.3-spatial-upscaler-x1.5-1.0",
    "ltx-2.3-spatial-upscaler-x2-1.1",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "dpt-hybrid-midas",
    "yolox-l-torchscript",
    "dw-ll-ucoco-384-bs5",
    "gemma-3-12b-it-qat-q4_0-unquantized",
    "z-image-turbo",
]

LTXLocalModelId = Literal["ltx-2.3-22b-distilled", "ltx-2.3-22b-dev"]

# ============================================================
# تنظیمات ویدیو
# ============================================================
LTXVideoGenResolution = Literal["480p", "720p", "1080p", "1440p", "2160p", "2k", "4k"]
LTXVideoGenDuration = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
]
LTXVideoGenFps = Literal[15, 24, 25, 30, 48, 50, 60, 120, 240]
LTXVideoGenPipeline = Literal["fast", "fast_hq", "pro"]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""
    path: str
    frame_idx: int
    strength: float


VideoCameraMotion = Literal[
    "none",
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "jib_up",
    "jib_down",
    "static",
    "focus_shift",
]


# ============================================================
# مدل‌های مشخصات (Specs)
# ============================================================
class LTXVideoGenerationResolutionSpec(BaseModel):
    """مشخصات یک رزولوشن خاص: نگاشت FPS به لیست durations مجاز"""
    fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]]


class LTXVideoGenerationSpec(BaseModel):
    """مشخصات کامل یک مدل برای تولید ویدیو"""
    display_name: str
    supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec]
    a2v_supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None = None


class LTXVideoGenerationModelSpecItem(BaseModel):
    """یک آیتم از لیست مشخصات مدل‌ها برای پاسخ API"""
    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


# ============================================================
# مدل‌های Response
# ============================================================
class ModelStatusItem(BaseModel):
    id: str
    name: str
    loaded: bool
    downloaded: bool


class GpuTelemetry(BaseModel):
    name: str
    vram: int
    vramUsed: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    models_loaded: bool
    active_model: str | None
    gpu_info: GpuTelemetry
    sage_attention: bool
    models_status: list[ModelStatusItem]


class GpuInfoResponse(BaseModel):
    cuda_available: bool
    mps_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None
    vram_gb: int | None
    gpu_info: GpuTelemetry


class RuntimePolicyResponse(BaseModel):
    force_api_generations: bool


class GenerationProgressResponse(BaseModel):
    status: Literal["idle", "running", "complete", "cancelled", "error"]
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None


class DownloadProgressRunningResponse(BaseModel):
    status: Literal["downloading"]
    current_downloading_file: ModelCheckpointID | None
    current_file_progress: float
    total_progress: float
    total_downloaded_bytes: int
    expected_total_bytes: int
    completed_files: set[ModelCheckpointID]
    all_files: set[ModelCheckpointID]
    error: None = None
    speed_bytes_per_sec: float


class DownloadProgressCompleteResponse(BaseModel):
    status: Literal["complete"]


class DownloadProgressErrorResponse(BaseModel):
    status: Literal["error"]
    error: str


DownloadProgressResponse: TypeAlias = (
    DownloadProgressRunningResponse |
    DownloadProgressCompleteResponse |
    DownloadProgressErrorResponse
)


class SuggestGapPromptResponse(BaseModel):
    status: Literal["success"] = "success"
    suggested_prompt: str


class GenerateVideoCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str


class GenerateVideoCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateVideoResponse: TypeAlias = GenerateVideoCompleteResponse | GenerateVideoCancelledResponse


class GenerateImageCompleteResponse(BaseModel):
    status: Literal["complete"]
    image_paths: list[str]


class GenerateImageCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateImageResponse: TypeAlias = GenerateImageCompleteResponse | GenerateImageCancelledResponse


class CancelCancellingResponse(BaseModel):
    status: Literal["cancelling"]


# ============================================================
# مدل‌های Request
# ============================================================
class GenerateVideoModelsSpecsResponse(BaseModel):
    local_models: list[LTXVideoGenerationModelSpecItem]
    api_models: list[LTXVideoGenerationModelSpecItem]
    upscalers: list[ModelCheckpointID] = [
        "ltx-2.3-spatial-upscaler-x1.5-1.0",
        "ltx-2.3-spatial-upscaler-x2-1.1",
    ]


class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    
    prompt: NonEmptyPrompt
    resolution: LTXVideoGenResolution = "1080p"
    model: LTXVideoGenPipeline = "fast"
    upscaler: ModelCheckpointID = "ltx-2.3-spatial-upscaler-x1.5-1.0"
    cameraMotion: VideoCameraMotion = "none"
    duration: LTXVideoGenDuration = 5
    fps: LTXVideoGenFps = 24
    cfgScale: float = Field(default=7.0, ge=1.0, le=20.0)
    steps: int = Field(default=8, ge=1, le=100)
    seed: int | None = None
    imageConditioning: ImageConditioningInput | None = None
    imagePath: str | None = None
    audioPath: str | None = None


class GenerateTextRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt: NonEmptyPrompt


class GenerateTextResponse(BaseModel):
    text: str


class GenerateImageRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt: NonEmptyPrompt
    negativePrompt: str = ""
    imagePath: str | None = None
    aspectRatio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = "1:1"


class GenerateImageResponse(BaseModel):
    imagePath: str


# ============================================================
# Error Response
# ============================================================
class HTTPErrorResponse(BaseModel):
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")
