# backend/api_model_specs.py
# Version: V3.0 / deepseek edit - 2026-07-15

"""
Canonical video-generation capability specs used by backend and frontend.
این فایل شامل مشخصات کامل مدل‌های ویدیویی پشتیبانی شده است.
"""

from __future__ import annotations

from api_types import (
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    LTXVideoGenerationModelSpecItem,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenResolution,
)
from runtime_config.model_download_specs import (
    ALL_LTX_LOCAL_MODEL_IDS,
    LTXLocalModelRelevant,
    get_ltx_model_spec,
)


def _resolution_spec(
    *,
    fps_to_durations: dict[LTXVideoGenFps, tuple[LTXVideoGenDuration, ...]],
) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            fps: list(durations)
            for fps, durations in fps_to_durations.items()
        }
    )


# ============================================================
# تنظیمات durations برای همه مدل‌ها (۱ تا ۳۰ ثانیه)
# ============================================================
ALL_DURATIONS_1_TO_30 = tuple(range(1, 31))  # (1, 2, 3, ..., 30)


# ============================================================
# مشخصات کامل مدل‌ها بر اساس درخواست شما
# ============================================================
ltx_api_model_specs: tuple[
    tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...
] = (
    # ========== مدل FAST ==========
    (
        "fast",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (Fast)",
            supported_resolutions_durations={
                "4k": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        25: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "2k": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        25: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "1080p": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        25: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                        50: ALL_DURATIONS_1_TO_30,
                        60: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "720p": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                        50: ALL_DURATIONS_1_TO_30,
                        60: ALL_DURATIONS_1_TO_30,
                        120: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "480p": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                        50: ALL_DURATIONS_1_TO_30,
                        60: ALL_DURATIONS_1_TO_30,
                        120: ALL_DURATIONS_1_TO_30,
                        240: ALL_DURATIONS_1_TO_30,
                    }
                ),
            },
            a2v_supported_resolutions_durations=None
        )
    ),
    # ========== مدل FAST HQ ==========
    (
        "fast_hq",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (Fast HQ)",
            supported_resolutions_durations={
                "4k": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        25: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "2k": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        25: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "1080p": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        25: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                        50: ALL_DURATIONS_1_TO_30,
                        60: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "720p": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                        50: ALL_DURATIONS_1_TO_30,
                        60: ALL_DURATIONS_1_TO_30,
                        120: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "480p": _resolution_spec(
                    fps_to_durations={
                        15: ALL_DURATIONS_1_TO_30,
                        24: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                        50: ALL_DURATIONS_1_TO_30,
                        60: ALL_DURATIONS_1_TO_30,
                        120: ALL_DURATIONS_1_TO_30,
                        240: ALL_DURATIONS_1_TO_30,
                    }
                ),
            },
            a2v_supported_resolutions_durations=None
        )
    ),
    # ========== مدل PRO ==========
    (
        "pro",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (PRO)",
            supported_resolutions_durations={
                "2k": _resolution_spec(
                    fps_to_durations={
                        24: ALL_DURATIONS_1_TO_30,
                        25: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "1080p": _resolution_spec(
                    fps_to_durations={
                        24: ALL_DURATIONS_1_TO_30,
                        25: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "720p": _resolution_spec(
                    fps_to_durations={
                        24: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                        50: ALL_DURATIONS_1_TO_30,
                        60: ALL_DURATIONS_1_TO_30,
                    }
                ),
                "480p": _resolution_spec(
                    fps_to_durations={
                        24: ALL_DURATIONS_1_TO_30,
                        30: ALL_DURATIONS_1_TO_30,
                        50: ALL_DURATIONS_1_TO_30,
                        60: ALL_DURATIONS_1_TO_30,
                        120: ALL_DURATIONS_1_TO_30,
                    }
                ),
            },
            a2v_supported_resolutions_durations=None
        )
    )
)


def _pairs_to_items(
    pairs: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...],
) -> list[LTXVideoGenerationModelSpecItem]:
    return [
        LTXVideoGenerationModelSpecItem(pipeline=pipeline, spec=spec)
        for pipeline, spec in pairs
    ]


def get_local_video_generation_model_specs() -> list[LTXVideoGenerationModelSpecItem]:
    """دریافت همه مدل‌های محلی موجود برای تولید ویدیو."""
    items: list[LTXVideoGenerationModelSpecItem] = []
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        spec = get_ltx_model_spec(model_id)
        if isinstance(spec.relevance, LTXLocalModelRelevant):
            items.extend(_pairs_to_items(spec.supported_pipelines))
    return items


def get_api_video_generation_model_specs() -> list[LTXVideoGenerationModelSpecItem]:
    """دریافت مدل‌های API برای تولید ویدیو."""
    return _pairs_to_items(ltx_api_model_specs)


def build_generate_video_model_specs_response() -> GenerateVideoModelsSpecsResponse:
    """ساخت پاسخ کامل مشخصات مدل‌ها برای API."""
    return GenerateVideoModelsSpecsResponse(
        local_models=get_local_video_generation_model_specs(),
        api_models=get_api_video_generation_model_specs()
    )


def _get_resolution_spec(
    item: LTXVideoGenerationModelSpecItem,
    *,
    resolution: LTXVideoGenResolution,
    is_a2v: bool,
) -> LTXVideoGenerationResolutionSpec | None:
    if is_a2v and item.spec.a2v_supported_resolutions_durations is not None:
        resolution_map = item.spec.a2v_supported_resolutions_durations
    else:
        resolution_map = item.spec.supported_resolutions_durations
    return resolution_map.get(resolution)


def get_supported_durations(
    resolution_spec: LTXVideoGenerationResolutionSpec,
    fps: LTXVideoGenFps,
) -> list[LTXVideoGenDuration]:
    return resolution_spec.fps_to_durations.get(fps, [])


def validate_generate_video_request(
    req: GenerateVideoRequest,
    use_api_specs: bool,
) -> str | None:
    """اعتبارسنجی درخواست تولید ویدیو بر اساس مشخصات مدل."""
    items = (
        get_api_video_generation_model_specs()
        if use_api_specs
        else get_local_video_generation_model_specs()
    )
    
    item = next(
        (candidate for candidate in items if candidate.pipeline == req.model),
        None,
    )
    
    generation_backend = "api" if use_api_specs else "local"
    generation_mode = (
        "audio-to-video"
        if req.audioPath is not None
        else "image-to-video"
        if req.imagePath is not None
        else "text-to-video"
    )
    
    if item is None:
        return (
            f"Unsupported {generation_backend} video generation pipeline: {req.model}. "
            f"Supported pipelines: {', '.join(candidate.pipeline for candidate in items)}"
        )
        
    resolution_spec = _get_resolution_spec(
        item,
        resolution=req.resolution,
        is_a2v=req.audioPath is not None,
    )
    
    if resolution_spec is None:
        return (
            f"Unsupported {generation_backend} {generation_mode} resolution "
            f"'{req.resolution}' for pipeline '{req.model}'"
        )
        
    if req.fps not in resolution_spec.fps_to_durations:
        return (
            f"Unsupported {generation_backend} {generation_mode} fps '{req.fps}' "
            f"for pipeline '{req.model}' at resolution '{req.resolution}'"
        )
        
    supported_durations = get_supported_durations(resolution_spec, fps=req.fps)
    if req.duration not in supported_durations:
        return (
            f"Unsupported {generation_backend} {generation_mode} duration "
            f"'{req.duration}' for pipeline '{req.model}' at resolution "
            f"'{req.resolution}' and fps '{req.fps}'"
        )
        
    return None
