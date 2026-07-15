# backend/services/pro_video_pipeline/ltx_pro_video_pipeline.py
# Version: V1.0 / deepseek edit - 2026-07-15

"""
پیاده‌سازی اختصاصی برای مدل LTX در حالت PRO
از مدل ltx-2.3-22b-dev.safetensors استفاده می‌کند.
"""

from .pro_video_pipeline import PROVideoPipeline
from typing import Optional

class LTXPROVideoPipeline(PROVideoPipeline):
    """
    کلاس ویژه برای مدل dev (PRO)
    """

    def __init__(
        self, 
        model_path: str, 
        upscaler_model_path: str = None, 
        device: str = "cuda", 
        dtype: str = "float16"
    ):
        super().__init__(model_path, upscaler_model_path, device, dtype)
        self.default_steps = 30
        self.model_display_name = "LTX 2.3 (PRO)"
        self.model_file = "ltx-2.3-22b-dev.safetensors"

    def generate_video(
        self,
        prompt: str,
        duration_seconds: int,
        resolution: str,
        fps: int,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        use_upscaler: bool = False,
    ) -> bytes:
        """
        override برای تنظیمات پیش‌فرض PRO
        """
        return super().generate_video(
            prompt=prompt,
            duration_seconds=duration_seconds,
            resolution=resolution,
            fps=fps,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            use_upscaler=use_upscaler,
        )