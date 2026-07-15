# backend/services/pro_video_pipeline/pro_video_pipeline.py
# Version: V1.0 / deepseek edit - 2026-07-15

"""
کلاس اصلی PROVideoPipeline
برای مدل ltx-2.3-22b-dev.safetensors با تنظیمات خاص و محدودیت‌های فریم‌ریت.
"""

import os
import io
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from enum import Enum
from pathlib import Path
import warnings

try:
    from diffusers import LTXPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    LTXPipeline = None
    warnings.warn("diffusers library not found. Please install it: pip install diffusers")

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# ===== تنظیمات مجاز برای PRO (متفاوت از Fast) =====
ALLOWED_RESOLUTIONS_FPS_PRO: Dict[str, List[int]] = {
    "2K":  [24, 25, 30],
    "1080": [24, 25, 30],
    "720":  [24, 30, 50, 60],
    "480":  [24, 30, 50, 60, 120],
}

RESOLUTION_DIMENSIONS_PRO: Dict[str, Tuple[int, int]] = {
    "2K":  (2560, 1440),
    "1080": (1920, 1080),
    "720":  (1280, 720),
    "480":  (854, 480),
}

class ResolutionProEnum(str, Enum):
    _2K = "2K"
    _1080 = "1080"
    _720 = "720"
    _480 = "480"

    @property
    def dimensions(self) -> Tuple[int, int]:
        return RESOLUTION_DIMENSIONS_PRO[self.value]

    @classmethod
    def from_string(cls, value: str) -> "ResolutionProEnum":
        value = value.strip()
        if value not in RESOLUTION_DIMENSIONS_PRO:
            raise ValueError(f"Resolution must be one of {list(RESOLUTION_DIMENSIONS_PRO.keys())}")
        return cls(value)

class PROVideoPipeline:
    """
    Pipeline برای ساخت ویدیو با کیفیت PRO
    از مدل ltx-2.3-22b-dev.safetensors استفاده می‌کند.
    """

    def __init__(
        self,
        model_path: str,
        upscaler_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "float16",
    ):
        self.model_path = Path(model_path)
        self.upscaler_model_path = Path(upscaler_model_path) if upscaler_model_path else None
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.pipeline = None
        self.upscaler_pipeline = None

        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library is required for PROVideoPipeline.")

        self._load_model()
        if self.upscaler_model_path:
            self._load_upscaler()

    def _load_model(self):
        """بارگذاری مدل PRO (dev)"""
        print(f"[PRO] Loading model from {self.model_path} ...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.pipeline = LTXPipeline.from_pretrained(
                pretrained_model_name_or_path=str(self.model_path.parent),
                torch_dtype=self.dtype,
                use_safetensors=True,
            ).to(self.device)
            self.pipeline.enable_model_cpu_offload()
            print("[PRO] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load LTX PRO model: {str(e)}")

    def _load_upscaler(self):
        """بارگذاری مدل آپ‌اسکیلر"""
        print(f"[PRO] Loading upscaler from {self.upscaler_model_path} ...")
        if not self.upscaler_model_path.exists():
            raise FileNotFoundError(f"Upscaler file not found: {self.upscaler_model_path}")
        # همانند HQ، placeholder برای بارگذاری
        from safetensors.torch import load_file
        self.upscaler_pipeline = load_file(self.upscaler_model_path)
        print("[PRO] Upscaler loaded successfully (placeholder).")

    def _apply_upscaler(self, frames: np.ndarray, scale_factor: float = 1.5) -> np.ndarray:
        """اعمال آپ‌اسکیل با fallback torch"""
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device)
        new_h = int(tensor.shape[2] * scale_factor)
        new_w = int(tensor.shape[3] * scale_factor)
        upscaled = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        return upscaled.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    def _encode_frames_to_mp4(self, frames: np.ndarray, fps: int) -> bytes:
        """تبدیل فریم‌ها به MP4 bytes"""
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio library is required to encode video.")
        if frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        with io.BytesIO() as output_buffer:
            writer = imageio.get_writer(
                output_buffer,
                format='FFMPEG',
                mode='I',
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p',
            )
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            return output_buffer.getvalue()

    def _generate_with_ltx(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        total_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> np.ndarray:
        """تولید فریم با LTXPipeline"""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=total_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np",
        )
        if hasattr(output, 'frames'):
            return output.frames
        elif isinstance(output, np.ndarray):
            return output
        else:
            raise TypeError("Unknown output type from LTX pipeline")

    def generate_video(
        self,
        prompt: str,
        duration_seconds: int,
        resolution: str,
        fps: int,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,  # پیش‌فرض PRO
        use_upscaler: bool = False,
    ) -> bytes:
        """
        تولید ویدیو با مدل PRO
        """
        # ================== اعتبارسنجی (محدودیت‌های خاص PRO) ==================
        if duration_seconds < 1 or duration_seconds > 30:
            raise ValueError("Duration must be between 1 and 30 seconds")

        if resolution not in ALLOWED_RESOLUTIONS_FPS_PRO:
            raise ValueError(f"Resolution {resolution} not allowed for PRO. Allowed: {list(ALLOWED_RESOLUTIONS_FPS_PRO.keys())}")
        
        allowed_fps = ALLOWED_RESOLUTIONS_FPS_PRO[resolution]
        if fps not in allowed_fps:
            raise ValueError(
                f"FPS {fps} not allowed for resolution {resolution} in PRO. Allowed: {allowed_fps}"
            )

        res_enum = ResolutionProEnum.from_string(resolution)
        width, height = res_enum.dimensions
        total_frames = duration_seconds * fps

        # ================== تولید ==================
        gen_width = width // 2 if use_upscaler else width
        gen_height = height // 2 if use_upscaler else height

        frames = self._generate_with_ltx(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=gen_width,
            height=gen_height,
            total_frames=total_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        if use_upscaler and self.upscaler_pipeline is not None:
            scale_factor = 2.0
            frames = self._apply_upscaler(frames, scale_factor)
            if frames.shape[1] != height or frames.shape[2] != width:
                frames = torch.nn.functional.interpolate(
                    torch.from_numpy(frames).permute(0, 3, 1, 2),
                    size=(height, width),
                    mode='bilinear'
                ).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

        return self._encode_frames_to_mp4(frames, fps)

    def set_upscaler(self, upscaler_path: str):
        self.upscaler_model_path = Path(upscaler_path)
        self._load_upscaler()

    @staticmethod
    def get_allowed_fps(resolution: str) -> List[int]:
        return ALLOWED_RESOLUTIONS_FPS_PRO.get(resolution, [])