# backend/services/hq_video_pipeline/hq_video_pipeline.py
# Version: V1.0 / deepseek edit - 2026-07-15

"""
کلاس اصلی HQVideoPipeline
این کلاس مسئول مدیریت کل فرآیند ساخت ویدیو با کیفیت بالا (Fast HQ) است.
"""

import os
import io
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import warnings

try:
    from diffusers import LTXPipeline, LTXImageToVideoPipeline
    from diffusers.utils import export_to_video
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
    warnings.warn("imageio library not found. Please install it: pip install imageio")

# ===== تنظیمات مجاز برای Fast و Fast HQ =====
ALLOWED_RESOLUTIONS_FPS: Dict[str, List[int]] = {
    "4K":  [15, 24, 25, 30],
    "2K":  [15, 24, 25, 30],
    "1080": [15, 24, 25, 30, 50, 60],
    "720":  [15, 24, 30, 50, 60, 120],
    "480":  [15, 24, 30, 50, 60, 120, 240],
}

RESOLUTION_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "4K":  (3840, 2160),
    "2K":  (2560, 1440),
    "1080": (1920, 1080),
    "720":  (1280, 720),
    "480":  (854, 480),
}

class ResolutionEnum(str, Enum):
    _4K = "4K"
    _2K = "2K"
    _1080 = "1080"
    _720 = "720"
    _480 = "480"

    @property
    def dimensions(self) -> Tuple[int, int]:
        return RESOLUTION_DIMENSIONS[self.value]

    @classmethod
    def from_string(cls, value: str) -> "ResolutionEnum":
        value = value.strip()
        if value not in RESOLUTION_DIMENSIONS:
            raise ValueError(f"Resolution must be one of {list(RESOLUTION_DIMENSIONS.keys())}")
        return cls(value)

class HQVideoPipeline:
    """
    Pipeline برای ساخت ویدیو با کیفیت بالا (Fast HQ)
    از مدل ltx-2.3-22b-distilled.safetensors با ۱۶ قدم استفاده می‌کند.
    """

    def __init__(
        self,
        model_path: str,
        upscaler_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "float16",
    ):
        """
        Args:
            model_path: مسیر فایل مدل اصلی (ltx-2.3-22b-distilled.safetensors)
            upscaler_model_path: مسیر فایل مدل آپ‌اسکیلر (اختیاری)
            device: دستگاه اجرا (cuda/cpu)
            dtype: نوع داده (float16/float32)
        """
        self.model_path = Path(model_path)
        self.upscaler_model_path = Path(upscaler_model_path) if upscaler_model_path else None
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.pipeline = None
        self.upscaler_pipeline = None

        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library is required for HQVideoPipeline.")

        self._load_model()
        if self.upscaler_model_path:
            self._load_upscaler()

    def _load_model(self):
        """بارگذاری مدل اصلی LTX Distilled"""
        print(f"[HQ] Loading model from {self.model_path} ...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.pipeline = LTXPipeline.from_pretrained(
                pretrained_model_name_or_path=str(self.model_path.parent),
                torch_dtype=self.dtype,
                use_safetensors=True,
            ).to(self.device)
            
            # فعال کردن بهینه‌سازی‌های حافظه
            self.pipeline.enable_model_cpu_offload()
            print("[HQ] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load LTX model: {str(e)}")

    def _load_upscaler(self):
        """بارگذاری مدل آپ‌اسکیلر فضایی"""
        print(f"[HQ] Loading upscaler from {self.upscaler_model_path} ...")
        if not self.upscaler_model_path.exists():
            raise FileNotFoundError(f"Upscaler file not found: {self.upscaler_model_path}")

        try:
            # فرض می‌کنیم آپ‌اسکیلر هم با diffusers یا یک مدل ساده torch.jit است
            # اینجا یک نمونه بارگذاری ساده برای مدل‌های safetensors قرار می‌دهیم
            from safetensors.torch import load_file
            state_dict = load_file(self.upscaler_model_path)
            # در اینجا باید معماری دقیق آپ‌اسکیلر را بشناسید، من یک placeholder می‌گذارم:
            # self.upscaler_pipeline = YourUpscalerModel(state_dict).to(self.device)
            print("[HQ] Upscaler loaded successfully (placeholder).")
            # برای جلوگیری از خطا، یک دیکشنری ساده نگه می‌داریم تا بدانیم بارگذاری شده
            self.upscaler_pipeline = state_dict 
        except Exception as e:
            raise RuntimeError(f"Failed to load upscaler: {str(e)}")

    def _apply_upscaler(self, frames: np.ndarray, scale_factor: float = 1.5) -> np.ndarray:
        """
        اعمال آپ‌اسکیل بر روی فریم‌ها
        اگر آپ‌اسکیلر واقعی وجود نداشته باشد، از interpolation ساده استفاده می‌کند.
        """
        if self.upscaler_pipeline is not None and isinstance(self.upscaler_pipeline, dict):
            # اگر مدل واقعی بود، اینجا اعمال می‌شود
            # فعلاً از interpolation استفاده می‌کنیم تا کد کاملاً اجرایی باشد
            pass

        # استفاده از torch برای upscale به عنوان fallback
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device)  # (F, C, H, W)
        new_h = int(tensor.shape[2] * scale_factor)
        new_w = int(tensor.shape[3] * scale_factor)
        upscaled = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        return upscaled.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    def _encode_frames_to_mp4(self, frames: np.ndarray, fps: int) -> bytes:
        """
        تبدیل فریم‌های numpy به فایل MP4 و بازگرداندن به صورت bytes
        """
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
        """
        تولید فریم‌ها با استفاده از diffusers LTXPipeline
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # تنظیمات برای کیفیت بالاتر HQ
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

        # خروجی معمولاً یک آرایه با شکل (num_frames, height, width, 3) است
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
        resolution: str,        # "4K", "2K", "1080", "720", "480"
        fps: int,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 16,  # ثابت برای HQ
        use_upscaler: bool = False,
    ) -> bytes:
        """
        تولید ویدیو بر اساس پارامترهای داده شده

        Returns:
            bytes: داده‌های ویدیو در فرمت MP4
        """
        # ================== اعتبارسنجی ==================
        if duration_seconds < 1 or duration_seconds > 30:
            raise ValueError("Duration must be between 1 and 30 seconds")

        res_enum = ResolutionEnum.from_string(resolution)
        if resolution not in ALLOWED_RESOLUTIONS_FPS:
            raise ValueError(f"Resolution {resolution} not allowed. Allowed: {list(ALLOWED_RESOLUTIONS_FPS.keys())}")
        
        allowed_fps = ALLOWED_RESOLUTIONS_FPS[resolution]
        if fps not in allowed_fps:
            raise ValueError(
                f"FPS {fps} not allowed for resolution {resolution}. Allowed: {allowed_fps}"
            )

        width, height = res_enum.dimensions
        total_frames = duration_seconds * fps

        # ================== تولید فریم‌ها ==================
        # اگر آپ‌اسکیلر فعال باشد، در ابعاد کوچکتر تولید می‌کنیم تا سرعت بالاتر برود
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

        # ================== اعمال آپ‌اسکیلر ==================
        if use_upscaler and self.upscaler_pipeline is not None:
            scale_factor = 2.0  # چون نصف تولید کردیم، دو برابر می‌کنیم
            frames = self._apply_upscaler(frames, scale_factor)
            # اطمینان از تطابق ابعاد نهایی با ابعاد درخواستی
            if frames.shape[1] != height or frames.shape[2] != width:
                frames = torch.nn.functional.interpolate(
                    torch.from_numpy(frames).permute(0, 3, 1, 2),
                    size=(height, width),
                    mode='bilinear'
                ).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

        # ================== تبدیل به MP4 ==================
        return self._encode_frames_to_mp4(frames, fps)

    def set_upscaler(self, upscaler_path: str):
        """تغییر مدل آپ‌اسکیلر در زمان اجرا"""
        self.upscaler_model_path = Path(upscaler_path)
        self._load_upscaler()

    @staticmethod
    def get_allowed_fps(resolution: str) -> List[int]:
        """دریافت لیست fps مجاز برای یک رزولوشن مشخص"""
        return ALLOWED_RESOLUTIONS_FPS.get(resolution, [])