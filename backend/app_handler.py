# backend/app_handler.py
# Version: V3.0 / deepseek edit - 2026-07-15

"""
Application state composition root and dependency wiring.
این فایل مسئول سیم‌کشی وابستگی‌ها و ساخت handlers برنامه است.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from state.app_settings import AppSettings
from handlers import (
    DownloadHandler,
    GenerationHandler,
    HealthHandler,
    HuggingFaceAuthHandler,
    IcLoraHandler,
    ImageGenerationHandler,
    ModelsHandler,
    PipelinesHandler,
    SuggestGapPromptHandler,
    RetakeHandler,
    RuntimePolicyHandler,
    SettingsHandler,
    TextHandler,
    VideoGenerationHandler,
)
from runtime_config.runtime_config import RuntimeConfig
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    HQVideoPipeline,
    ProVideoPipeline,
    ZitAPIClient,
    ImageGenerationPipeline,
    GpuCleaner,
    GpuInfo,
    HTTPClient,
    IcLoraPipeline,
    LTXAPIClient,
    ModelDownloader,
    PoseProcessorPipeline,
    RetakePipeline,
    TaskRunner,
    TextEncoder,
    VideoProcessor,
)
from state.app_state_types import AppState, TextEncoderState


class AppHandler:
    """
    Composition-only state service exposing typed domain handlers.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        default_settings: AppSettings,
        http: HTTPClient,
        gpu_cleaner: GpuCleaner,
        model_downloader: ModelDownloader,
        gpu_info: GpuInfo,
        video_processor: VideoProcessor,
        text_encoder: TextEncoder,
        task_runner: TaskRunner,
        ltx_api_client: LTXAPIClient,
        zit_api_client: ZitAPIClient,
        fast_video_pipeline_class: type[FastVideoPipeline],
        hq_video_pipeline_class: type[HQVideoPipeline],
        pro_video_pipeline_class: type[ProVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
    ) -> None:
        self.config = config
        self.http = http
        self.gpu_cleaner = gpu_cleaner
        self.model_downloader = model_downloader
        self.gpu_info = gpu_info
        self.video_processor = video_processor
        self.task_runner = task_runner
        self.ltx_api_client = ltx_api_client
        self.zit_api_client = zit_api_client
        
        # Pipeline classes
        self.fast_video_pipeline_class = fast_video_pipeline_class
        self.hq_video_pipeline_class = hq_video_pipeline_class
        self.pro_video_pipeline_class = pro_video_pipeline_class
        self.image_generation_pipeline_class = image_generation_pipeline_class
        self.ic_lora_pipeline_class = ic_lora_pipeline_class
        self.depth_processor_pipeline_class = depth_processor_pipeline_class
        self.pose_processor_pipeline_class = pose_processor_pipeline_class
        self.a2v_pipeline_class = a2v_pipeline_class
        self.retake_pipeline_class = retake_pipeline_class
        
        self._lock = threading.RLock()
        
        self.state = AppState(
            downloading_session=None,
            gpu_slot=None,
            active_generation=None,
            cpu_slot=None,
            text_encoder=TextEncoderState(service=text_encoder),
            app_settings=default_settings.model_copy(deep=True),
        )

        # ============================================================
        # Handlers (wired in dependency order)
        # ============================================================
        self.settings = SettingsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )
        
        self.models = ModelsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )
        
        self.hf_auth = HuggingFaceAuthHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )
        
        self.downloads = DownloadHandler(
            state=self.state,
            lock=self._lock,
            models_handler=self.models,
            model_downloader=model_downloader,
            task_runner=task_runner,
            config=config,
        )
        
        self.text = TextHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )
        
        self.pipelines = PipelinesHandler(
            state=self.state,
            lock=self._lock,
            text_handler=self.text,
            gpu_cleaner=gpu_cleaner,
            fast_video_pipeline_class=fast_video_pipeline_class,
            hq_video_pipeline_class=hq_video_pipeline_class,
            pro_video_pipeline_class=pro_video_pipeline_class,
            image_generation_pipeline_class=image_generation_pipeline_class,
            ic_lora_pipeline_class=ic_lora_pipeline_class,
            depth_processor_pipeline_class=depth_processor_pipeline_class,
            pose_processor_pipeline_class=pose_processor_pipeline_class,
            a2v_pipeline_class=a2v_pipeline_class,
            retake_pipeline_class=retake_pipeline_class,
            config=config,
        )
        
        self.runtime_policy = RuntimePolicyHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )
        
        self.health = HealthHandler(
            state=self.state,
            lock=self._lock,
            gpu_info=gpu_info,
            config=config,
        )
        
        self.generation = GenerationHandler(
            state=self.state,
            lock=self._lock,
            task_runner=task_runner,
            video_processor=video_processor,
            config=config,
        )
        
        self.video_generation = VideoGenerationHandler(
            state=self.state,
            lock=self._lock,
            pipelines_handler=self.pipelines,
            generation_handler=self.generation,
            config=config,
        )
        
        self.image_generation = ImageGenerationHandler(
            state=self.state,
            lock=self._lock,
            pipelines_handler=self.pipelines,
            generation_handler=self.generation,
            config=config,
        )
        
        self.ic_lora = IcLoraHandler(
            state=self.state,
            lock=self._lock,
            pipelines_handler=self.pipelines,
            generation_handler=self.generation,
            config=config,
        )
        
        self.retake = RetakeHandler(
            state=self.state,
            lock=self._lock,
            pipelines_handler=self.pipelines,
            generation_handler=self.generation,
            config=config,
        )
        
        self.suggest_gap_prompt = SuggestGapPromptHandler(
            state=self.state,
            lock=self._lock,
            ltx_api_client=ltx_api_client,
            zit_api_client=zit_api_client,
            config=config,
        )


@dataclass
class ServiceBundle:
    """مجموعه سرویس‌های مورد نیاز برای ساخت AppHandler"""
    http: HTTPClient
    gpu_cleaner: GpuCleaner
    model_downloader: ModelDownloader
    gpu_info: GpuInfo
    video_processor: VideoProcessor
    text_encoder: TextEncoder
    task_runner: TaskRunner
    ltx_api_client: LTXAPIClient
    zit_api_client: ZitAPIClient
    fast_video_pipeline_class: type[FastVideoPipeline]
    hq_video_pipeline_class: type[HQVideoPipeline]
    pro_video_pipeline_class: type[ProVideoPipeline]
    image_generation_pipeline_class: type[ImageGenerationPipeline]
    ic_lora_pipeline_class: type[IcLoraPipeline]
    depth_processor_pipeline_class: type[DepthProcessorPipeline]
    pose_processor_pipeline_class: type[PoseProcessorPipeline]
    a2v_pipeline_class: type[A2VPipeline]
    retake_pipeline_class: type[RetakePipeline]


def build_default_service_bundle() -> ServiceBundle:
    """ساخت بسته سرویس‌های پیش‌فرض با تمام pipelineهای مورد نیاز."""
    from services.a2v_pipeline.ltx_a2v_pipeline import LTXA2VPipeline
    from services.depth_processor_pipeline.midas_dpt_pipeline import MidasDPTPipeline
    from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline
    from services.gpu_cleaner.torch_cleaner import TorchCleaner
    from services.gpu_info.gpu_info_impl import GPUInfoImpl
    from services.hq_video_pipeline.ltx_hq_video_pipeline import LTXHQVideoPipeline
    from services.http_client.http_client_impl import HTTPClientImpl
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    from services.image_generation_pipeline.zit_image_generation_pipeline import (
        ZitImageGenerationPipeline,
    )
    from services.ltx_api_client.ltx_api_client_impl import LTXAPIClientImpl
    from services.model_downloader.hugging_face_downloader import (
        HuggingFaceDownloader,
    )
    from services.pose_processor_pipeline.dw_pose_pipeline import DWPosePipeline
    from services.pro_video_pipeline.ltx_pro_video_pipeline import LTXProVideoPipeline
    from services.retake_pipeline.ltx_retake_pipeline import LTXRetakePipeline
    from services.task_runner.threading_runner import ThreadingRunner
    from services.text_encoder.ltx_text_encoder import LTXTextEncoder
    from services.video_processor.video_processor_impl import VideoProcessorImpl
    from services.zit_api_client.zit_api_client_impl import ZitAPIClientImpl

    return ServiceBundle(
        http=HTTPClientImpl,
        gpu_cleaner=TorchCleaner,
        model_downloader=HuggingFaceDownloader,
        gpu_info=GPUInfoImpl,
        video_processor=VideoProcessorImpl,
        text_encoder=LTXTextEncoder,
        task_runner=ThreadingRunner,
        ltx_api_client=LTXAPIClientImpl,
        zit_api_client=ZitAPIClientImpl,
        fast_video_pipeline_class=LTXFastVideoPipeline,
        hq_video_pipeline_class=LTXHQVideoPipeline,
        pro_video_pipeline_class=LTXProVideoPipeline,
        image_generation_pipeline_class=ZitImageGenerationPipeline,
        ic_lora_pipeline_class=LTXIcLoraPipeline,
        depth_processor_pipeline_class=MidasDPTPipeline,
        pose_processor_pipeline_class=DWPosePipeline,
        a2v_pipeline_class=LTXA2VPipeline,
        retake_pipeline_class=LTXRetakePipeline,
    )


def build_initial_state(
    config: RuntimeConfig,
    default_settings: AppSettings,
    services: ServiceBundle | None = None,
) -> AppHandler:
    """
    ساخت state اولیه برنامه با سرویس‌های داده شده.
    
    Args:
        config: تنظیمات زمان اجرا
        default_settings: تنظیمات پیش‌فرض برنامه
        services: بسته سرویس‌ها (در صورت نبود، از پیش‌فرض استفاده می‌شود)
    
    Returns:
        AppHandler: نمونه پیکربندی شده AppHandler
    """
    service_bundle = services or build_default_service_bundle()

    return AppHandler(
        config=config,
        default_settings=default_settings,
        http=service_bundle.http,
        gpu_cleaner=service_bundle.gpu_cleaner,
        model_downloader=service_bundle.model_downloader,
        gpu_info=service_bundle.gpu_info,
        video_processor=service_bundle.video_processor,
        text_encoder=service_bundle.text_encoder,
        task_runner=service_bundle.task_runner,
        ltx_api_client=service_bundle.ltx_api_client,
        zit_api_client=service_bundle.zit_api_client,
        fast_video_pipeline_class=service_bundle.fast_video_pipeline_class,
        hq_video_pipeline_class=service_bundle.hq_video_pipeline_class,
        pro_video_pipeline_class=service_bundle.pro_video_pipeline_class,
        image_generation_pipeline_class=service_bundle.image_generation_pipeline_class,
        ic_lora_pipeline_class=service_bundle.ic_lora_pipeline_class,
        depth_processor_pipeline_class=service_bundle.depth_processor_pipeline_class,
        pose_processor_pipeline_class=service_bundle.pose_processor_pipeline_class,
        a2v_pipeline_class=service_bundle.a2v_pipeline_class,
        retake_pipeline_class=service_bundle.retake_pipeline_class,
    )
