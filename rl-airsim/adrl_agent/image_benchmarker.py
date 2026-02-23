from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

try:  # optional dependency
    import cv2
except Exception:  # pragma: no cover - runtime optional dependency
    cv2 = None


@dataclass
class ImageBenchmarkerConfig:
    benchmark_type: str = "simGetImages"
    camera_name: str = "fpv_cam"
    vehicle_name: str = ""
    period_sec: float = 0.05
    decode_image: bool = True


class AirSimImageBenchmarker:
    """Background image API benchmark utility inspired by ADRL baseline benchmarker."""

    def __init__(self, image_client: Any, config: Optional[Mapping[str, Any]] = None):
        cfg = dict(config or {})
        benchmark_type = str(cfg.get("benchmark_type", ImageBenchmarkerConfig.benchmark_type))
        if benchmark_type not in {"simGetImage", "simGetImages"}:
            raise ValueError(f"Unknown benchmark_type: {benchmark_type}")

        decode_image = bool(cfg.get("decode_image", ImageBenchmarkerConfig.decode_image))
        if decode_image and cv2 is None:
            decode_image = False

        self.config = ImageBenchmarkerConfig(
            benchmark_type=benchmark_type,
            camera_name=str(cfg.get("camera_name", ImageBenchmarkerConfig.camera_name)),
            vehicle_name=str(cfg.get("vehicle_name", ImageBenchmarkerConfig.vehicle_name)),
            period_sec=float(cfg.get("period_sec", ImageBenchmarkerConfig.period_sec)),
            decode_image=decode_image,
        )
        self.image_client = image_client

        self._num_images = 0
        self._total_time_sec = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="airsim-image-benchmarker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def reset_stats(self) -> None:
        with self._lock:
            self._num_images = 0
            self._total_time_sec = 0.0

    def get_metrics(self) -> dict[str, float]:
        with self._lock:
            num_images = self._num_images
            total_time_sec = self._total_time_sec
        avg_fps = (num_images / total_time_sec) if num_images > 0 and total_time_sec > 0.0 else 0.0
        return {
            "num_images": float(num_images),
            "total_time_sec": float(total_time_sec),
            "avg_fps": float(avg_fps),
        }

    def _run_loop(self) -> None:
        while self._running:
            iter_start = time.time()
            try:
                if self.config.benchmark_type == "simGetImage":
                    self._benchmark_sim_get_image()
                else:
                    self._benchmark_sim_get_images()
                elapsed = time.time() - iter_start
                with self._lock:
                    self._num_images += 1
                    self._total_time_sec += elapsed
            except Exception:
                # Keep benchmark thread alive through transient RPC hiccups.
                pass

            sleep_sec = self.config.period_sec - (time.time() - iter_start)
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)

    def _benchmark_sim_get_image(self) -> None:
        import airsimdroneracinglab as airsim

        response = self.image_client.simGetImage(
            self.config.camera_name,
            airsim.ImageType.Scene,
            vehicle_name=self.config.vehicle_name,
        )
        if self.config.decode_image and response:
            image_bytes = airsim.string_to_uint8_array(response)
            if cv2 is not None:
                cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)

    def _benchmark_sim_get_images(self) -> None:
        import airsimdroneracinglab as airsim

        request = [
            airsim.ImageRequest(
                self.config.camera_name,
                airsim.ImageType.Scene,
                False,
                False,
            )
        ]
        response = self.image_client.simGetImages(
            request,
            vehicle_name=self.config.vehicle_name,
        )
        if not response:
            return
        if self.config.decode_image:
            img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
            expected = int(response[0].height) * int(response[0].width) * 3
            if expected > 0 and img_rgb_1d.size == expected:
                img_rgb_1d.reshape(response[0].height, response[0].width, 3)


def build_image_benchmarker(image_client: Any, config: Optional[Mapping[str, Any]] = None):
    return AirSimImageBenchmarker(image_client=image_client, config=config or {})
