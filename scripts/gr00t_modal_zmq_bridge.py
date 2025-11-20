#!/usr/bin/env python
"""
ZMQ-to-HTTP Bridge for Modal Inference

This script creates a local ZMQ server that forwards inference requests
to a Modal HTTP endpoint. This allows existing robot evaluation scripts
that use ZMQ (ExternalRobotInferenceClient) to work with Modal deployments.

Usage:
    python modal_zmq_bridge.py --modal-url https://your-modal-url --port 5555

Example:
    python modal_zmq_bridge.py --modal-url https://lbxa--groot-inference-serve-dev.modal.run
"""

from dataclasses import dataclass
import base64
import logging
import time

import cv2
import numpy as np
import requests
import tyro

from gr00t.eval.service import BaseInferenceServer

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for Modal-ZMQ bridge."""

    modal_url: str = "https://lbxa--groot-inference-serve-dev.modal.run"
    """The Modal HTTP endpoint URL (root path, no /act needed)."""

    port: int = 5555
    """Local ZMQ port to listen on."""

    timeout: int = 120
    """HTTP request timeout in seconds."""

    compress_images: bool = True
    """Whether to compress images with JPEG before sending (reduces payload ~50x)."""

    jpeg_quality: int = 85
    """JPEG compression quality (0-100, higher = better quality but larger size)."""

    api_token: str = None
    """Optional API token for ZMQ authentication."""


def compress_image_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """
    Compress image to JPEG and encode as base64 string.

    Args:
        image: Image array (H, W, C) in RGB format
        quality: JPEG quality (0-100)

    Returns:
        Base64-encoded JPEG string
    """
    # OpenCV uses BGR, so convert if needed
    if image.shape[-1] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    # Encode as JPEG
    success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Failed to encode image as JPEG")

    # Convert to base64
    jpeg_base64 = base64.b64encode(buffer).decode("utf-8")
    return jpeg_base64


def decompress_base64_to_image(jpeg_base64: str) -> np.ndarray:
    """
    Decode base64 JPEG string back to image array.

    Args:
        jpeg_base64: Base64-encoded JPEG string

    Returns:
        Image array (H, W, C) in RGB format
    """
    # Decode base64
    jpeg_bytes = base64.b64decode(jpeg_base64)

    # Decode JPEG
    image_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


class ModalZMQBridge(BaseInferenceServer):
    """
    Bridge server that translates ZMQ requests to Modal HTTP calls.

    This allows robot eval scripts using ExternalRobotInferenceClient (ZMQ)
    to connect to Modal inference servers (HTTP).
    """

    def __init__(
        self,
        modal_url: str,
        port: int = 5555,
        timeout: int = 30,
        compress_images: bool = True,
        jpeg_quality: int = 85,
        api_token: str = None,
    ):
        super().__init__(host="*", port=port, api_token=api_token)
        self.modal_url = modal_url
        self.timeout = timeout
        self.compress_images = compress_images
        self.jpeg_quality = jpeg_quality
        self.request_count = 0

        # Register the get_action endpoint to forward to Modal
        self.register_endpoint("get_action", self._forward_to_modal)

        print("=" * 80)
        print("Modal-ZMQ Bridge Server")
        print("=" * 80)
        print(f"Local ZMQ endpoint:  tcp://localhost:{port}")
        print(f"Forwarding to Modal: {modal_url}")
        print(f"Timeout:             {timeout} seconds")
        print(f"Image compression:   {'ENABLED' if compress_images else 'DISABLED'}")
        if compress_images:
            print(f"JPEG quality:        {jpeg_quality}")
        print("Debug logging:       ENABLED")
        print("=" * 80)
        logger.info("Bridge server initialized and ready")

    def _forward_to_modal(self, observation: dict) -> dict:
        """
        Forward observation to Modal HTTP endpoint and return action.

        Args:
            observation: Dictionary containing robot observations

        Returns:
            Dictionary containing predicted actions

        Raises:
            RuntimeError: If Modal request fails
        """
        self.request_count += 1
        request_id = self.request_count

        logger.info("=" * 60)
        logger.info(f"REQUEST #{request_id} - Received from ZMQ client")
        logger.info("=" * 60)

        # Log observation details
        logger.info(f"Observation keys: {list(observation.keys())}")
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: type={type(value).__name__}")

        try:
            # Convert numpy arrays to lists for JSON serialization
            # Compress video frames to reduce payload size
            obs_serialized = {}
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    # Compress video frames (identified by 'video.' prefix)
                    if self.compress_images and key.startswith("video."):
                        # Handle batched images (N, H, W, C)
                        if value.ndim == 4:
                            compressed_frames = []
                            for frame in value:
                                compressed = compress_image_to_base64(frame, self.jpeg_quality)
                                compressed_frames.append(compressed)
                            obs_serialized[key] = {"type": "jpeg_base64_batch", "data": compressed_frames}
                            logger.info(f"  {key}: compressed {len(compressed_frames)} frames")
                        # Handle single image (H, W, C)
                        elif value.ndim == 3:
                            compressed = compress_image_to_base64(value, self.jpeg_quality)
                            obs_serialized[key] = {"type": "jpeg_base64", "data": compressed}
                            logger.info(f"  {key}: compressed single frame")
                        else:
                            obs_serialized[key] = value.tolist()
                    else:
                        obs_serialized[key] = value.tolist()
                elif isinstance(value, list):
                    obs_serialized[key] = value
                else:
                    obs_serialized[key] = value

            # Send request to Modal with observation data
            logger.info(f"Sending POST request to Modal: {self.modal_url}")
            start_time = time.time()

            response = requests.post(self.modal_url, json={"observation": obs_serialized}, timeout=self.timeout)

            elapsed_time = time.time() - start_time
            logger.info(f"Modal response received in {elapsed_time:.2f} seconds")
            logger.info(f"Response status code: {response.status_code}")

            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"Modal returned status {response.status_code}: {response.text[:200]}"
                logger.error(f"ERROR: {error_msg}")
                print(f"\n❌ REQUEST #{request_id} FAILED: {error_msg}\n")
                raise RuntimeError(error_msg)

            # Parse and return action
            action_response = response.json()
            logger.info(f"Action keys received: {list(action_response.keys())}")

            # Convert lists back to numpy arrays (Modal returns lists for JSON serialization)
            action = {}
            for key, value in action_response.items():
                if isinstance(value, list):
                    arr = np.array(value)
                    action[key] = arr
                    logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    action[key] = value
                    logger.info(f"  {key}: type={type(value).__name__}")

            print(f"✅ REQUEST #{request_id} SUCCESS - {elapsed_time:.2f}s - {len(action)} action keys")
            logger.info("=" * 60)
            return action

        except requests.exceptions.Timeout:
            error_msg = f"Modal request timed out after {self.timeout} seconds"
            logger.error(f"ERROR: {error_msg}")
            print(f"\n❌ REQUEST #{request_id} TIMEOUT: {error_msg}\n")
            raise RuntimeError(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Modal request failed: {str(e)}"
            logger.error(f"ERROR: {error_msg}")
            print(f"\n❌ REQUEST #{request_id} FAILED: {error_msg}\n")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"ERROR: {error_msg}", exc_info=True)
            print(f"\n❌ REQUEST #{request_id} ERROR: {error_msg}\n")
            raise RuntimeError(error_msg)


def main(config: BridgeConfig):
    """Start the Modal-ZMQ bridge server."""
    # Create and run bridge server
    server = ModalZMQBridge(
        modal_url=config.modal_url,
        port=config.port,
        timeout=config.timeout,
        compress_images=config.compress_images,
        jpeg_quality=config.jpeg_quality,
        api_token=config.api_token,
    )

    print("\n" + "=" * 80)
    print("🚀 Bridge server is ready! Waiting for ZMQ requests...")
    print("=" * 80)
    print("\nYou can now run your robot evaluation script.")
    print("Press Ctrl+C to stop the server.\n")
    print("Listening for requests on tcp://localhost:{}...".format(config.port))
    print("(If you don't see any REQUEST messages below, no packets are coming through)")
    print()

    try:
        server.run()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print(f"Shutting down bridge server... (Processed {server.request_count} requests)")
        print("=" * 80)


if __name__ == "__main__":
    config = tyro.cli(BridgeConfig)
    main(config)
