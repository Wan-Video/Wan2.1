import replicate
from core.config import settings
from typing import Dict, Any, Optional

# Initialize Replicate client
replicate_client = replicate.Client(api_token=settings.REPLICATE_API_TOKEN)


class ReplicateService:
    """Service for interacting with Replicate API for video generation"""

    # Model versions - these would be updated with actual Wan2.1 model versions
    WAN_T2V_14B = "wan-ai/wan2.1-t2v-14b:latest"
    WAN_T2V_1_3B = "wan-ai/wan2.1-t2v-1.3b:latest"
    WAN_I2V_14B = "wan-ai/wan2.1-i2v-14b:latest"

    @staticmethod
    async def generate_text_to_video(
        prompt: str,
        negative_prompt: Optional[str] = None,
        model: str = "t2v-14B",
        resolution: str = "720p",
        duration: int = 5,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate video from text using Wan2.1 models via Replicate

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt to avoid certain features
            model: Model version to use (t2v-14B or t2v-1.3B)
            resolution: Video resolution (480p or 720p)
            duration: Video duration in seconds
            seed: Random seed for reproducibility

        Returns:
            Prediction ID from Replicate
        """
        model_version = (
            ReplicateService.WAN_T2V_14B if model == "t2v-14B" else ReplicateService.WAN_T2V_1_3B
        )

        # Map resolution to size
        size_map = {"480p": "832*480", "720p": "1280*720"}

        input_params = {
            "prompt": prompt,
            "size": size_map.get(resolution, "1280*720"),
            "sample_steps": 50 if model == "t2v-14B" else 40,
        }

        if negative_prompt:
            input_params["negative_prompt"] = negative_prompt
        if seed is not None:
            input_params["seed"] = seed

        # Create prediction
        prediction = replicate_client.predictions.create(
            version=model_version, input=input_params
        )

        return prediction.id

    @staticmethod
    async def generate_image_to_video(
        prompt: str,
        image_url: str,
        negative_prompt: Optional[str] = None,
        resolution: str = "720p",
        duration: int = 5,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate video from image using Wan2.1 I2V model via Replicate

        Args:
            prompt: Text prompt for video generation
            image_url: URL of the input image
            negative_prompt: Negative prompt
            resolution: Video resolution
            duration: Video duration in seconds
            seed: Random seed

        Returns:
            Prediction ID from Replicate
        """
        size_map = {"480p": "832*480", "720p": "1280*720"}

        input_params = {
            "prompt": prompt,
            "image": image_url,
            "size": size_map.get(resolution, "1280*720"),
            "sample_steps": 40,
        }

        if negative_prompt:
            input_params["negative_prompt"] = negative_prompt
        if seed is not None:
            input_params["seed"] = seed

        prediction = replicate_client.predictions.create(
            version=ReplicateService.WAN_I2V_14B, input=input_params
        )

        return prediction.id

    @staticmethod
    async def get_prediction_status(prediction_id: str) -> Dict[str, Any]:
        """
        Get the status of a Replicate prediction

        Args:
            prediction_id: The prediction ID

        Returns:
            Dictionary containing status, output, and error information
        """
        prediction = replicate_client.predictions.get(prediction_id)

        return {
            "id": prediction.id,
            "status": prediction.status,
            "output": prediction.output,
            "error": prediction.error,
            "logs": prediction.logs,
        }

    @staticmethod
    async def cancel_prediction(prediction_id: str) -> bool:
        """
        Cancel a running prediction

        Args:
            prediction_id: The prediction ID

        Returns:
            True if cancelled successfully
        """
        try:
            prediction = replicate_client.predictions.cancel(prediction_id)
            return prediction.status == "canceled"
        except Exception as e:
            print(f"Error cancelling prediction: {e}")
            return False
