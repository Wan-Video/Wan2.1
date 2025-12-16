# Deploying to Google Cloud

This guide provides instructions on how to build and deploy this application to Google Cloud using Docker, Cloud Build, and Cloud Run or Vertex AI Endpoints.

## Prerequisites

1.  **Google Cloud Project:** You have a Google Cloud Project with billing enabled.
2.  **Enable APIs:** Ensure the following APIs are enabled in your project:
    *   Cloud Build API
    *   Artifact Registry API
    *   Cloud Run API
    *   Vertex AI API (if deploying to Vertex AI Endpoints)
    *   Identity and Access Management (IAM) API
3.  **`gcloud` CLI:** You have the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed and configured.
4.  **Permissions:** You have sufficient permissions to create Artifact Registry repositories, trigger Cloud Builds, and deploy to Cloud Run/Vertex AI. Roles like `Artifact Registry Administrator`, `Cloud Build Editor`, `Cloud Run Admin`, and `Vertex AI Administrator` are typically needed.
5.  **Docker configured for gcloud:** Configure Docker to authenticate with Artifact Registry:
    ```bash
    gcloud auth configure-docker ${_REGION}-docker.pkg.dev
    ```
    Replace `${_REGION}` with the region of your Artifact Registry (e.g., `us-central1`). You'll define this region in the Cloud Build substitutions.

## Step 1: Configure Cloud Build Substitutions

The `cloudbuild.yaml` file uses substitutions for your project ID, Artifact Registry region, repository name, and image name. You need to set these. The easiest way for a manual build is via the `gcloud` command.

**Important:** Before running the build, decide on:
*   `YOUR_GCP_PROJECT_ID`: Your actual Google Cloud Project ID.
*   `YOUR_ARTIFACT_REGISTRY_REGION`: The Google Cloud region for your Artifact Registry (e.g., `us-central1`).
*   `YOUR_ARTIFACT_REGISTRY_REPO`: The name for your Artifact Registry repository (e.g., `wan-video-repo`).
*   `YOUR_IMAGE_NAME`: The name for your Docker image (e.g., `wan-video-generator`).

The `cloudbuild.yaml` has default values for `_REGION`, `_REPOSITORY`, and `_IMAGE_NAME`. You can either:
*   Modify these defaults directly in `cloudbuild.yaml`.
*   Override them when submitting the build using the `--substitutions` flag (recommended for flexibility).

Example of overriding:
`--substitutions=PROJECT_ID="your-gcp-project-id",_REGION="us-central1",_REPOSITORY="wan-video-repo",_IMAGE_NAME="wan-video-generator"`

## Step 2: Create an Artifact Registry Repository

If you haven't already, create a Docker repository in Artifact Registry:

```bash
gcloud artifacts repositories create YOUR_ARTIFACT_REGISTRY_REPO   --repository-format=docker   --location=YOUR_ARTIFACT_REGISTRY_REGION   --description="Docker repository for Wan video generator"
```
Ensure `YOUR_ARTIFACT_REGISTRY_REPO` and `YOUR_ARTIFACT_REGISTRY_REGION` match the values you'll use for the `_REPOSITORY` and `_REGION` substitutions in Cloud Build.

## Step 3: Build and Push the Docker Image

Submit the build to Google Cloud Build:

```bash
# Ensure you are in the root directory of the project where cloudbuild.yaml is located.
# Replace placeholders with your actual values.
gcloud builds submit --config cloudbuild.yaml   --substitutions=PROJECT_ID="YOUR_GCP_PROJECT_ID",_REGION="YOUR_ARTIFACT_REGISTRY_REGION",_REPOSITORY="YOUR_ARTIFACT_REGISTRY_REPO",_IMAGE_NAME="YOUR_IMAGE_NAME"   .
```
This command uses `cloudbuild.yaml` to build your Docker image and push it to your Artifact Registry.

## Step 4: Deploy the Container

You can deploy the container to Cloud Run or Vertex AI Endpoints.

### Option A: Deploy to Cloud Run

Cloud Run is suitable for web applications like Gradio interfaces.

1.  **Basic Deployment (CPU only):**

    ```bash
    # Replace placeholders with your values.
    # YOUR_CLOUD_RUN_REGION should be a region where Cloud Run is available (e.g., us-central1).
    gcloud run deploy YOUR_IMAGE_NAME       --image YOUR_ARTIFACT_REGISTRY_REGION-docker.pkg.dev/YOUR_GCP_PROJECT_ID/YOUR_ARTIFACT_REGISTRY_REPO/YOUR_IMAGE_NAME:latest       --platform managed       --region YOUR_CLOUD_RUN_REGION       --allow-unauthenticated       --port 7860       --set-env-vars GRADIO_APP_SCRIPT="gradio/t2v_14B_singleGPU.py"       --memory=4Gi       --cpu=2
      # Adjust memory, CPU, and other flags as needed.
      # For larger models, you will need significantly more memory and potentially more CPU.
    ```
    *   The `--port 7860` matches the `EXPOSE 7860` in the Dockerfile.
    *   Use `--set-env-vars` to specify which Gradio app to run via the `GRADIO_APP_SCRIPT` environment variable.
    *   The image path format is `REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/IMAGE_NAME:latest`.

2.  **Deployment with GPU:**

    Cloud Run supports GPUs (check availability and machine types in your region).
    ```bash
    # Replace placeholders. YOUR_CLOUD_RUN_REGION must support GPUs.
    gcloud beta run deploy YOUR_IMAGE_NAME-gpu       --image YOUR_ARTIFACT_REGISTRY_REGION-docker.pkg.dev/YOUR_GCP_PROJECT_ID/YOUR_ARTIFACT_REGISTRY_REPO/YOUR_IMAGE_NAME:latest       --platform managed       --region YOUR_CLOUD_RUN_REGION       --allow-unauthenticated       --port 7860       --set-env-vars GRADIO_APP_SCRIPT="gradio/t2v_14B_singleGPU.py"       --memory=16Gi       --cpu=4       --execution-environment gen2       --args=--machine-type=a2-highgpu-1g # Or other suitable GPU machine types.
      # Consult Cloud Run documentation for current GPU options.
    ```

3.  **Considerations for Cloud Run:**
    *   **Timeout:** Default request timeout is 5 minutes. Increase if needed (max 60 minutes for Gen2).
    *   **Concurrency:** Adjust based on instance capacity.
    *   **Model Loading & Storage:**
        *   Models are currently packaged in the Docker image. This increases image size.
        *   For very large models, consider downloading them at startup from Google Cloud Storage (GCS) into the container. This would require modifying the Docker `CMD` or `ENTRYPOINT`.

### Option B: Deploy to Vertex AI Endpoints

Vertex AI Endpoints are better for dedicated ML model serving and offer more powerful hardware options.

1.  **Create an Endpoint:**
    ```bash
    # Replace placeholders. YOUR_VERTEX_AI_REGION is e.g., us-central1.
    gcloud ai endpoints create       --project=YOUR_GCP_PROJECT_ID       --region=YOUR_VERTEX_AI_REGION       --display-name="wan-video-endpoint"
    ```
    Note the `ENDPOINT_ID` from the output.

2.  **Deploy the model (container) to the Endpoint:**
    ```bash
    # Replace placeholders.
    # ENDPOINT_ID is from the previous command.
    # MACHINE_TYPE can be n1-standard-4, or a GPU type like a2-highgpu-1g.
    gcloud ai models deploy wan-video-model       --project=YOUR_GCP_PROJECT_ID       --region=YOUR_VERTEX_AI_REGION       --endpoint=ENDPOINT_ID       --display-name="v1"       --container-image-uri="YOUR_ARTIFACT_REGISTRY_REGION-docker.pkg.dev/YOUR_GCP_PROJECT_ID/YOUR_ARTIFACT_REGISTRY_REPO/YOUR_IMAGE_NAME:latest"       --machine-type=MACHINE_TYPE       # --container-env-vars="GRADIO_APP_SCRIPT=gradio/t2v_14B_singleGPU.py" # If serving Gradio UI
      # --container-ports=7860 # If serving Gradio UI
      # For dedicated prediction (non-Gradio), you'd typically implement /predict and /health routes.
      # --container-predict-route="/predict"
      # --container-health-route="/health"
    ```
    *   Vertex AI is more commonly used for direct prediction endpoints. If serving a Gradio UI, ensure networking and port configurations are appropriate. Accessing a UI might require additional setup (e.g., IAP).

## Step 5: Accessing your Deployed Application

*   **Cloud Run:** The `gcloud run deploy` command will output a service URL.
*   **Vertex AI Endpoint:** Typically accessed programmatically via SDK or REST API.

## Step 6: Checking Logs for Troubleshooting

*   **Cloud Build Logs:** Google Cloud Console > Cloud Build > History.
*   **Cloud Run Logs:** Google Cloud Console > Cloud Run > Select Service > Logs tab.
*   **Vertex AI Endpoint Logs:** Google Cloud Console > Vertex AI > Endpoints > Select Endpoint > View Logs. Also available in Cloud Logging.

Look for errors related to dependency installation, model loading, resource limits (memory/CPU), or port configurations.

## Choosing Machine Types and Resources

*   **CPU/Memory:** The 14B models are very demanding. Start with high CPU/memory (e.g., 4+ vCPUs, 16GB+ RAM) and scale up.
*   **GPU:** Essential for 14B models and highly recommended for 1.3B models.
    *   **Cloud Run:** Gen2 execution environment with GPU machine types.
    *   **Vertex AI:** Offers a wide variety of GPUs (T4, V100, A100).
*   **Model Sizes & Compatibility:**
    *   The `nvcr.io/nvidia/pytorch:24.03-py3` base image uses CUDA 12.1. Ensure chosen GPUs are compatible (e.g., Ampere, Hopper).
    *   For 14B models, you'll likely need GPUs with large VRAM (e.g., A100 40GB or 80GB). Check the model's specific requirements.

This guide provides a starting point. You may need to adjust configurations based on the specific model you are deploying and your performance requirements.
