import os
import torch
import uuid
import time
import asyncio
import numpy as np
from threading import Lock
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, ValidationError
from diffusers.utils import export_to_video, load_image
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel
from PIL import Image
import requests
from io import BytesIO
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from requests.exceptions import RequestException

# 创建存储目录
os.makedirs("generated_videos", exist_ok=True)
os.makedirs("temp_images", exist_ok=True)

# ======================
# 生命周期管理
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """资源管理器"""
    try:
        # 初始化认证系统
        app.state.valid_api_keys = {
            "密钥"
        }

        # 初始化模型
        model_id = "./Wan2.1-I2V-14B-480P-Diffusers"
      
        # 加载图像编码器
        image_encoder = CLIPVisionModel.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float32
        )
      
        # 加载VAE
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32
        )
      
        # 配置调度器
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=3.0
        )
      
        # 创建管道
        app.state.pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        app.state.pipe.scheduler = scheduler

        # 初始化任务系统
        app.state.tasks: Dict[str, dict] = {}
        app.state.pending_queue: List[str] = []
        app.state.model_lock = Lock()
        app.state.task_lock = Lock()
        app.state.base_url = "ip地址+端口"
        app.state.semaphore = asyncio.Semaphore(2)  # 并发限制

        # 启动后台处理器
        asyncio.create_task(task_processor())

        print("✅ 系统初始化完成")
        yield

    finally:
        # 资源清理
        if hasattr(app.state, 'pipe'):
            del app.state.pipe
            torch.cuda.empty_cache()
            print("♻️ 资源已释放")

# ======================
# FastAPI应用
# ======================
app = FastAPI(lifespan=lifespan)
app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")
# 认证模块
security = HTTPBearer(auto_error=False)

# ======================
# 数据模型--查询参数模型
# ======================
class VideoSubmitRequest(BaseModel):
    model: str = Field(
        default="Wan2.1-I2V-14B-480P",
        description="模型版本"
    )
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="视频描述提示词，10-500个字符"
    )
    image_url: str = Field(
        ...,
        description="输入图像URL，需支持HTTP/HTTPS协议"
    )
    image_size: str = Field(
        default="auto",
        description="输出分辨率，格式：宽x高 或 auto（自动计算）"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="排除不需要的内容"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="随机数种子，范围0-2147483647"
    )
    num_frames: int = Field(
        default=81,
        ge=24,
        le=120,
        description="视频帧数，24-89帧"
    )
    guidance_scale: float = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="引导系数，1.0-20.0"
    )
    infer_steps: int = Field(
        default=30,
        ge=20,
        le=100,
        description="推理步数，20-100步"
    )

    @field_validator('image_size')
    def validate_image_size(cls, v):
        allowed_sizes = {"480x832", "832x480", "auto"}
        if v not in allowed_sizes:
            raise ValueError(f"支持的分辨率: {', '.join(allowed_sizes)}")
        return v

class VideoStatusRequest(BaseModel):
    requestId: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="32位任务ID"
    )

class VideoStatusResponse(BaseModel):
    status: str = Field(..., description="任务状态: Succeed, InQueue, InProgress, Failed,Cancelled")
    reason: Optional[str] = Field(None, description="失败原因")
    results: Optional[dict] = Field(None, description="生成结果")
    queue_position: Optional[int] = Field(None, description="队列位置")

class VideoCancelRequest(BaseModel):
    requestId: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="32位任务ID"
    )

# ======================
# 核心逻辑
# ======================
async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """统一认证验证"""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={"status": "Failed", "reason": "缺少认证头"},
            headers={"WWW-Authenticate": "Bearer"}
        )
    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=401,
            detail={"status": "Failed", "reason": "无效的认证方案"},
            headers={"WWW-Authenticate": "Bearer"}
        )
    if credentials.credentials not in app.state.valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail={"status": "Failed", "reason": "无效的API密钥"},
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True

async def task_processor():
    """任务处理器"""
    while True:
        async with app.state.semaphore:
            task_id = await get_next_task()
            if task_id:
                await process_task(task_id)
            else:
                await asyncio.sleep(0.5)

async def get_next_task():
    """获取下一个任务"""
    with app.state.task_lock:
        return app.state.pending_queue.pop(0) if app.state.pending_queue else None

async def process_task(task_id: str):
    """处理单个任务"""
    task = app.state.tasks.get(task_id)
    if not task:
        return

    try:
        # 更新任务状态
        task['status'] = 'InProgress'
        task['started_at'] = int(time.time())
        print(task['request'].image_url)
        # 下载输入图像
        image = await download_image(task['request'].image_url)
        image_path = f"temp_images/{task_id}.jpg"
        image.save(image_path)

        # 生成视频
        video_path = await generate_video(task['request'], task_id, image)
      
        # 生成下载链接
        download_url = f"{app.state.base_url}/videos/{os.path.basename(video_path)}"
      
        # 更新任务状态
        task.update({
            'status': 'Succeed',
            'download_url': download_url,
            'completed_at': int(time.time())
        })
      
        # 安排清理
        asyncio.create_task(cleanup_files([image_path, video_path]))
    except Exception as e:
        handle_task_error(task, e)

def handle_task_error(task: dict, error: Exception):
    """错误处理（包含详细错误信息）"""
    error_msg = str(error)
  
    # 1. 显存不足错误
    if isinstance(error, torch.cuda.OutOfMemoryError):
        error_msg = "显存不足，请降低分辨率"
      
    # 2. 网络请求相关错误
    elif isinstance(error, (RequestException, HTTPException)):
        # 从异常中提取具体信息
        if isinstance(error, HTTPException):
            # 如果是 HTTPException，获取其 detail 字段
            error_detail = getattr(error, "detail", "")
            error_msg = f"图像下载失败: {error_detail}"
          
        elif isinstance(error, Timeout):
            error_msg = "图像下载超时，请检查网络"
          
        elif isinstance(error, ConnectionError):
            error_msg = "无法连接到服务器，请检查 URL"
          
        elif isinstance(error, HTTPError):
            # requests 的 HTTPError（例如 4xx/5xx 状态码）
            status_code = error.response.status_code
            error_msg = f"服务器返回错误状态码: {status_code}"
          
        else:
            # 其他 RequestException 错误
            error_msg = f"图像下载失败: {str(error)}"
  
    # 3. 其他未知错误
    else:
        error_msg = f"未知错误: {str(error)}"
  
    # 更新任务状态
    task.update({
        'status': 'Failed',
        'reason': error_msg,
        'completed_at': int(time.time())
    })
# ======================
# 视频生成逻辑
# ======================
async def download_image(url: str) -> Image.Image:
    """异步下载图像（包含详细错误信息）"""
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None, 
            lambda: requests.get(url)  # 将 timeout 传递给 requests.get
        )
      
        # 如果状态码非 200，主动抛出 HTTPException
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"服务器返回状态码 {response.status_code}"
            )
          
        return Image.open(BytesIO(response.content)).convert("RGB")
      
    except RequestException as e:
        # 将原始 requests 错误信息抛出
        raise HTTPException(
            status_code=500,
            detail=f"请求失败: {str(e)}"
        )
async def generate_video(request: VideoSubmitRequest, task_id: str, image: Image.Image):
    """异步生成入口"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        sync_generate_video,
        request,
        task_id,
        image
    )

def sync_generate_video(request: VideoSubmitRequest, task_id: str, image: Image.Image):
    """同步生成核心"""
    with app.state.model_lock:
        try:
            # 解析分辨率
            mod_value = 16  # 模型要求的模数
            print(request.image_size)
            print('--------------------------------')
            if request.image_size == "auto":
                # 原版自动计算逻辑
                aspect_ratio = image.height / image.width
                print(image.height,image.width)
                max_area = 399360  # 模型基础分辨率
              
                # 计算理想尺寸
                height = round(np.sqrt(max_area * aspect_ratio)) 
                width = round(np.sqrt(max_area / aspect_ratio))
              
                # 应用模数调整
                height = height // mod_value * mod_value
                width = width // mod_value * mod_value
                resized_image = image.resize((width, height))
            else:
                width_str, height_str = request.image_size.split('x')
                width = int(width_str)
                height = int(height_str)
                mod_value = 16          
                # 调整图像尺寸
                resized_image = image.resize((width, height))
            
              
            # 设置随机种子
            generator = None
            # 修改点1: 使用属性访问seed
            if request.seed is not None:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(request.seed)  # 修改点2
                print(f"🔮 使用随机种子: {request.seed}")
            print(resized_image)
            print(height,width)

            # 执行推理
            output = app.state.pipe(
                image=resized_image,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=height,
                width=width,
                num_frames=request.num_frames,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.infer_steps,
                generator=generator
            ).frames[0]

            # 导出视频
            video_id = uuid.uuid4().hex
            output_path = f"generated_videos/{video_id}.mp4"
            export_to_video(output, output_path, fps=16)
            return output_path
        except Exception as e:
            raise RuntimeError(f"视频生成失败: {str(e)}") from e

# ======================
# API端点
# ======================
@app.post("/video/submit",
          response_model=dict,
          status_code=status.HTTP_202_ACCEPTED,
          tags=["视频生成"])
async def submit_task(
    request: VideoSubmitRequest,
    auth: bool = Depends(verify_auth)
):
    """提交生成任务"""
    # 参数验证
    if request.image_url is None:
        raise HTTPException(
            status_code=422,
            detail={"status": "Failed", "reason": "需要图像URL参数"}
        )

    # 创建任务记录
    task_id = uuid.uuid4().hex
    with app.state.task_lock:
        app.state.tasks[task_id] = {
            "request": request,
            "status": "InQueue",
            "created_at": int(time.time())
        }
        app.state.pending_queue.append(task_id)
  
    return {"requestId": task_id}

@app.post("/video/status",
          response_model=VideoStatusResponse,
          tags=["视频生成"])
async def get_status(
    request: VideoStatusRequest,
    auth: bool = Depends(verify_auth)
):
    """查询任务状态"""
    task = app.state.tasks.get(request.requestId)
    if not task:
        raise HTTPException(
            status_code=404,
            detail={"status": "Failed", "reason": "无效的任务ID"}
        )

    # 计算队列位置（仅当在队列中时）
    queue_pos = 0
    if task['status'] == "InQueue" and request.requestId in app.state.pending_queue:
        queue_pos = app.state.pending_queue.index(request.requestId) + 1

    response = {
        "status": task['status'],
        "reason": task.get('reason'),
        "queue_position": queue_pos if task['status'] == "InQueue" else None  # 非排队状态返回null
    }

    # 成功状态的特殊处理
    if task['status'] == "Succeed":
        response["results"] = {
            "videos": [{"url": task['download_url']}],
            "timings": {
                "inference": task['completed_at'] - task['started_at']
            },
            "seed": task['request'].seed
        }
    # 取消状态的补充信息
    elif task['status'] == "Cancelled":
        response["reason"] = task.get('reason', "用户主动取消")  # 确保原因字段存在

    return response

@app.post("/video/cancel",
         response_model=dict,
         tags=["视频生成"])
async def cancel_task(
    request: VideoCancelRequest,
    auth: bool = Depends(verify_auth)
):
    """取消排队中的生成任务"""
    task_id = request.requestId
  
    with app.state.task_lock:
        task = app.state.tasks.get(task_id)
      
        # 检查任务是否存在
        if not task:
            raise HTTPException(
                status_code=404,
                detail={"status": "Failed", "reason": "无效的任务ID"}
            )
          
        current_status = task['status']
      
        # 仅允许取消排队中的任务
        if current_status != "InQueue":
            raise HTTPException(
                status_code=400,
                detail={"status": "Failed", "reason": f"仅允许取消排队任务，当前状态: {current_status}"}
            )
      
        # 从队列移除
        try:
            app.state.pending_queue.remove(task_id)
        except ValueError:
            pass  # 可能已被处理
      
        # 更新任务状态
        task.update({
            "status": "Cancelled",
            "reason": "用户主动取消",
            "completed_at": int(time.time())
        })
      
    return {"status": "Succeed"}

async def cleanup_files(paths: List[str], delay: int = 3600):
    """定时清理文件"""
    await asyncio.sleep(delay)
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"清理失败 {path}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)