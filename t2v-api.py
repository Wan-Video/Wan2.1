import os
import torch
import uuid
import time
import asyncio
from enum import Enum
from threading import Lock
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, ValidationError
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

# 创建视频存储目录
os.makedirs("generated_videos", exist_ok=True)

# 生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期"""
    # 初始化模型和资源
    try:
        # 初始化认证密钥
        app.state.valid_api_keys = {
            "密钥"
        }

        # 初始化视频生成模型
        model_id = "./Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32
        )
      
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=3.0
        )
      
        app.state.pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        app.state.pipe.scheduler = scheduler

        # 初始化任务系统
        app.state.tasks: Dict[str, dict] = {}
        app.state.pending_queue: List[str] = []
        app.state.model_lock = Lock()
        app.state.task_lock = Lock()
        app.state.base_url = "ip地址+端口"
        app.state.max_concurrent = 2
        app.state.semaphore = asyncio.Semaphore(app.state.max_concurrent)

        # 启动后台任务处理器
        asyncio.create_task(task_processor())
      
        print("✅ 应用初始化完成")
        yield
      
    finally:
        # 清理资源
        if hasattr(app.state, 'pipe'):
            del app.state.pipe
            torch.cuda.empty_cache()
            print("♻️ 已释放模型资源")

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan)
app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

# 认证模块
security = HTTPBearer(auto_error=False)

# ======================
# 数据模型--查询参数模型
# ======================
class VideoSubmitRequest(BaseModel):
    model: str = Field(default="Wan2.1-T2V-1.3B",description="使用的模型版本")
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="视频描述提示词，10-500个字符"
    )
    image_size: str = Field(
        ...,
        description="视频分辨率，仅支持480x832或832x480"
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
        description="视频帧数，24-120帧"
    )
    guidance_scale: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="引导系数，1.0-20.0"
    )
    infer_steps: int = Field(
        default=50,
        ge=20,
        le=100,
        description="推理步数，20-100步"
    )

    @field_validator('image_size', mode='before')
    @classmethod
    def validate_image_size(cls, value):
        allowed_sizes = {"480x832", "832x480"}
        if value not in allowed_sizes:
            raise ValueError(f"仅支持以下分辨率: {', '.join(allowed_sizes)}")
        return value

class VideoStatusRequest(BaseModel):
    requestId: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="32位任务ID"
    )

class VideoSubmitResponse(BaseModel):
    requestId: str

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

# # 自定义HTTP异常处理器
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content=exc.detail,  # 直接返回detail内容（不再包装在detail字段）
#         headers=exc.headers
#     )

# ======================
# 后台任务处理
# ======================
async def task_processor():
    """处理任务队列"""
    while True:
        async with app.state.semaphore:
            task_id = await get_next_task()
            if task_id:
                await process_task(task_id)
            else:
                await asyncio.sleep(0.5)

async def get_next_task():
    """获取下一个待处理任务"""
    with app.state.task_lock:
        if app.state.pending_queue:
            return app.state.pending_queue.pop(0)
    return None

async def process_task(task_id: str):
    """处理单个任务"""
    task = app.state.tasks.get(task_id)
    if not task:
        return

    try:
        # 更新任务状态
        task['status'] = 'InProgress'
        task['started_at'] = int(time.time())

        # 执行视频生成
        video_path = await generate_video(task['request'], task_id)
      
        # 生成下载链接
        download_url = f"{app.state.base_url}/videos/{os.path.basename(video_path)}"
      
        # 更新任务状态
        task.update({
            'status': 'Succeed',
            'download_url': download_url,
            'completed_at': int(time.time())
        })
      
        # 安排自动清理
        asyncio.create_task(auto_cleanup(video_path))
    except Exception as e:
        handle_task_error(task, e)

def handle_task_error(task: dict, error: Exception):
    """统一处理任务错误"""
    error_msg = str(error)
    if isinstance(error, torch.cuda.OutOfMemoryError):
        error_msg = "显存不足，请降低分辨率或减少帧数"
    elif isinstance(error, ValidationError):
        error_msg = "参数校验失败: " + str(error)
  
    task.update({
        'status': 'Failed',
        'reason': error_msg,
        'completed_at': int(time.time())
    })

# ======================
# 视频生成核心逻辑
# ======================
async def generate_video(request: dict, task_id: str) -> str:
    """异步执行视频生成"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        sync_generate_video,
        request,
        task_id
    )

def sync_generate_video(request: dict, task_id: str) -> str:
    """同步生成视频"""
    with app.state.model_lock:
        try:
            generator = None
            if request.get('seed') is not None:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(request['seed'])
                print(f"🔮 使用随机种子: {request['seed']}")
            
            # 执行模型推理
            result = app.state.pipe(
                prompt=request['prompt'],
                negative_prompt=request['negative_prompt'],
                height=request['height'],
                width=request['width'],
                num_frames=request['num_frames'],
                guidance_scale=request['guidance_scale'],
                num_inference_steps=request['infer_steps'],
                generator=generator
            )
          
            # 导出视频文件
            video_id = uuid.uuid4().hex
            output_path = f"generated_videos/{video_id}.mp4"
            export_to_video(result.frames[0], output_path, fps=16)
          
            return output_path
        except Exception as e:
            raise RuntimeError(f"视频生成失败: {str(e)}") from e

# ======================
# API端点
# ======================
async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """认证验证"""
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

@app.post("/video/submit",
          response_model=VideoSubmitResponse,
          status_code=status.HTTP_202_ACCEPTED,
          tags=["视频生成"],
          summary="提交视频生成请求")
async def submit_video_task(
    request: VideoSubmitRequest,
    auth: bool = Depends(verify_auth)
):
    """提交新的视频生成任务"""
    try:
        # 解析分辨率参数
        width, height = map(int, request.image_size.split('x'))
      
        # 创建任务记录
        task_id = uuid.uuid4().hex
        task_data = {
            'request': {
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': width,
                'height': height,
                'num_frames': request.num_frames,
                'guidance_scale': request.guidance_scale,
                'infer_steps': request.infer_steps,
                'seed': request.seed
            },
            'status': 'InQueue',
            'created_at': int(time.time())
        }
      
        # 加入任务队列
        with app.state.task_lock:
            app.state.tasks[task_id] = task_data
            app.state.pending_queue.append(task_id)
      
        return {"requestId": task_id}
  
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"status": "Failed", "reason": str(e)}
        )

@app.post("/video/status",
          response_model=VideoStatusResponse,
          tags=["视频生成"],
          summary="查询任务状态")
async def get_video_status(
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
            "seed": task['request']['seed']
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


# ======================
# 工具函数
# ======================
async def auto_cleanup(file_path: str, delay: int = 3600):
    """自动清理生成的视频文件"""
    await asyncio.sleep(delay)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已清理文件: {file_path}")
    except Exception as e:
        print(f"文件清理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)