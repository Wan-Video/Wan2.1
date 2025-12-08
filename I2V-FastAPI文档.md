
# 图像到视频生成服务API文档

## 一、功能概述
基于Wan2.1-I2V-14B-480P模型实现图像到视频生成，核心功能包括：
1. **异步任务队列**：支持多任务排队和并发控制（最大2个并行任务）
2. **智能分辨率适配**：
   - 支持自动计算最佳分辨率（保持原图比例）
   - 支持手动指定分辨率（480x832/832x480）
3. **资源管理**：
   - 显存优化（bfloat16精度）
   - 生成文件自动清理（默认1小时）
4. **安全认证**：基于API Key的Bearer Token验证
5. **任务控制**：支持任务提交/状态查询/取消操作

技术栈：
- FastAPI框架
- CUDA加速
- 异步任务处理
- Diffusers推理库

---

## 二、接口说明

### 1. 提交生成任务
**POST /video/submit**
```json
{
  "model": "Wan2.1-I2V-14B-480P",
  "prompt": "A dancing cat in the style of Van Gogh",
  "image_url": "https://example.com/input.jpg",
  "image_size": "auto",
  "num_frames": 81,
  "guidance_scale": 3.0,
  "infer_steps": 30
}
```

**响应示例**：
```json
{
  "requestId": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

### 2. 查询任务状态
**POST /video/status**
```json
{
  "requestId": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

**响应示例**：
```json
{
  "status": "Succeed",
  "results": {
    "videos": [{"url": "http://localhost:8088/videos/abcd1234.mp4"}],
    "timings": {"inference": 90},
    "seed": 123456
  }
}
```

### 3. 取消任务
**POST /video/cancel**
```json
{
  "requestId": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

**响应示例**：
```json
{
  "status": "Succeed"
}
```

---

## 三、Postman使用指南

### 1. 基础配置
- 服务器地址：`http://ip地址:8088`
- 认证方式：Bearer Token
- Token值：需替换为有效API Key

### 2. 提交任务
1. 选择POST方法，URL填写`/video/submit`
2. Headers添加：
   ```text
   Authorization: Bearer YOUR_API_KEY
   Content-Type: application/json
   ```
3. Body示例（图像生成视频）：
   ```json
   {
     "prompt": "Sunset scene with mountains",
     "image_url": "https://example.com/mountain.jpg",
     "image_size": "auto",
     "num_frames": 50
   }
   ```

### 3. 特殊处理
- **图像下载失败**：返回400错误，包含具体原因（如URL无效/超时）
- **显存不足**：返回500错误并提示降低分辨率

---

## 四、参数规范
| 参数名           | 允许值范围                     | 必填 | 说明                                      |
|------------------|-------------------------------|------|------------------------------------------|
| image_url        | 有效HTTP/HTTPS URL            | 是   | 输入图像地址                              |
| prompt           | 10-500字符                    | 是   | 视频内容描述                              |
| image_size       | "480x832", "832x480", "auto"  | 是   | auto模式自动适配原图比例                  |
| num_frames       | 24-120                        | 是   | 视频总帧数                                |
| guidance_scale   | 1.0-20.0                      | 是   | 文本引导强度                              |
| infer_steps      | 20-100                        | 是   | 推理步数                                  |
| seed             | 0-2147483647                  | 否   | 随机种子                                  |

---

## 五、状态码说明
| 状态码 | 含义                               |
|--------|-----------------------------------|
| 202    | 任务已接受                         |
| 400    | 图像下载失败/参数错误              |
| 401    | 认证失败                           |
| 404    | 任务不存在                         |
| 422    | 参数校验失败                       |
| 500    | 服务端错误（显存不足/模型异常等）  |

---

## 六、特殊功能说明
1. **智能分辨率适配**：
   - 当`image_size="auto"`时，自动计算符合模型要求的最优分辨率
   - 保持原始图像宽高比，最大像素面积不超过399,360（约640x624）

2. **图像预处理**：
   - 自动转换为RGB模式
   - 根据目标分辨率进行等比缩放
   

**重要提示**：输入图像URL需保证公开可访问，私有资源需提供有效鉴权

**提示** ：访问`http://服务器地址:8088/docs`可查看交互式API文档，支持在线测试所有接口