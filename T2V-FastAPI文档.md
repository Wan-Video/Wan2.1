
# 视频生成服务API文档

## 一、功能概述
本服务基于Wan2.1-T2V-1.3B模型实现文本到视频生成，包含以下核心功能：
1. **异步任务队列**：支持多任务排队和并发控制（最大2个并行任务）
2. **资源管理**：
   - 显存优化（使用bfloat16精度）
   - 生成视频自动清理（默认1小时后删除）
3. **安全认证**：基于API Key的Bearer Token验证
4. **任务控制**：支持任务提交/状态查询/取消操作

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
  "model": "Wan2.1-T2V-1.3B",
  "prompt": "A beautiful sunset over the mountains",
  "image_size": "480x832",
  "num_frames": 81,
  "guidance_scale": 5.0,
  "infer_steps": 50
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
    "timings": {"inference": 120}
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
1. 选择POST方法，输入URL：`/video/submit`
2. Headers添加：
   ```text
   Authorization: Bearer YOUR_API_KEY
   Content-Type: application/json
   ```
3. Body选择raw/JSON格式，输入请求参数

### 3. 查询状态
1. 新建请求，URL填写`/video/status`
2. 使用相同认证头
3. Body中携带requestId

### 4. 取消任务
1. 新建DELETE请求，URL填写`/video/cancel`
2. Body携带需要取消的requestId

### 注意事项
1. 所有接口必须携带有效API Key
2. 视频生成耗时约2-5分钟（根据参数配置）
3. 生成视频默认保留1小时

---

## 四、参数规范
| 参数名           | 允许值范围                     | 必填 | 说明                     |
|------------------|-------------------------------|------|--------------------------|
| prompt           | 10-500字符                    | 是   | 视频内容描述             |
| image_size       | "480x832" 或 "832x480"        | 是   | 分辨率                   |
| num_frames       | 24-120                        | 是   | 视频总帧数               |
| guidance_scale   | 1.0-20.0                      | 是   | 文本引导强度             |
| infer_steps      | 20-100                        | 是   | 推理步数                 |
| seed             | 0-2147483647                  | 否   | 随机种子                 |

---

## 五、状态码说明
| 状态码 | 含义                     |
|--------|--------------------------|
| 202    | 任务已接受               |
| 401    | 认证失败                 |
| 404    | 任务不存在               |
| 422    | 参数校验失败             |
| 500    | 服务端错误（显存不足等） |


**提示**：建议使用Swagger文档进行接口测试，访问`http://服务器地址:8088/docs`可查看自动生成的API文档界面