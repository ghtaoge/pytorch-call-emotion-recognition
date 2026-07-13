# 架构说明

`app/main.py` 是 HTTP 边界，只处理上传限制、错误转换、并发锁和流式响应。`app/audio.py` 负责 FFmpeg 解码、16 kHz 单声道归一化、静音检测和窗口切分。`app/model.py` 延迟加载 HuBERT，并使用 `torch.inference_mode()` 推理。`app/analyzer.py` 将模型输出投影为四类、计算可靠性并按有效时长与能量聚合。

```text
浏览器上传/录音
      |
      v
FastAPI -> 大小校验 -> 临时解码 -> 16 kHz 波形
                                      |
                                      v
                              6 秒窗口 / 静音过滤
                                      |
                                      v
                           HuBERT 六分类 PyTorch 推理
                                      |
                                      v
                         四分类投影 + 可靠性 + 聚合
                                      |
                                      v
                           NDJSON 进度与结果 -> 页面
```

服务没有数据库。运行时状态仅包含模型实例和单任务并发锁，刷新页面不会保留结果。
