"""
test_frontend —— 前端页面元素与样式的验证测试

本模块检查前端静态资源中是否包含必要的交互控件与样式定义：
- index.html：工作台页面必须包含所有交互元素（上传/录音/分析/清除等）
- styles.css：样式文件必须定义响应式布局与四类情感的颜色变量

这些测试确保前端重构后不会意外删除关键元素。
"""

from pathlib import Path


def test_workbench_contains_required_controls_and_local_assets() -> None:
    """验证前端工作台页面包含所有必要的交互控件与本地资源引用。

    逐一检查以下元素 ID 是否存在于 HTML 中：
    - modeUpload / modeRecord：上传与录音模式切换
    - fileInput：文件选择输入框
    - audioPlayer：音频播放器
    - waveform：波形可视化区域
    - recordButton / pauseButton / stopButton：录音控制按钮
    - analyzeButton / clearButton：分析与清除按钮
    - dominantEmotion：主导情感显示区域
    - probabilities：概率分布显示区域
    - timeline：时间线区域

    同时验证：
    - HTML 中不包含外部 URL（http/https），确保所有资源为本地引用
    - 页面引用了 /static/app.js 作为脚本资源
    """
    html = Path("app/static/index.html").read_text(encoding="utf-8")
    # 检查所有必需的交互元素 ID 是否存在于 HTML 中
    for element_id in (
        "modeUpload",  # 上传模式切换
        "modeRecord",  # 录音模式切换
        "fileInput",  # 文件选择输入框
        "audioPlayer",  # 音频播放器
        "waveform",  # 波形可视化区域
        "recordButton",  # 开始录音按钮
        "pauseButton",  # 暂停录音按钮
        "stopButton",  # 停止录音按钮
        "analyzeButton",  # 开始分析按钮
        "clearButton",  # 清除按钮
        "dominantEmotion",  # 主导情感显示
        "probabilities",  # 概率分布显示
        "timeline",  # 时间线区域
    ):
        assert f'id="{element_id}"' in html  # 每个 ID 都必须出现在 HTML 中
    # 确保不引用任何外部 URL，所有资源均为本地
    assert "http://" not in html
    assert "https://" not in html
    # 确保引用了本地 JavaScript 资源
    assert "/static/app.js" in html


def test_styles_define_responsive_layout_and_four_emotion_colors() -> None:
    """验证样式文件定义了响应式布局与四类情感颜色变量。

    检查 styles.css 中是否包含：
    - 响应式媒体查询 @media (max-width: 860px)：确保小屏幕适配
    - 四类情感 CSS 变量：--neutral、--happy、--anger、--sad
    - overflow-x: auto：确保内容溢出时可横向滚动
    """
    css = Path("app/static/styles.css").read_text(encoding="utf-8")
    assert "@media (max-width: 860px)" in css  # 应定义响应式断点
    assert "--neutral:" in css  # neutral 情感颜色变量
    assert "--happy:" in css  # happy 情感颜色变量
    assert "--anger:" in css  # anger 情感颜色变量
    assert "--sad:" in css  # sad 情感颜色变量
    assert "overflow-x: auto" in css  # 横向溢出滚动样式
