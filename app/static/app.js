/* ================================================================
   声析 · 通话情绪识别 — 前端应用逻辑 (app.js)

   模块职责：
   1. 状态管理：跟踪当前音频文件、录音状态、分析结果等
   2. 音频获取：支持文件上传和浏览器录音两种方式
   3. 波形绘制：通过 AudioContext 解码音频并绘制采样峰值波形
   4. 情绪分析：向后端发送 NDJSON 流式请求，实时接收进度和结果
   5. 结果渲染：展示主导情绪、概率条、分段时间轴等可视化组件
   6. 事件绑定：拖拽上传、录音控制、分析触发、分段选择等交互

   数据流：
   用户上传/录音 → 生成File对象 → 绘制波形 → 点击分析 →
   POST /api/analyze (multipart) → NDJSON流式响应 →
   status/progress事件 → result事件 → 渲染结果面板
   ================================================================ */

// ====== 工具函数 ======
// $: 快捷 DOM 查询，通过 id 获取元素，减少重复的 document.getElementById 调用
const $ = (id) => document.getElementById(id);

// labels: 四分类情绪的中英文映射字典，用于渲染时将后端英文标签转换为中文显示
const labels = { neutral: "中性", happy: "开心", anger: "愤怒", sad: "悲伤" };

// ====== 全局状态对象 ======
// 页面只保留当前音频和录音流的内存引用，清除或离开页面时统一释放。
// state 是整个应用的核心数据模型，所有函数都依赖它来追踪当前状态：
//   file      - 当前选中的音频 File 对象（上传文件或录音生成的 Blob）
//   url       - 通过 URL.createObjectURL 创建的临时音频 URL，用于 <audio> 播放
//   recorder  - MediaRecorder 实例，录音过程中持有对媒体流的引用
//   chunks    - 录音过程中收集的数据块数组，录音结束后合并为 Blob
//   timer     - 录音计时器的 setInterval 引用，用于更新计时器显示
//   seconds   - 当前录音时长（秒），到达300秒（5分钟）时自动停止录音
//   result    - 最终分析结果对象，包含主导情绪、概率、分段信息等
const state = { file: null, url: null, recorder: null, chunks: [], timer: null, seconds: 0, result: null };

// ====== 状态指示函数 ======

/**
 * setStatus - 更新全局状态指示器的文字和样式
 * @param {string} message - 状态文字内容（如"等待音频"、"正在连接本地模型"、"分析完成"）
 * @param {boolean} busy - 是否处于忙碌状态；为 true 时图标显示脉冲动画
 * 使用场景：分析流程的各个阶段（连接、加载、推理）都会调用此函数更新进度提示
 */
function setStatus(message, busy = false) {
  $("globalStatus").querySelector("span").textContent = message;
  $("globalStatus").classList.toggle("busy", busy);
}

/**
 * showError - 显示或隐藏错误提示文字
 * @param {string} message - 错误信息内容；为空字符串时隐藏提示
 * 使用场景：文件过大、格式不支持、麦克风权限缺失、网络异常、后端返回错误等
 */
function showError(message = "") { $("errorMessage").textContent = message; $("errorMessage").hidden = !message; }

// ====== 格式化工具函数 ======

/**
 * formatTime - 将秒数格式化为 MM:SS 时间字符串
 * @param {number} seconds - 秒数（可为负数或浮点数，函数内部取整并保证非负）
 * @returns {string} - 如 "02:35" 格式的时间字符串
 * 使用场景：录音计时器显示、分段时间轴的时间标签
 */
function formatTime(seconds) { const value = Math.max(0, Math.floor(seconds)); return `${String(Math.floor(value / 60)).padStart(2,"0")}:${String(value % 60).padStart(2,"0")}`; }

/**
 * formatSize - 将文件字节大小格式化为易读的 KB 或 MB 字符串
 * @param {number} bytes - 字节数
 * @returns {string} - 小于1MB时显示KB（如 "3.2 KB"），否则显示MB（如 "12.5 MB"）
 * 使用场景：音频文件信息行的文件大小显示
 */
function formatSize(bytes) { return bytes < 1048576 ? `${(bytes/1024).toFixed(1)} KB` : `${(bytes/1048576).toFixed(1)} MB`; }

// ====== 模式切换函数 ======

/**
 * switchMode - 在"上传音频"和"现场录音"之间切换面板显示
 * @param {string} mode - "upload" 或 "record"
 * 切换逻辑：
 *   1. 录音进行中（recording/paused状态）时禁止切换，防止录音数据丢失
 *   2. 更新两个 tab 按钮的 aria-selected 属性（无障碍语义）
 *   3. 显示对应面板，隐藏另一面板
 *   4. 清除已有错误提示
 */
function switchMode(mode) {
  if (state.recorder?.state === "recording" || state.recorder?.state === "paused") return;
  const upload = mode === "upload";
  $("modeUpload").setAttribute("aria-selected", String(upload)); $("modeRecord").setAttribute("aria-selected", String(!upload));
  $("uploadPanel").hidden = !upload; $("recordPanel").hidden = upload; showError();
}

// ====== 音频资源管理函数 ======

/**
 * releaseAudio - 释放当前音频资源，恢复初始状态
 * 执行步骤：
 *   1. 通过 URL.revokeObjectURL 释放临时 URL，避免内存泄漏
 *   2. 清空 state 中的 file 和 url 引用
 *   3. 移除音频播放器的 src 属性
 *   4. 隐藏音频预览区
 *   5. 禁用分析按钮
 *   6. 隐藏结果内容区，显示空状态占位
 *   7. 重置全局状态为"等待音频"
 * 使用场景：清除按钮点击时、切换模式时可能调用
 */
function releaseAudio() {
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = null; state.file = null; $("audioPlayer").removeAttribute("src"); $("audioPreview").hidden = true;
  $("analyzeButton").disabled = true; $("resultContent").hidden = true; $("resultEmpty").hidden = false; setStatus("等待音频");
}

/**
 * clearAll - 完全重置所有状态，恢复页面到初始空白状态
 * 执行步骤：
 *   1. 停止录音器（discard=true 模式，丢弃录音数据）
 *   2. 释放音频资源
 *   3. 清除错误提示
 *   4. 清空文件输入框的 value（防止重复选择同一文件）
 *   5. 清空 state.result
 * 使用场景：清除按钮点击时
 */
function clearAll() { stopRecorder(true); releaseAudio(); showError(); $("fileInput").value = ""; state.result = null; }

// ====== 文件加载函数 ======

/**
 * useFile - 将音频文件加载到预览区，准备分析
 * @param {File} file - 音频文件对象（来自文件输入框或录音 Blob）
 * @param {string} source - 来源标签文字，默认"上传"，录音时为"录音"
 * 执行步骤：
 *   1. 清除已有错误提示
 *   2. 校验文件：空文件或超过50MB时显示错误并返回
 *   3. 释放之前的临时 URL（如有）
 *   4. 将文件存入 state.file，创建 ObjectURL 存入 state.url
 *   5. 设置音频播放器的 src 为 ObjectURL
 *   6. 更新文件名、大小、来源标签
 *   7. 显示音频预览区，启用分析按钮
 *   8. 更新全局状态为"音频已就绪"
 *   9. 调用 drawWaveform 绘制波形图
 * 注意：ObjectURL 是浏览器内存引用，页面关闭或刷新后自动释放，
 *       但为避免内存泄漏，切换文件时主动释放旧的 URL
 */
async function useFile(file, source = "上传") {
  showError();
  if (!file || file.size > 50 * 1024 * 1024) return showError("音频文件不能超过 50 MB");
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file; state.url = URL.createObjectURL(file); $("audioPlayer").src = state.url;
  $("fileName").textContent = file.name; $("fileName").title = file.name; $("fileMeta").textContent = formatSize(file.size); $("sourceBadge").textContent = source;
  $("audioPreview").hidden = false; $("analyzeButton").disabled = false; setStatus("音频已就绪");
  drawWaveform(file);
}

// ====== 波形绘制函数 ======

/**
 * drawWaveform - 在 Canvas 上绘制音频采样峰值波形
 * @param {File} file - 音频文件对象
 * 算法流程：
 *   1. 获取 Canvas 2D 上下文，清除画布
 *   2. 绘制中线（浅灰色水平线，作为振幅零点参考）
 *   3. 通过 AudioContext.decodeAudioData 解码音频文件为 PCM 数据
 *   4. 取第一个声道（左声道/单声道）的采样数据
 *   5. 计算降采样步长 step = 采样总数 / Canvas宽度，确保每个像素对应一段采样
 *   6. 对每个像素位置 x：
 *      - 遍历 step 个采样点，取绝对值最大的峰值 peak
 *      - 将峰值映射为画布高度的 44%（保留上下留白）
 *      - 绘制垂直线段：从 (x, center-h) 到 (x, center+h)
 *   7. 关闭 AudioContext 释放资源
 *   8. 解码失败时保留中线，不阻断后续流程（后端会给出最终格式校验）
 * 注意：此波形仅为视觉预览，不影响分析精度
 */
async function drawWaveform(file) {
  const canvas = $("waveform"), ctx = canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.strokeStyle="#c9ccc8"; ctx.beginPath(); ctx.moveTo(0,canvas.height/2); ctx.lineTo(canvas.width,canvas.height/2); ctx.stroke();
  try {
    const audioContext = new AudioContext(); const buffer = await audioContext.decodeAudioData(await file.arrayBuffer()); const samples = buffer.getChannelData(0); const step = Math.max(1,Math.floor(samples.length/canvas.width));
    ctx.strokeStyle="#30363d"; ctx.beginPath();
    for (let x=0;x<canvas.width;x++) { let peak=0; for(let j=0;j<step;j++) peak=Math.max(peak,Math.abs(samples[x*step+j]||0)); const h=peak*canvas.height*.44; ctx.moveTo(x,canvas.height/2-h); ctx.lineTo(x,canvas.height/2+h); } ctx.stroke(); await audioContext.close();
  } catch { /* 后端会给出最终格式校验，浏览器无法解码时保留播放器状态。 */ }
}

// ====== 录音控制函数 ======

/**
 * startRecording - 开始或恢复浏览器录音
 * 流程分两种情况：
 *   A. 暂停状态恢复录音：直接调用 recorder.resume()，更新界面状态
 *   B. 新开始录音：
 *     1. 通过 navigator.mediaDevices.getUserMedia 获取麦克风音频流
 *        参数：单声道、开启回声消除和噪声抑制（提升通话录音质量）
 *     2. 创建 MediaRecorder 实例，设置数据收集回调
 *        - ondataavailable: 每次收到数据块（每500ms触发一次）时存入 chunks 数组
 *     3. 设置录音结束回调 onstop：
 *        - 将 chunks 合并为 Blob，再转为 File 对象
 *        - 停止麦克风媒体流的所有轨道（释放硬件资源）
 *        - 调用 useFile 将录音文件加载到预览区
 *     4. 调用 recorder.start(500) 开始录音，每500ms触发一次数据收集
 *     5. 更新界面：录音按钮添加 recording 样式（脉冲环），状态文字更新
 *     6. 启动计时器：每秒更新计时器显示，到达300秒（5分钟）时自动停止
 *     7. 麦克风权限被拒绝或设备不可用时显示错误提示
 * 注意：MediaRecorder 不上传媒体流；录音块只在浏览器内合并成一个临时 File。
 */
async function startRecording() {
  if (state.recorder?.state === "paused") { state.recorder.resume(); $("recordState").textContent="正在录音"; $("pauseButton").querySelector("img").src="/static/icons/pause.svg"; return; }
  try {
    // MediaRecorder 不上传媒体流；录音块只在浏览器内合并成一个临时 File。
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount:1, echoCancellation:true, noiseSuppression:true } });
    const recorder = new MediaRecorder(stream); state.recorder=recorder; state.chunks=[]; state.seconds=0;
    recorder.ondataavailable = (event) => { if(event.data.size) state.chunks.push(event.data); };
    recorder.onstop = () => { const type=recorder.mimeType||"audio/webm"; const blob=new Blob(state.chunks,{type}); const file=new File([blob],`本地录音-${Date.now()}.webm`,{type}); stream.getTracks().forEach((track)=>track.stop()); useFile(file,"录音"); };
    recorder.start(500); $("recordButton").classList.add("recording"); $("recordState").textContent="正在录音"; $("pauseButton").disabled=false; $("stopButton").disabled=false;
    state.timer=setInterval(()=>{ state.seconds++; $("recordTimer").textContent=formatTime(state.seconds); if(state.seconds>=300) stopRecorder(); },1000);
  } catch { showError("无法使用麦克风，请在浏览器设置中允许录音权限"); }
}

/**
 * pauseRecording - 暂停或恢复录音（切换按钮行为）
 * 逻辑：
 *   - 录音中（state==="recording"）：调用 recorder.pause()，暂停按钮图标切换为播放
 *   - 已暂停（state==="paused"）：调用 startRecording() 恢复录音（内部走 resume 逻辑）
 * 注意：暂停期间不收集数据块，但已收集的 chunks 保留
 */
function pauseRecording() { if(!state.recorder) return; if(state.recorder.state==="recording") { state.recorder.pause(); $("recordState").textContent="录音已暂停"; $("pauseButton").querySelector("img").src="/static/icons/play.svg"; } else if(state.recorder.state==="paused") startRecording(); }

/**
 * stopRecorder - 停止录音器并清理资源
 * @param {boolean} discard - 是否丢弃录音数据（true时不调用useFile，直接释放麦克风）
 * 执行步骤：
 *   1. 清除计时器 clearInterval
 *   2. 若录音器非 inactive 状态：
 *      - discard 模式：覆盖 onstop 回调为仅释放麦克风轨道，不生成文件
 *      - 正常模式：保留默认 onstop 回调，录音结束后自动调用 useFile
 *      - 调用 recorder.stop() 触发 onstop 回调
 *   3. 重置 state.recorder 为 null
 *   4. 重置界面状态：移除 recording 样式、重置状态文字和计时器、禁用暂停/停止按钮
 */
function stopRecorder(discard=false) { if(state.timer) clearInterval(state.timer); state.timer=null; if(state.recorder && state.recorder.state!=="inactive") { if(discard) state.recorder.onstop=()=>state.recorder.stream.getTracks().forEach(t=>t.stop()); state.recorder.stop(); } state.recorder=null; $("recordButton").classList.remove("recording"); $("recordState").textContent="点击开始录音"; $("recordTimer").textContent="00:00"; $("pauseButton").disabled=true; $("stopButton").disabled=true; $("pauseButton").querySelector("img").src="/static/icons/pause.svg"; }

// ====== 结果渲染函数 ======

/**
 * renderProbabilities - 更新四分类概率条的可视化
 * @param {Object} probabilities - 包含四种情绪概率值的对象，如 { neutral: 0.6, happy: 0.15, anger: 0.1, sad: 0.15 }
 * 逻辑：遍历所有 .probability-row 元素，根据 data-emotion 属性查找对应概率值，
 *        将 <i> 填充条的 width 设置为概率百分比，将 <b> 数值文字同步更新
 * 进度条宽度变化带有 0.4s CSS 过渡动画，实现平滑展开效果
 */
function renderProbabilities(probabilities) { document.querySelectorAll(".probability-row").forEach((row)=>{ const value=probabilities[row.dataset.emotion]||0; row.querySelector("i").style.width=`${(value*100).toFixed(1)}%`; row.querySelector("b").textContent=`${(value*100).toFixed(1)}%`; }); }

/**
 * renderResult - 渲染完整的分析结果到结果面板
 * @param {Object} result - 后端返回的分析结果对象，包含以下字段：
 *   - dominant_emotion: 主导情绪标签（英文）
 *   - reliability: { level: "high"/"low" } 可靠性等级
 *   - elapsed_ms: 分析耗时毫秒数
 *   - device: 推理设备名称（如 CPU/CUDA）
 *   - voiced_ratio: 有效语音占比（0~1）
 *   - excluded_probability: 四分类外情绪占比（0~1）
 *   - probabilities: 四分类概率字典
 *   - segments: 分段检测结果数组
 * 渲染步骤：
 *   1. 保存结果到 state.result
 *   2. 隐藏空状态占位，显示结果内容区
 *   3. 设置主导情绪文字（低可靠性时前缀"更倾向于"）
 *   4. 设置可靠性徽章文字和样式类（high=绿色，low=橙色）
 *   5. 设置耗时和设备信息
 *   6. 设置有效语音占比文字
 *   7. 设置低可靠性时的排除情绪占比提示（高可靠性时隐藏此提示）
 *   8. 调用 renderProbabilities 更新概率条
 *   9. 构建时间轴按钮列表：
 *      - 清空现有按钮
 *      - 为每个分段创建按钮：
 *        - 静音分段：灰色底色 + "静音"文字
 *        - 有声分段：情绪对应颜色底色 + 中文情绪文字
 *        - 低可靠性分段：添加 "low" 类名使按钮半透明
 *        - 每个按钮的 title 显示时间范围（如 "0.0s - 3.0s"）
 *        - 添加 click 事件监听器，点击时调用 selectSegment 查看详情
 *   10. 更新全局状态为"分析完成"
 */
function renderResult(result) {
  state.result=result; $("resultEmpty").hidden=true; $("resultContent").hidden=false; $("dominantEmotion").textContent=(result.reliability.level==="low"?"更倾向于":"")+labels[result.dominant_emotion];
  const badge=$("reliabilityBadge"); badge.textContent=result.reliability.level==="high"?"可靠性较高":"谨慎参考"; badge.className=`reliability ${result.reliability.level}`; $("elapsedTime").textContent=`${result.elapsed_ms} ms · ${result.device}`;
  $("voicedRatio").textContent=`有效语音 ${(result.voiced_ratio*100).toFixed(0)}%`; const notice=$("excludedNotice"); notice.hidden=result.reliability.level==="high"; notice.textContent=`四分类外情绪占比 ${(result.excluded_probability*100).toFixed(1)}%，建议结合原音判断。`;
  renderProbabilities(result.probabilities); const timeline=$("timeline"); timeline.replaceChildren(); result.segments.forEach((segment)=>{ const button=document.createElement("button"); button.type="button"; button.className=segment.is_silent?"silent":`${segment.dominant_emotion} ${segment.reliability.level}`; button.textContent=segment.is_silent?"静音":labels[segment.dominant_emotion]; button.title=`${segment.start_seconds.toFixed(1)}s - ${segment.end_seconds.toFixed(1)}s`; button.addEventListener("click",()=>selectSegment(segment,button)); timeline.append(button); }); setStatus("分析完成");
}

/**
 * selectSegment - 选中时间轴分段，显示详细信息并跳转音频播放位置
 * @param {Object} segment - 分段数据对象，包含：
 *   - start_seconds / end_seconds: 时间范围
 *   - is_silent: 是否为静音段
 *   - dominant_emotion: 主导情绪标签
 *   - probabilities: 该段的概率字典
 *   - reliability.level: 可靠性等级
 * @param {HTMLElement} button - 被点击的时间轴按钮元素
 * 执行步骤：
 *   1. 移除所有时间轴按钮的 selected 类名
 *   2. 为当前按钮添加 selected 类名（2px墨色描边突出）
 *   3. 将音频播放器跳转到该段起始时间
 *   4. 更新分段详情文字：
 *      - 静音段：显示时间范围 + "静音"
 *      - 有声段：显示时间范围 + 情绪名称 + 主导概率百分比 + 可靠性标签
 */
function selectSegment(segment,button) { document.querySelectorAll("#timeline button").forEach((item)=>item.classList.remove("selected")); button.classList.add("selected"); $("audioPlayer").currentTime=segment.start_seconds; if(segment.is_silent) $("segmentDetail").textContent=`${segment.start_seconds.toFixed(1)}–${segment.end_seconds.toFixed(1)} 秒 · 静音`; else { const top=(segment.probabilities[segment.dominant_emotion]*100).toFixed(1); $("segmentDetail").textContent=`${segment.start_seconds.toFixed(1)}–${segment.end_seconds.toFixed(1)} 秒 · ${labels[segment.dominant_emotion]} ${top}% · ${segment.reliability.level==="high"?"可靠性较高":"谨慎参考"}`; } }

// ====== 分析请求函数 ======

/**
 * analyze - 向后端发起情绪分析请求，使用 NDJSON 流式响应实时接收结果
 * 流程：
 *   1. 校验 state.file 存在，清除错误提示，禁用分析按钮（防重复提交）
 *   2. 设置全局状态为"正在连接本地模型"（busy=true，图标脉冲动画）
 *   3. 构建 FormData，将音频文件作为 "audio" 字段附加
 *   4. 发起 POST /api/analyze 请求
 *   5. 检查响应状态：非 200 时解析错误 JSON 并抛出异常
 *   6. 进入流式读取循环：
 *      - 使用 ReadableStream reader 逐块读取响应体
 *      - 将二进制块通过 TextDecoder 解码为 UTF-8 文本
 *      - 按换行符 \n 分割文本为多行（NDJSON 格式：每行一个 JSON 对象）
 *      - 网络分块不保证与 JSON 行对齐，因此保留最后一个不完整行到下一次读取。
 *      - 对每行完整 JSON 调用 handleEvent 处理
 *      - 流结束时（done=true）处理 buffer 中可能残留的最后一行
 *   7. 错误时显示错误提示，重置全局状态为"分析未完成"
 *   8. finally 块中根据 state.file 是否存在重新设置按钮状态
 *
 * NDJSON 事件类型：
 *   - status/progress: 更新全局状态文字（如"正在加载模型"、"正在推理第N段"）
 *   - result: 完整分析结果，调用 renderResult 渲染
 *   - error: 服务端错误，显示错误信息
 */
async function analyze() {
  if(!state.file) return; showError(); $("analyzeButton").disabled=true; setStatus("正在连接本地模型",true); const body=new FormData(); body.append("audio",state.file,state.file.name);
  try { const response=await fetch("/api/analyze",{method:"POST",body}); if(!response.ok) { const payload=await response.json(); throw new Error(payload.error?.message||"分析失败"); } const reader=response.body.getReader(),decoder=new TextDecoder(); let buffer="";
    // 网络分块不保证与 JSON 行对齐，因此保留最后一个不完整行到下一次读取。
    while(true) { const {value,done}=await reader.read(); buffer+=decoder.decode(value||new Uint8Array(),{stream:!done}); const lines=buffer.split("\n"); buffer=lines.pop()||""; for(const line of lines) if(line.trim()) handleEvent(JSON.parse(line)); if(done) break; }
    if(buffer.trim()) handleEvent(JSON.parse(buffer));
  } catch(error) { showError(error.message||"分析失败，请重试"); setStatus("分析未完成"); } finally { $("analyzeButton").disabled=!state.file; }
}

/**
 * handleEvent - 处理 NDJSON 流式响应中的单个事件
 * @param {Object} event - 事件对象，包含 type 字段和对应的数据
 * 事件路由：
 *   - type === "status" 或 "progress": 更新全局状态文字，保持 busy 动画
 *   - type === "result": 调用 renderResult 渲染完整分析结果
 *   - type === "error": 显示错误信息，重置全局状态为"分析未完成"
 */
function handleEvent(event) { if(event.type==="status"||event.type==="progress") setStatus(event.message,true); else if(event.type==="result") renderResult(event.result); else if(event.type==="error") { showError(event.error.message); setStatus("分析未完成"); } }

// ====== 健康检查函数 ======

/**
 * checkHealth - 页面加载时检查后端服务健康状态
 * 通过 GET /api/health 获取服务信息：
 *   - model_status: 模型加载状态（not_loaded/loading loaded/error）
 *   - device: 推理设备名称（CPU/CUDA/MPS 等）
 * 正常时：状态圆点变绿，服务状态文字更新为"服务正常"
 *         模型状态和设备信息翻译为中文显示
 * 异常时（网络不通）：服务状态文字更新为"服务离线"
 * 此函数在页面脚本末尾自动调用一次
 */
async function checkHealth() { try { const response=await fetch("/api/health"); const health=await response.json(); $("serviceDot").classList.add("online"); $("serviceStatus").textContent="服务正常"; $("modelStatus").textContent={not_loaded:"未加载",loading:"加载中",loaded:"已加载",error:"异常"}[health.model_status]; $("deviceStatus").textContent=health.device.toUpperCase(); } catch { $("serviceStatus").textContent="服务离线"; } }

// ====== 事件监听器绑定 ======

// 模式切换：点击"上传音频"/"现场录音" tab 按钮切换面板显示
$("modeUpload").addEventListener("click",()=>switchMode("upload")); $("modeRecord").addEventListener("click",()=>switchMode("record"));

// 文件选择：通过文件输入框选择音频文件后加载到预览区
$("fileInput").addEventListener("change",(event)=>useFile(event.target.files[0]));

// 录音控制：开始/暂停/停止录音按钮
$("recordButton").addEventListener("click",startRecording); $("pauseButton").addEventListener("click",pauseRecording); $("stopButton").addEventListener("click",()=>stopRecorder());

// 清除按钮：重置所有状态恢复初始页面
$("clearButton").addEventListener("click",clearAll);

// 分析按钮：向后端发起情绪分析请求
$("analyzeButton").addEventListener("click",analyze);

// ====== 拖拽上传事件绑定 ======

// 获取拖拽区域元素
const dropzone=$("dropzone");

// 拖拽进入/悬停：阻止默认行为，添加 dragover 高亮样式
["dragenter","dragover"].forEach((name)=>dropzone.addEventListener(name,(event)=>{event.preventDefault();dropzone.classList.add("dragover");}));

// 拖拽离开/释放：阻止默认行为，移除 dragover 高亮样式
["dragleave","drop"].forEach((name)=>dropzone.addEventListener(name,(event)=>{event.preventDefault();dropzone.classList.remove("dragover");}));

// 拖拽释放时：获取拖入的第一个文件，调用 useFile 加载
dropzone.addEventListener("drop",(event)=>useFile(event.dataTransfer.files[0]));

// ====== 页面卸载清理 ======

// beforeunload 事件：页面关闭或刷新前释放所有资源
//   1. 停止录音器（discard 模式，丢弃数据）
//   2. 释放 ObjectURL（避免浏览器保留已关闭页面的内存引用）
window.addEventListener("beforeunload",()=>{stopRecorder(true);if(state.url)URL.revokeObjectURL(state.url);});

// ====== 初始化 ======

// 页面加载完成后自动检查后端服务健康状态
checkHealth();
