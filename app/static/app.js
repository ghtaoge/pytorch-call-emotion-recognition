const $ = (id) => document.getElementById(id);
const labels = { neutral: "中性", happy: "开心", anger: "愤怒", sad: "悲伤" };
// 页面只保留当前音频和录音流的内存引用，清除或离开页面时统一释放。
const state = { file: null, url: null, recorder: null, chunks: [], timer: null, seconds: 0, result: null };

function setStatus(message, busy = false) {
  $("globalStatus").querySelector("span").textContent = message;
  $("globalStatus").classList.toggle("busy", busy);
}
function showError(message = "") { $("errorMessage").textContent = message; $("errorMessage").hidden = !message; }
function formatTime(seconds) { const value = Math.max(0, Math.floor(seconds)); return `${String(Math.floor(value / 60)).padStart(2,"0")}:${String(value % 60).padStart(2,"0")}`; }
function formatSize(bytes) { return bytes < 1048576 ? `${(bytes/1024).toFixed(1)} KB` : `${(bytes/1048576).toFixed(1)} MB`; }

function switchMode(mode) {
  if (state.recorder?.state === "recording" || state.recorder?.state === "paused") return;
  const upload = mode === "upload";
  $("modeUpload").setAttribute("aria-selected", String(upload)); $("modeRecord").setAttribute("aria-selected", String(!upload));
  $("uploadPanel").hidden = !upload; $("recordPanel").hidden = upload; showError();
}
function releaseAudio() {
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = null; state.file = null; $("audioPlayer").removeAttribute("src"); $("audioPreview").hidden = true;
  $("analyzeButton").disabled = true; $("resultContent").hidden = true; $("resultEmpty").hidden = false; setStatus("等待音频");
}
function clearAll() { stopRecorder(true); releaseAudio(); showError(); $("fileInput").value = ""; state.result = null; }

async function useFile(file, source = "上传") {
  showError();
  if (!file || file.size > 50 * 1024 * 1024) return showError("音频文件不能超过 50 MB");
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file; state.url = URL.createObjectURL(file); $("audioPlayer").src = state.url;
  $("fileName").textContent = file.name; $("fileName").title = file.name; $("fileMeta").textContent = formatSize(file.size); $("sourceBadge").textContent = source;
  $("audioPreview").hidden = false; $("analyzeButton").disabled = false; setStatus("音频已就绪");
  drawWaveform(file);
}
async function drawWaveform(file) {
  const canvas = $("waveform"), ctx = canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.strokeStyle="#c9ccc8"; ctx.beginPath(); ctx.moveTo(0,canvas.height/2); ctx.lineTo(canvas.width,canvas.height/2); ctx.stroke();
  try {
    const audioContext = new AudioContext(); const buffer = await audioContext.decodeAudioData(await file.arrayBuffer()); const samples = buffer.getChannelData(0); const step = Math.max(1,Math.floor(samples.length/canvas.width));
    ctx.strokeStyle="#30363d"; ctx.beginPath();
    for (let x=0;x<canvas.width;x++) { let peak=0; for(let j=0;j<step;j++) peak=Math.max(peak,Math.abs(samples[x*step+j]||0)); const h=peak*canvas.height*.44; ctx.moveTo(x,canvas.height/2-h); ctx.lineTo(x,canvas.height/2+h); } ctx.stroke(); await audioContext.close();
  } catch { /* 后端会给出最终格式校验，浏览器无法解码时保留播放器状态。 */ }
}

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
function pauseRecording() { if(!state.recorder) return; if(state.recorder.state==="recording") { state.recorder.pause(); $("recordState").textContent="录音已暂停"; $("pauseButton").querySelector("img").src="/static/icons/play.svg"; } else if(state.recorder.state==="paused") startRecording(); }
function stopRecorder(discard=false) { if(state.timer) clearInterval(state.timer); state.timer=null; if(state.recorder && state.recorder.state!=="inactive") { if(discard) state.recorder.onstop=()=>state.recorder.stream.getTracks().forEach(t=>t.stop()); state.recorder.stop(); } state.recorder=null; $("recordButton").classList.remove("recording"); $("recordState").textContent="点击开始录音"; $("recordTimer").textContent="00:00"; $("pauseButton").disabled=true; $("stopButton").disabled=true; $("pauseButton").querySelector("img").src="/static/icons/pause.svg"; }

function renderProbabilities(probabilities) { document.querySelectorAll(".probability-row").forEach((row)=>{ const value=probabilities[row.dataset.emotion]||0; row.querySelector("i").style.width=`${(value*100).toFixed(1)}%`; row.querySelector("b").textContent=`${(value*100).toFixed(1)}%`; }); }
function renderResult(result) {
  state.result=result; $("resultEmpty").hidden=true; $("resultContent").hidden=false; $("dominantEmotion").textContent=(result.reliability.level==="low"?"更倾向于":"")+labels[result.dominant_emotion];
  const badge=$("reliabilityBadge"); badge.textContent=result.reliability.level==="high"?"可靠性较高":"谨慎参考"; badge.className=`reliability ${result.reliability.level}`; $("elapsedTime").textContent=`${result.elapsed_ms} ms · ${result.device}`;
  $("voicedRatio").textContent=`有效语音 ${(result.voiced_ratio*100).toFixed(0)}%`; const notice=$("excludedNotice"); notice.hidden=result.reliability.level==="high"; notice.textContent=`四分类外情绪占比 ${(result.excluded_probability*100).toFixed(1)}%，建议结合原音判断。`;
  renderProbabilities(result.probabilities); const timeline=$("timeline"); timeline.replaceChildren(); result.segments.forEach((segment)=>{ const button=document.createElement("button"); button.type="button"; button.className=segment.is_silent?"silent":`${segment.dominant_emotion} ${segment.reliability.level}`; button.textContent=segment.is_silent?"静音":labels[segment.dominant_emotion]; button.title=`${segment.start_seconds.toFixed(1)}s - ${segment.end_seconds.toFixed(1)}s`; button.addEventListener("click",()=>selectSegment(segment,button)); timeline.append(button); }); setStatus("分析完成");
}
function selectSegment(segment,button) { document.querySelectorAll("#timeline button").forEach((item)=>item.classList.remove("selected")); button.classList.add("selected"); $("audioPlayer").currentTime=segment.start_seconds; if(segment.is_silent) $("segmentDetail").textContent=`${segment.start_seconds.toFixed(1)}–${segment.end_seconds.toFixed(1)} 秒 · 静音`; else { const top=(segment.probabilities[segment.dominant_emotion]*100).toFixed(1); $("segmentDetail").textContent=`${segment.start_seconds.toFixed(1)}–${segment.end_seconds.toFixed(1)} 秒 · ${labels[segment.dominant_emotion]} ${top}% · ${segment.reliability.level==="high"?"可靠性较高":"谨慎参考"}`; } }

async function analyze() {
  if(!state.file) return; showError(); $("analyzeButton").disabled=true; setStatus("正在连接本地模型",true); const body=new FormData(); body.append("audio",state.file,state.file.name);
  try { const response=await fetch("/api/analyze",{method:"POST",body}); if(!response.ok) { const payload=await response.json(); throw new Error(payload.error?.message||"分析失败"); } const reader=response.body.getReader(),decoder=new TextDecoder(); let buffer="";
    // 网络分块不保证与 JSON 行对齐，因此保留最后一个不完整行到下一次读取。
    while(true) { const {value,done}=await reader.read(); buffer+=decoder.decode(value||new Uint8Array(),{stream:!done}); const lines=buffer.split("\n"); buffer=lines.pop()||""; for(const line of lines) if(line.trim()) handleEvent(JSON.parse(line)); if(done) break; }
    if(buffer.trim()) handleEvent(JSON.parse(buffer));
  } catch(error) { showError(error.message||"分析失败，请重试"); setStatus("分析未完成"); } finally { $("analyzeButton").disabled=!state.file; }
}
function handleEvent(event) { if(event.type==="status"||event.type==="progress") setStatus(event.message,true); else if(event.type==="result") renderResult(event.result); else if(event.type==="error") { showError(event.error.message); setStatus("分析未完成"); } }
async function checkHealth() { try { const response=await fetch("/api/health"); const health=await response.json(); $("serviceDot").classList.add("online"); $("serviceStatus").textContent="服务正常"; $("modelStatus").textContent={not_loaded:"未加载",loading:"加载中",loaded:"已加载",error:"异常"}[health.model_status]; $("deviceStatus").textContent=health.device.toUpperCase(); } catch { $("serviceStatus").textContent="服务离线"; } }

$("modeUpload").addEventListener("click",()=>switchMode("upload")); $("modeRecord").addEventListener("click",()=>switchMode("record")); $("fileInput").addEventListener("change",(event)=>useFile(event.target.files[0])); $("recordButton").addEventListener("click",startRecording); $("pauseButton").addEventListener("click",pauseRecording); $("stopButton").addEventListener("click",()=>stopRecorder()); $("clearButton").addEventListener("click",clearAll); $("analyzeButton").addEventListener("click",analyze);
const dropzone=$("dropzone"); ["dragenter","dragover"].forEach((name)=>dropzone.addEventListener(name,(event)=>{event.preventDefault();dropzone.classList.add("dragover");})); ["dragleave","drop"].forEach((name)=>dropzone.addEventListener(name,(event)=>{event.preventDefault();dropzone.classList.remove("dragover");})); dropzone.addEventListener("drop",(event)=>useFile(event.dataTransfer.files[0])); window.addEventListener("beforeunload",()=>{stopRecorder(true);if(state.url)URL.revokeObjectURL(state.url);}); checkHealth();
