"use strict";
const $ = s => document.querySelector(s);
let controller, result, audioUrl, selectedFile;
const message = (text, error=false) => { $("#message").textContent=text; $("#message").classList.toggle("error",error); };
const headers = () => $("#api-key").value ? {"X-API-Key":$("#api-key").value} : {};
async function status() {
  $("#services").textContent="Checking local services…";
  try {
    const response=await fetch("/status",{headers:headers()}); const data=await response.json();
    if (!data.dependencies) throw Error(data.detail || "Cannot read service status.");
    $("#services").replaceChildren(...data.dependencies.map(d=>{
      const item=document.createElement("span"); item.className="service "+(d.status==="ready"?"ready":"down");
      item.textContent=d.label+": "+d.status; item.title=d.detail; return item;
    }));
  } catch(e) { $("#services").textContent=e.message; }
}
$("#refresh").addEventListener("click",status);
async function select(file) {
  selectedFile=undefined;
  if (!file || !file.name.toLowerCase().endsWith(".wav") || file.size>20*1024**2) { message("Choose a WAV recording smaller than 20 MiB.",true); return; }
  try {
    const bytes=await file.arrayBuffer();let buffer;
    try{const pcm=CardioWave.decode(bytes);buffer={...pcm,getChannelData:()=>pcm.signal};}
    catch(error){const context=new AudioContext();try{buffer=await context.decodeAudioData(bytes.slice(0));}finally{await context.close();}}
    if(buffer.duration>120) throw Error("Recording exceeds 120 seconds.");
    const signal=buffer.getChannelData(0), canvas=$("#waveform"), ctx=canvas.getContext("2d");
    const color=getComputedStyle(document.documentElement).getPropertyValue("--teal").trim();
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
    const step=Math.max(1,Math.floor(signal.length/canvas.width));
    for(let x=0;x<canvas.width;x++){let lo=0,hi=0;for(let j=x*step;j<Math.min((x+1)*step,signal.length);j++){lo=Math.min(lo,signal[j]);hi=Math.max(hi,signal[j]);}ctx.moveTo(x,90+lo*80);ctx.lineTo(x,90+hi*80);}ctx.stroke();
    if(audioUrl) URL.revokeObjectURL(audioUrl); audioUrl=URL.createObjectURL(CardioWave.playback(signal,buffer.sampleRate));
    $("#playback").src=audioUrl;$("#playback").hidden=false;
    $("#file-info").textContent=`${file.name} / ${buffer.duration.toFixed(1)} s / ${buffer.sampleRate.toLocaleString()} Hz`;
    selectedFile=file;message("Recording ready. Analysis uses a normalized five-second window.");
  }catch(e){message(e.message || "Cannot decode this WAV recording.",true);}
}
$("#audio-file").addEventListener("change",e=>select(e.target.files[0]));
const drop=$("#drop");
drop.addEventListener("dragover",e=>{e.preventDefault();drop.classList.add("drag");});
drop.addEventListener("dragleave",()=>drop.classList.remove("drag"));
drop.addEventListener("drop",e=>{e.preventDefault();drop.classList.remove("drag");const file=e.dataTransfer.files[0];if(file){const dt=new DataTransfer();dt.items.add(file);$("#audio-file").files=dt.files;select(file);}});
function paragraph(parent,text){const p=document.createElement("p");p.textContent=text;parent.append(p);}
function render(data){
  result=data;$("#download").disabled=false;$("#result").hidden=false;
  const audio=data.audio||{};$("#classification").textContent=audio.classification ? audio.classification+" acoustic pattern" : "No classification";
  $("#probability").textContent=Number.isFinite(audio.probability_abnormal)?`${(audio.probability_abnormal*100).toFixed(1)}% abnormal score`:"";
  const provenance=audio.provenance||{};
  $("#provenance").textContent=`Model: ${provenance.name||"unavailable"}${provenance.subset?" (smoke-subset model; not validated for clinical use)":""}. Run: ${provenance.run_id||"unknown"}`;
  $("#report").replaceChildren();
  if(data.report){paragraph($("#report"),data.report.summary);paragraph($("#report"),data.report.limitations);paragraph($("#report"),data.report.next_step);}
  else paragraph($("#report"),(data.audit?.errors||["No report generated."]).join(" "));
  $("#sources").replaceChildren();
  for(const source of data.sources||[]){const box=document.createElement("article");box.className="source";paragraph(box,`${source.source}, page ${source.page}`);paragraph(box,source.text);try{const url=new URL(source.url);if(url.protocol==="https:"){const a=document.createElement("a");a.href=url.href;a.target="_blank";a.rel="noopener noreferrer";a.textContent="Open original source";box.append(a);}}catch{}$("#sources").append(box);}
  paragraph($("#report"),data.disclaimer||"");
  message(data.status==="ok"?"Report passed the structural, citation and safety audit.":"Analysis completed with abstention or a rejected report.",data.status==="rejected");
}
function frame(event,data){
  if(event==="stage"){const item=document.querySelector(`[data-stage="${data.stage}"]`);if(item)item.className=data.status;message(`${data.stage.replaceAll("_"," ")}: ${data.status}`);}
  if(event==="token"){$("#draft-panel").hidden=false;$("#draft").textContent+=data.text;}
  if(event==="result")render(data);
  if(event==="error")message(data.message||"Analysis unavailable.",true);
}
$("#upload-form").addEventListener("submit",async e=>{
  e.preventDefault();if(!selectedFile){message("Wait for a valid recording to finish loading.",true);return;}
  controller=new AbortController();result=undefined;$("#download").disabled=true;$("#analyze").disabled=true;$("#cancel").disabled=false;$("#result").hidden=true;$("#draft").textContent="";$("#draft-panel").hidden=true;document.querySelectorAll("#stages li").forEach(n=>n.className="");
  const data=new FormData();data.append("file",selectedFile);data.append("prompt",$("#prompt").value);data.append("external_consent",String($("#consent").checked));message("Uploading recording…");
  try{
    const response=await fetch("/analyze_pcg/stream",{method:"POST",body:data,headers:headers(),signal:controller.signal});
    if(!response.ok){const error=await response.json();throw Error(typeof error.detail==="string"?error.detail:"The request was rejected.");}
    const reader=response.body.getReader(), decoder=new TextDecoder();let pending="",done=false;
    while(!done){const packet=await reader.read();done=packet.done;pending+=decoder.decode(packet.value||new Uint8Array(),{stream:!done}).replaceAll("\r\n","\n");let boundary;
      while((boundary=pending.indexOf("\n\n"))>=0){const raw=pending.slice(0,boundary);pending=pending.slice(boundary+2);const lines=raw.split("\n");const event=lines.find(l=>l.startsWith("event: "))?.slice(7)||"message";const text=lines.filter(l=>l.startsWith("data: ")).map(l=>l.slice(6)).join("\n");if(text)frame(event,JSON.parse(text));}}
    if(!result && !$("#message").classList.contains("error"))message("Stream ended without an audited result.",true);
  }catch(e){message(e.name==="AbortError"?"Analysis cancelled.":e.message,true);}finally{$("#analyze").disabled=false;$("#cancel").disabled=true;controller=undefined;}
});
$("#cancel").addEventListener("click",()=>controller?.abort());
$("#download").addEventListener("click",()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="cardioagent-research-report.json";a.click();URL.revokeObjectURL(url);});
status();
