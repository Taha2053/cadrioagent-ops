"use strict";
// Browsers often reject 2 kHz WAV in decodeAudioData; decode linear PCM directly.
function pcmWave(bytes) {
  const view=new DataView(bytes), tag=(offset)=>String.fromCharCode(...new Uint8Array(bytes,offset,4));
  if(view.byteLength<44||tag(0)!=="RIFF"||tag(8)!=="WAVE")throw Error("Invalid RIFF/WAVE file.");
  let channels,rate,bits,format,start,length;
  for(let offset=12;offset+8<=view.byteLength;){const size=view.getUint32(offset+4,true),end=offset+8+size;if(end>view.byteLength)throw Error("Truncated WAV file.");
    if(tag(offset)==="fmt "){if(size<16)throw Error("Invalid WAV format.");format=view.getUint16(offset+8,true);channels=view.getUint16(offset+10,true);rate=view.getUint32(offset+12,true);bits=view.getUint16(offset+22,true);}
    if(tag(offset)==="data"){start=offset+8;length=size;}offset=end+(size%2);}
  if(format!==1||![8,16,24,32].includes(bits)||!channels||channels>8||!rate||start===undefined)throw Error("Browser preview requires linear PCM WAV.");
  const stride=bits/8,frames=Math.floor(length/(channels*stride));if(frames/rate>120)throw Error("Recording exceeds 120 seconds.");
  const signal=new Float32Array(frames);
  for(let i=0;i<frames;i++)for(let c=0;c<channels;c++){const p=start+(i*channels+c)*stride;let value;
    if(bits===8)value=(view.getUint8(p)-128)/128;
    else if(bits===16)value=view.getInt16(p,true)/32768;
    else if(bits===24){let n=view.getUint8(p)|(view.getUint8(p+1)<<8)|(view.getUint8(p+2)<<16);if(n&0x800000)n-=0x1000000;value=n/8388608;}
    else value=view.getInt32(p,true)/2147483648;
    signal[i]+=value/channels;}
  if(!frames)throw Error("Recording is empty.");
  return {signal,sampleRate:rate,duration:frames/rate};
}
function playbackWave(signal,rate){
  const target=Math.max(8000,rate),count=Math.ceil(signal.length*target/rate),bytes=new ArrayBuffer(44+count*2),v=new DataView(bytes);
  const text=(p,s)=>[...s].forEach((c,i)=>v.setUint8(p+i,c.charCodeAt(0)));
  text(0,"RIFF");v.setUint32(4,36+count*2,true);text(8,"WAVE");text(12,"fmt ");v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,target,true);v.setUint32(28,target*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);text(36,"data");v.setUint32(40,count*2,true);
  for(let i=0;i<count;i++){const position=i*rate/target,lo=Math.floor(position),fraction=position-lo;const value=signal[lo]*(1-fraction)+signal[Math.min(lo+1,signal.length-1)]*fraction;v.setInt16(44+i*2,Math.max(-1,Math.min(1,value))*32767,true);}
  return new Blob([bytes],{type:"audio/wav"});
}
globalThis.CardioWave = {decode: pcmWave, playback: playbackWave};
