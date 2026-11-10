import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { pipeline } from 'node:stream/promises';
import os from 'node:os';
import { Transform } from 'node:stream';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
loadEnv(path.join(ROOT, '.env'));
const PORT = Number(process.env.PORT || 4175);
const MAX_JSON_BYTES = 40 * 1024 * 1024;
const LOCAL_STT = 'http://127.0.0.1:5095';
const TRANSCRIPT_CACHE_DIR=path.join(ROOT,'.airwave-cache','transcripts');
let hostVoiceEmbedding = null;
const audioAssets=new Map();
let groqApiKey = process.env.GROQ_API_KEY || '';
let groqModels = { a: process.env.GROQ_MODEL_A || 'openai/gpt-oss-120b', b: process.env.GROQ_MODEL_B || 'qwen/qwen3.8-27b' };
const streamSessions = new Map();
let lastGroqRequestAt = 0;
let groqRequestQueue=Promise.resolve();
const GROQ_MIN_REQUEST_GAP_MS = 20000;

function loadEnv(file) {
  if (!fs.existsSync(file)) return;
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    const match = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (!match || process.env[match[1]] !== undefined) continue;
    let value = match[2].trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) value = value.slice(1, -1);
    process.env[match[1]] = value;
  }
}
function sendJson(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'content-length': Buffer.byteLength(body), 'cache-control': 'no-store' });
  res.end(body);
}
function cleanupStreamSession(id){const session=streamSessions.get(id);if(!session)return;streamSessions.delete(id);if(session.child&&session.child.exitCode===null)session.child.kill();for(const file of [session.audioPath,session.adPath])if(file)fs.promises.rm(file,{force:true}).catch(()=>{});}
function cleanupAudioAsset(id){const asset=audioAssets.get(id);if(!asset)return;audioAssets.delete(id);if(asset.timer)clearTimeout(asset.timer);if(asset.audioPath)fs.promises.rm(asset.audioPath,{force:true}).catch(()=>{});}
function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0; const chunks = [];
    req.on('data', chunk => { size += chunk.length; if (size > MAX_JSON_BYTES) { reject(Object.assign(new Error('Request body is too large.'), { status: 413 })); req.destroy(); return; } chunks.push(chunk); });
    req.on('end', () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')); } catch { reject(Object.assign(new Error('Request body must be valid JSON.'), { status: 400 })); } });
    req.on('error', reject);
  });
}
async function fetchWithTimeout(url, options, timeoutMs = 45000) {
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), timeoutMs);
  try { return await fetch(url, { ...options, signal: controller.signal }); }
  finally { clearTimeout(timer); }
}
function groqRequestWithGap(options){const call=groqRequestQueue.then(async()=>{const wait=Math.max(0,GROQ_MIN_REQUEST_GAP_MS-(Date.now()-lastGroqRequestAt));if(wait)await delay(wait);lastGroqRequestAt=Date.now();return fetchWithTimeout('https://api.groq.com/openai/v1/chat/completions',options,55000);});groqRequestQueue=call.then(()=>undefined,()=>undefined);return call;}
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function upstreamError(response, service) {
  let detail = '';
  try { const body = await response.json(); const error=body.error||{};const metadata=error.metadata;detail = error.message || body.detail?.message || body.message || body.detail || JSON.stringify(body);if(metadata?.raw)detail+=` (${String(metadata.raw).slice(0,180)})`;if(metadata?.limit_source)detail+=` [${String(metadata.limit_source).slice(0,70)}]`; }
  catch { try { detail = (await response.text()).slice(0, 280); } catch {} }
  return `${service} returned ${response.status}${detail ? `: ${detail}` : '.'}`;
}
function textModelConfig() {
  return { key: groqApiKey, ...groqModels };
}
function ffmpegAvailable(){const configured=process.env.FFMPEG_PATH;if(configured)return fs.existsSync(configured)&&fs.statSync(configured).isFile();const binary=process.platform==='win32'?'ffmpeg.exe':'ffmpeg';return String(process.env.PATH||'').split(path.delimiter).some(dir=>{try{return fs.statSync(path.join(dir,binary)).isFile();}catch{return false;}});}
async function runFfmpegToFile(args,outputPath,maxBytes=134217728){
  const binary=process.env.FFMPEG_PATH||'ffmpeg',child=spawn(binary,args,{windowsHide:true,stdio:['ignore','pipe','pipe']});let stderr='',bytes=0;
  child.stderr.on('data',chunk=>{stderr=(stderr+chunk.toString()).slice(-1600);});
  const limiter=new Transform({transform(chunk,encoding,callback){bytes+=chunk.length;if(bytes>maxBytes){callback(new Error('FFmpeg output exceeded the size limit.'));return;}callback(null,chunk);}});
  const exited=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',code=>code===0?resolve():reject(new Error(`FFmpeg exited with code ${code}${stderr?`: ${stderr}`:''}`)));});
  try{await Promise.all([pipeline(child.stdout,limiter,fs.createWriteStream(outputPath)),exited]);if(!bytes)throw new Error('FFmpeg produced an empty audio file.');}
  catch(error){if(child.exitCode===null)child.kill();await fs.promises.rm(outputPath,{force:true}).catch(()=>{});throw error;}
}
async function detectSilenceIntervals(audioPath,startSeconds,endSeconds){
  const binary=process.env.FFMPEG_PATH||'ffmpeg',duration=endSeconds-startSeconds;if(!audioPath||duration<=0)return[];
  const child=spawn(binary,['-hide_banner','-nostats','-loglevel','info','-ss',startSeconds.toFixed(3),'-i',audioPath,'-t',duration.toFixed(3),'-vn','-af','asetpts=PTS-STARTPTS,silencedetect=noise=-38dB:d=0.10','-f','null',process.platform==='win32'?'NUL':'-'],{windowsHide:true,stdio:['ignore','ignore','pipe']});let stderr='';child.stderr.on('data',chunk=>{stderr=(stderr+chunk.toString()).slice(-2_000_000);});
  await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',code=>code===0?resolve():reject(new Error(`FFmpeg silence analysis exited with code ${code}${stderr?`: ${stderr.slice(-800)}`:''}`)));});
  const intervals=[];let open=null;for(const match of stderr.matchAll(/silence_start:\s*([\d.]+)/g)){
    const relative=Number(match[1]);if(Number.isFinite(relative))open=relative;
    const endMatch=stderr.slice(match.index).match(/silence_end:\s*([\d.]+)/);if(endMatch&&open!==null){const end=Number(endMatch[1]);if(Number.isFinite(end)&&end>=open)intervals.push({startSeconds:Number((startSeconds+open).toFixed(3)),endSeconds:Number((startSeconds+end).toFixed(3))});open=null;}
  }
  if(open!==null)intervals.push({startSeconds:Number((startSeconds+open).toFixed(3)),endSeconds:Number(endSeconds.toFixed(3))});
  return intervals;
}
async function completion(model, messages, maxTokens = 900, audit = null) {
  const { key } = textModelConfig();
  if (!key) throw new Error('Add your Groq API key in Airwave settings.');
  const headers = { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' };
  const requestBody = { model, messages:[...messages], temperature: 0.35, max_completion_tokens: maxTokens, response_format: { type: 'json_object' } };
  // GPT-OSS otherwise spends a small response budget on hidden reasoning and can return no JSON content.
  if (model.includes('gpt-oss')) { requestBody.reasoning_effort = 'low'; requestBody.include_reasoning = false; }
  let lastError='',repairAttempted=false;
  const send=async body=>{const entry={model,request:structuredClone(body),startedAt:Date.now()};if(audit)audit.push(entry);let response;try{response=await groqRequestWithGap({method:'POST',headers,body:JSON.stringify(body)});}catch(error){entry.error=String(error?.message||error);entry.finishedAt=Date.now();throw error;}entry.httpStatus=response.status;try{entry.response=await response.clone().json();}catch{entry.response=await response.clone().text().catch(()=>null);}entry.finishedAt=Date.now();return response;};
  for(let attempt=0;attempt<4;attempt++){
    let response=await send(requestBody);
    // Some free/open model endpoints don't expose JSON-mode even when they accept chat completions.
    if(response.status===400){await response.arrayBuffer();const fallback={model,messages:requestBody.messages,temperature:.35,max_completion_tokens:requestBody.max_completion_tokens};if(requestBody.reasoning_effort)fallback.reasoning_effort=requestBody.reasoning_effort;if(requestBody.include_reasoning===false)fallback.include_reasoning=false;response=await send(fallback);}
    if(response.status===429||response.status>=500){lastError=await upstreamError(response,'Groq');if(attempt<3){const retryHeader=Number(response.headers.get('retry-after'))*1000;const retryBody=Number(lastError.match(/try again in\s+([\d.]+)s/i)?.[1]||0)*1000;await delay(Math.min(60000,Math.max(retryHeader,retryBody,attempt===0?4000:8000)+300));continue;}throw new Error(lastError);}
    if(!response.ok)throw new Error(await upstreamError(response,'Groq'));
    const data=await response.json(),raw=data.choices?.[0]?.message?.content;
    if(!raw){lastError=`The model returned no JSON content (finish reason: ${data.choices?.[0]?.finish_reason||'unknown'}).`;if(attempt<3){requestBody.max_completion_tokens=Math.min(1400,Math.max(maxTokens*2,800));requestBody.reasoning_effort='low';requestBody.include_reasoning=false;continue;}throw new Error(lastError);}
    try{const clean=raw.replace(/^```(?:json)?\s*|\s*```$/g,'').trim();const start=clean.indexOf('{'),end=clean.lastIndexOf('}');return JSON.parse(start>=0&&end>start?clean.slice(start,end+1):clean);}
    catch{if(repairAttempted||attempt>=3)throw new Error('The model response was not valid JSON after an automatic retry.');repairAttempted=true;requestBody.messages.push({role:'assistant',content:raw},{role:'user',content:'Your previous answer was not valid JSON. Return the same answer again as one complete, compact JSON object only. Do not use markdown or add text before or after the JSON.'});requestBody.max_completion_tokens=Math.min(900,Math.max(maxTokens,600));continue;}
  }
  throw new Error(lastError||'The model is temporarily unavailable.');
}
function clampScore(value, fallback = 70) { const n = Number(value); return Number.isFinite(n) ? Math.max(0, Math.min(100, Math.round(n))) : fallback; }
function requireScoreFields(value,label){const fields=['relevance','safety','naturalness','slotFit'];if(!value||typeof value!=='object'||fields.some(key=>!Number.isFinite(Number(value[key]))))throw new Error(`${label} omitted one or more numeric fit and safety scores.`);return Object.fromEntries(fields.map(key=>[key,clampScore(value[key])]));}
function cleanCandidate(result, model, advertisers, label, payload) {
  const ad = advertisers.find(item => item.id === result.advertiserId) || advertisers.find(item => item.name?.toLowerCase() === String(result.advertiserName || '').toLowerCase());
  if (!ad) throw new Error(`${label} picked an advertiser outside the catalog.`);
  const script = String(result.script || '').trim();
  if (script.length < 30 || script.length > 1600) throw new Error(`${label} returned an ad script outside the expected length.`);
  const anchor=payload.transcriptSegments.find(turn=>turn.id===String(result.anchorTurnId||''));
  if(!anchor)throw new Error(`${label} did not return an exact eligible host-turn ID.`);
  if(!(payload.eligibleAdMoments||[]).some(moment=>moment.anchorTurnId===String(anchor.id)))throw new Error(`${label} chose a host turn that was not sent as an eligible ad moment.`);
  if(anchor.speakerId!==payload.selectedHostSpeakerId)throw new Error(`${label} chose ${anchor.speakerId}, but the selected host is ${payload.selectedHostSpeakerId}.`);
  if(anchor.uncertain||anchor.overlap)throw new Error(`${label} chose a turn marked ${anchor.uncertain?'uncertain':'overlapping'}; select a clean completed host turn.`);
  const momentSeconds=Number(anchor.endSeconds);
  if(!Number.isFinite(momentSeconds)||momentSeconds<0||momentSeconds>Number(payload.searchEnd)||momentSeconds<Number(payload.searchStart))throw new Error(`${label} chose a turn with an invalid timestamp or outside the selected search range.`);
  const breakdown=requireScoreFields(result.scores||result.breakdown,label),confidence=Number(result.confidence);if(!Number.isFinite(confidence))throw new Error(`${label} omitted its numeric confidence estimate.`);
  const localMoment=(payload.eligibleAdMoments||[]).find(moment=>moment.anchorTurnId===String(anchor.id));
  return { modelId: model, label, advertiserId: ad.id, advertiserName: ad.name, script, anchorTurnId:anchor.id, momentSeconds, momentEvidence:String(anchor.text||''), localMomentScore:Number(localMoment?.localFitScore||0),localSignals:localMoment?.localSignals||[],audioPauseSeconds:Number(localMoment?.audioPauseSeconds||0),audioEvidence:localMoment?.audioEvidence||'turn_timing', score:Math.round(breakdown.relevance*.34+breakdown.safety*.3+breakdown.naturalness*.2+breakdown.slotFit*.16), confidence: Math.max(0.05, Math.min(0.99, confidence)), critique: String(result.reasoning || result.critique || 'Selected from the advertiser catalog using the supplied context.'), breakdown };
}
const MAX_MODEL_MOMENTS=5,MIN_MOMENT_SEPARATION_SECONDS=24;
function cleanTurnForModel(turn){return{id:String(turn.id),startSeconds:Number(turn.startSeconds),endSeconds:Number(turn.endSeconds),speakerId:String(turn.speakerId||''),text:String(turn.text||'').trim().slice(0,280)};}
function buildEligibleAdMoments(payload,{limit=true}={}){
  const turns=(Array.isArray(payload.transcriptSegments)?payload.transcriptSegments:[]).filter(turn=>Number.isFinite(Number(turn.startSeconds))&&Number.isFinite(Number(turn.endSeconds))).slice().sort((a,b)=>Number(a.startSeconds)-Number(b.startSeconds));
  const selectedHost=String(payload.selectedHostSpeakerId||''),searchStart=Number(payload.searchStart),searchEnd=Number(payload.searchEnd),pool=[];
  const isClean=turn=>turn&&!turn.uncertain&&!turn.overlap&&String(turn.speakerId||'')!=='UNKNOWN'&&String(turn.text||'').trim();
  const endsSentence=text=>/[.!?…]["'”’)}\]]*\s*$/u.test(String(text||'').trim());
  const dangling=/\b(?:and|but|because|so|if|when|which|that|to|of|with|for|as|or|then|though|while|although|who|what|where|how|a|an|the)\s*[.!?…]*["'”’)}\]]*$/iu;
  for(let i=0;i<turns.length-1;i++){
    const host=turns[i],next=turns[i+1],text=String(host.text||'').trim(),words=text.split(/\s+/).filter(Boolean);
    const start=Number(host.startSeconds),end=Number(host.endSeconds),nextStart=Number(next.startSeconds),gap=nextStart-end;
    if(!isClean(host)||host.speakerId!==selectedHost||start<searchStart||end>searchEnd||end<=start||words.length<4||words.length>80)continue;
    if(!isClean(next)||!Number.isFinite(gap)||gap<-.08||dangling.test(text))continue;
    const previous=i>0&&isClean(turns[i-1])?cleanTurnForModel(turns[i-1]):null;
    const silenceIntervals=Array.isArray(payload.audioSilenceIntervals)?payload.audioSilenceIntervals:[];
    const detectedSilence=silenceIntervals.find(item=>Number(item.startSeconds)<=end+.3&&Number(item.endSeconds)>=end-.2&&Number(item.endSeconds)<=nextStart+.12);
    const audioPause=detectedSilence?Math.max(0,Number(detectedSilence.endSeconds)-Math.max(end,Number(detectedSilence.startSeconds))):Math.max(0,gap);
    const terminal=endsSentence(text),question=/\?+["'”’)}\]]*\s*$/u.test(text),sameSpeakerContinuation=next.speakerId===host.speakerId&&gap<.9;
    if(question&&next.speakerId!==host.speakerId&&gap<1.2)continue;
    const cue=/\b(?:the point is|what matters is|the takeaway|in short|to sum up|anyway|before we move on|speaking of|that said|so that is why|so that's why)\b/i.test(text),signals=[];
    if(terminal)signals.push('terminal punctuation');else signals.push('no terminal punctuation');
    if(detectedSilence)signals.push(`measured audio pause ${audioPause.toFixed(2)}s`);else signals.push(`turn-timing gap ${audioPause.toFixed(2)}s`);
    if(question)signals.push('question may expect a reply');
    if(sameSpeakerContinuation)signals.push('same speaker resumes quickly');
    if(cue)signals.push('possible topic/summary bridge');
    const wordScore=words.length>=7&&words.length<=42?16:words.length>=5&&words.length<=60?11:6;
    const pauseScore=Math.round(14*Math.min(1,audioPause/1.1));
    const score=Math.max(0,Math.min(100,22+(terminal?21:8)+wordScore+pauseScore+(cue?14:0)-(question&&next.speakerId!==host.speakerId?27:0)-(sameSpeakerContinuation?18:0)));
    pool.push({anchorTurnId:String(host.id),endSeconds:end,gapSeconds:Number(gap.toFixed(2)),audioPauseSeconds:Number(audioPause.toFixed(2)),audioEvidence:detectedSilence?'ffmpeg_silence':'turn_timing',localFitScore:score,localSignals:signals,hostTurn:cleanTurnForModel(host),contextBefore:previous,contextAfter:cleanTurnForModel(next)});
  }
  pool.sort((a,b)=>b.localFitScore-a.localFitScore||a.endSeconds-b.endSeconds);
  if(!limit)return pool;
  const selected=[];
  for(const item of pool){if(selected.every(existing=>Math.abs(existing.endSeconds-item.endSeconds)>=MIN_MOMENT_SEPARATION_SECONDS)){selected.push(item);if(selected.length>=MAX_MODEL_MOMENTS)break;}}
  return selected;
}
function modelAdvertisers(payload){return(payload.advertisers||[]).map(a=>({id:a.id,name:String(a.name||'').slice(0,40),product:String(a.product||'').slice(0,80),audience:String(a.audience||'').slice(0,70),offer:String(a.offer||'').slice(0,80),facts:String(Array.isArray(a.facts)?a.facts.join('; '):a.facts||'').slice(0,120),avoid:String(Array.isArray(a.avoid)?a.avoid.join('; '):a.avoid||'').slice(0,100),tags:(a.tags||[]).slice(0,4)}));}
function candidatePrompt(payload, modelLabel) {
  const system = `You are ${modelLabel}, one of two independent podcast ad editors. Rank only the supplied local shortlist; these moments are exact completed-turn-end candidates and have passed overlap/uncertainty checks. Each has a local fit score and timing/acoustic evidence. Treat the score as a hint, not a decision. Prioritize conversational sequence: avoid interrupting an expected question-answer pair, an unfolding story, or a thought continued by the next turn. A pause is useful but not sufficient. Select the exact anchorTurnId; never invent a timestamp or another turn. If none is genuinely natural, return {"noFit":true,"reasoning":"..."}. Otherwise write a short, clearly disclosed sponsor transition bridging the completed thought to the supplied next turn. Use campaign facts only; do not imply host endorsement. Return JSON: {"advertiserId":"exact catalog id","anchorTurnId":"exact ID from eligibleAdMoments","script":"spoken copy","reasoning":"why this moment and advertiser fit","confidence":0.0,"scores":{"relevance":0-100,"safety":0-100,"naturalness":0-100,"slotFit":0-100}}. Keep within approximately ${Math.max(12,Math.floor(payload.slotSeconds*2.15))} words.`;
  const eligibleAdMoments=payload.eligibleAdMoments||[];
  const user = { episodeTopic: payload.topic, eligibleAdMoments, listenerInterests: (payload.interests||[]).slice(0,5), language: payload.locale, targetAdDurationSeconds: payload.slotSeconds, advertiserCatalog: modelAdvertisers(payload) };
  return [{ role: 'system', content: system }, { role: 'user', content: JSON.stringify(user) }];
}
function counterPrompt(payload, candidateA) {
  const system = `You are Perspective B. Independently review Perspective A and choose only among the supplied eligibleAdMoments. Pick an eligible moment, advertiser and sponsor script. Return its exact anchorTurnId; the server will insert at that turn's end. Do not invent timestamps. The moment must be a complete host thought with a natural pause. Clearly identify the sponsor early and hand back; don't imply personal endorsement; use campaign facts only. Compare A and B on moment fit, advertiser relevance, naturalness, safety and duration. Return JSON: {"candidateB":{"advertiserId":"exact catalog id","anchorTurnId":"exact eligibleAdMoments ID","script":"spoken copy","reasoning":"why this turn and ad fit","confidence":0.0,"scores":{"relevance":0-100,"safety":0-100,"naturalness":0-100,"slotFit":0-100}},"reviews":[{"candidate":0,"relevance":0-100,"safety":0-100,"naturalness":0-100,"slotFit":0-100,"confidence":0.0,"critique":"one concise sentence"},{"candidate":1,"relevance":0-100,"safety":0-100,"naturalness":0-100,"slotFit":0-100,"confidence":0.0,"critique":"one concise sentence"}]}. Keep scripts within approximately ${Math.max(12,Math.floor(payload.slotSeconds*2.15))} words.`;
  const user = { episodeTopic: payload.topic, eligibleAdMoments:payload.eligibleAdMoments||[], listenerInterests: (payload.interests||[]).slice(0,5), language: payload.locale, targetAdDurationSeconds: payload.slotSeconds, advertiserCatalog: modelAdvertisers(payload), perspectiveAProposal: { advertiserId: candidateA.advertiserId, advertiserName: candidateA.advertiserName, anchorTurnId:String(candidateA.anchorTurnId||''),script:String(candidateA.script||'').slice(0,500),reasoning:String(candidateA.reasoning||'').slice(0,240) } };
  return [{ role: 'system', content: system }, { role: 'user', content: JSON.stringify(user) }];
}
function revisionPrompt(payload, own, other, label, otherReview) {
  const system=`You are ${label} in round two of a two-model ad debate. Reconsider your proposal using only the supplied eligibleAdMoments and reviewer note. Keep an anchor only if it is an exact eligibleAdMoments anchorTurnId. The server inserts at that turn's end. Do not invent facts, claims, offers or timestamps. Make the sponsorship clear, bridge naturally from the candidate context, and avoid implying host endorsement. You may keep your draft if it remains best. Return only JSON: {"advertiserId":"exact catalog id","anchorTurnId":"exact eligible moment ID","script":"revised or retained copy","reasoning":"what you kept or changed and why","confidence":0.0,"scores":{"relevance":0-100,"safety":0-100,"naturalness":0-100,"slotFit":0-100}}. Keep within ${Math.max(12,Math.floor(payload.slotSeconds*2.15))} words.`;
  const user={episodeTopic:payload.topic,eligibleAdMoments:payload.eligibleAdMoments||[],listenerInterests:(payload.interests||[]).slice(0,5),language:payload.locale,targetAdDurationSeconds:payload.slotSeconds,advertiserCatalog:modelAdvertisers(payload),yourRoundOneProposal:own,otherRoundOneProposal:other,reviewerNoteAboutYourRoundOneProposal:otherReview};
  return [{role:'system',content:system},{role:'user',content:JSON.stringify(user)}];
}
function scoreDebateRound(candidate, review) {
  const own=candidate.breakdown||{};const avg=key=>Math.round((clampScore(own[key])+clampScore(review?.[key]))/2);
  candidate.breakdown={relevance:avg('relevance'),safety:avg('safety'),naturalness:avg('naturalness'),slotFit:avg('slotFit')};
  candidate.score=Math.round(candidate.breakdown.relevance*.34+candidate.breakdown.safety*.3+candidate.breakdown.naturalness*.2+candidate.breakdown.slotFit*.16);
  candidate.confidence=Math.min(.99,Math.max(.05,(candidate.confidence+(Number(review?.confidence)||candidate.confidence))/2));
  candidate.critique=[candidate.critique,String(review?.critique||'')].filter(Boolean).join(' ');return candidate;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const allowedOrigins=new Set([`http://localhost:${PORT}`,`http://127.0.0.1:${PORT}`]);
  if(req.headers.origin&&!allowedOrigins.has(req.headers.origin))return sendJson(res,403,{error:'Cross-origin requests are not allowed.'});
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }
  try {
    if (url.pathname === '/api/status' && req.method === 'GET') {
      const models = textModelConfig();let localStt=false;try{const probe=await fetchWithTimeout(`${LOCAL_STT}/api/upload-config`,{},1200);localStt=probe.ok&&(await probe.json()).enabled===true;}catch{}return sendJson(res, 200, { models: !!(models.key && models.a && models.b), groqKeySet:!!models.key,groqModels:{a:models.a,b:models.b}, gnani: !!process.env.GNANI_API_KEY,localStt,ffmpegReady:ffmpegAvailable(),voiceCloneReady:!!hostVoiceEmbedding });
    }
    if(url.pathname==='/api/transcript-cache'&&req.method==='GET'){
      const key=String(url.searchParams.get('key')||'');if(!/^[a-f0-9]{64}$/i.test(key))return sendJson(res,400,{error:'Invalid transcript cache key.'});const cachePath=path.join(TRANSCRIPT_CACHE_DIR,`${key.toLowerCase()}.json`);
      try{const saved=JSON.parse(await fs.promises.readFile(cachePath,'utf8'));if(Date.now()-Number(saved.savedAt)>30*24*60*60*1000){await fs.promises.rm(cachePath,{force:true});return sendJson(res,404,{error:'Transcript cache expired.'});}return sendJson(res,200,saved);}catch(error){if(error.code==='ENOENT')return sendJson(res,404,{error:'Transcript not cached.'});throw error;}
    }
    if(url.pathname==='/api/transcript-cache'&&req.method==='PUT'){
      const payload=await readJson(req),key=String(payload.key||'');if(!/^[a-f0-9]{64}$/i.test(key)||!payload.transcript||!Array.isArray(payload.transcript.turns))return sendJson(res,400,{error:'Invalid transcript cache entry.'});
      await fs.promises.mkdir(TRANSCRIPT_CACHE_DIR,{recursive:true});const cachePath=path.join(TRANSCRIPT_CACHE_DIR,`${key.toLowerCase()}.json`),tempPath=`${cachePath}.${randomUUID()}.tmp`;await fs.promises.writeFile(tempPath,JSON.stringify({...payload.transcript,savedAt:Date.now()}));await fs.promises.rename(tempPath,cachePath);return sendJson(res,200,{saved:true});
    }
    if(url.pathname==='/api/groq/settings'&&req.method==='POST'){const payload=await readJson(req);if(payload.apiKey!==undefined)groqApiKey=String(payload.apiKey).trim();if(payload.modelA)groqModels.a=String(payload.modelA).trim();if(payload.modelB)groqModels.b=String(payload.modelB).trim();if(!groqModels.a||!groqModels.b||groqModels.a===groqModels.b)return sendJson(res,400,{error:'Choose two different Groq model IDs.'});return sendJson(res,200,{ok:true,models:!!groqApiKey,groqModels});}
    if(url.pathname==='/api/groq/models'&&req.method==='GET'){if(!groqApiKey)return sendJson(res,401,{error:'Add a Groq API key first.'});const response=await fetchWithTimeout('https://api.groq.com/openai/v1/models',{headers:{Authorization:`Bearer ${groqApiKey}`}},15000);if(!response.ok)return sendJson(res,response.status,{error:await upstreamError(response,'Groq models')});const data=await response.json();return sendJson(res,200,{models:(data.data||[]).map(x=>({id:x.id,active:x.active!==false})).filter(x=>x.active).map(x=>x.id).sort()});}
    if(url.pathname==='/api/audio-assets'&&req.method==='POST'){
      const payload=await readJson(req),filename=String(payload.filename||'episode.mp3'),duration=Number(payload.duration);if(!/^[^\\/:\x00-\x1f]+\.(mp3|wav|m4a|mp4|flac|ogg|webm)$/i.test(filename))return sendJson(res,400,{error:'Choose a supported podcast audio file.'});if(!Number.isFinite(duration)||duration<=0||duration>8*60*60)return sendJson(res,400,{error:'The episode duration is invalid.'});
      const id=randomUUID(),asset={id,filename,duration,audioPath:'',silenceIntervals:null,createdAt:Date.now(),timer:null};asset.timer=setTimeout(()=>cleanupAudioAsset(id),60*60*1000);asset.timer.unref?.();audioAssets.set(id,asset);return sendJson(res,201,{audioAssetId:id,status:'upload-required'});
    }
    const audioAssetRoute=url.pathname.match(/^\/api\/audio-assets\/([\da-f-]+)$/i);
    if(audioAssetRoute&&req.method==='GET'){const asset=audioAssets.get(audioAssetRoute[1]);return asset?sendJson(res,200,{audioAssetId:asset.id,ready:!!asset.audioPath,filename:asset.filename}):sendJson(res,404,{error:'Local audio asset expired.'});}
    if(audioAssetRoute&&req.method==='DELETE'){cleanupAudioAsset(audioAssetRoute[1]);return sendJson(res,200,{deleted:true});}
    const audioAssetUpload=url.pathname.match(/^\/api\/audio-assets\/([\da-f-]+)\/audio$/i);
    if(audioAssetUpload&&req.method==='PUT'){
      const asset=audioAssets.get(audioAssetUpload[1]);if(!asset)return sendJson(res,404,{error:'Local audio asset expired.'});const audioPath=path.join(os.tmpdir(),`airwave-asset-${asset.id}${path.extname(asset.filename).toLowerCase()}`);let bytes=0;const limiter=new Transform({transform(chunk,encoding,callback){bytes+=chunk.length;if(bytes>134217728){callback(new Error('Podcast file exceeds the 128 MB stream limit.'));return;}callback(null,chunk);}});
      try{await pipeline(req,limiter,fs.createWriteStream(audioPath));if(!bytes)return sendJson(res,400,{error:'Podcast upload was empty.'});asset.audioPath=audioPath;asset.silenceIntervals=null;return sendJson(res,200,{status:'ready',audioAssetId:asset.id});}catch(error){await fs.promises.rm(audioPath,{force:true});if(bytes>134217728)return sendJson(res,413,{error:'Podcast file exceeds the 128 MB stream limit.'});throw error;}
    }
    if(url.pathname==='/api/local-transcribe'&&req.method==='POST'){
      const length=Number(req.headers['content-length']||0),filename=decodeURIComponent(String(req.headers['x-filename']||''));
      if(!Number.isInteger(length)||length<1||length>134217728)return sendJson(res,413,{error:'Choose an audio file under 128 MB.'});
      if(!/^[^\\/:\x00-\x1f]+\.(mp3|wav|m4a|mp4|flac|ogg|webm)$/i.test(filename))return sendJson(res,400,{error:'The local STT project does not support this audio filename.'});
      if(!ffmpegAvailable())return sendJson(res,424,{error:'FFmpeg is required to crop the search area without decoding the full episode in your browser. Set FFMPEG_PATH in .env and restart Airwave.'});
      const offset=Number(url.searchParams.get('offset')),start=Number(url.searchParams.get('start')),end=Number(url.searchParams.get('end'));
      if(![offset,start,end].every(Number.isFinite)||start<0||end<=start||end-start>1800||Math.abs(offset-start)>.01)return sendJson(res,400,{error:'Choose a valid search area up to 30 minutes long.'});
      const sourcePath=path.join(os.tmpdir(),`airwave-scan-${randomUUID()}${path.extname(filename).toLowerCase()}`),clipPath=path.join(os.tmpdir(),`airwave-scan-${randomUUID()}.wav`);let size=0;
      const sourceLimiter=new Transform({transform(chunk,encoding,callback){size+=chunk.length;if(size>134217728){callback(Object.assign(new Error('Audio exceeds the local STT size limit.'),{status:413}));return;}callback(null,chunk);}});let keepSource=false,audioAssetId='';
      try{
        await pipeline(req,sourceLimiter,fs.createWriteStream(sourcePath));if(size!==length)throw Object.assign(new Error('Audio upload was incomplete.'),{status:400});
        await runFfmpegToFile(['-hide_banner','-loglevel','error','-ss',start.toFixed(3),'-i',sourcePath,'-t',(end-start).toFixed(3),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','-f','wav','pipe:1'],clipPath,128*1024*1024);
        const clipSize=(await fs.promises.stat(clipPath)).size;let upload;
        try{upload=await fetchWithTimeout(`${LOCAL_STT}/api/uploads`,{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Filename':encodeURIComponent(`${path.parse(filename).name}-search.wav`),'Content-Length':String(clipSize)},body:fs.createReadStream(clipPath),duplex:'half'},90000);}catch{return sendJson(res,503,{error:'Start your local STT project with Start-Listening-Room.cmd, then try again.'});}
        if(!upload.ok)return sendJson(res,upload.status,{error:(await upload.json().catch(()=>({}))).error||'Local STT could not accept this episode.'});const job=await upload.json();
        const jobId=job.id;if(!/^[a-f0-9]{32}$/.test(jobId||''))throw new Error('Local STT returned an invalid job identifier.');
        audioAssetId=randomUUID();const asset={id:audioAssetId,filename,duration:0,audioPath:sourcePath,silenceIntervals:null,createdAt:Date.now(),timer:null};asset.timer=setTimeout(()=>cleanupAudioAsset(audioAssetId),60*60*1000);asset.timer.unref?.();audioAssets.set(audioAssetId,asset);keepSource=true;
        return sendJson(res,202,{jobId,audioAssetId,status:job.status||'queued',message:job.message||'Queued for local processing.'});
      }catch(error){if(error.code==='ECONNREFUSED'||error.name==='AbortError')return sendJson(res,503,{error:'Start your local STT project with Start-Listening-Room.cmd, then try again.'});if(error.code==='ENOENT')return sendJson(res,424,{error:'FFmpeg could not be found. Set FFMPEG_PATH to the full path of ffmpeg.exe, then restart Airwave.'});if(error.message.startsWith('FFmpeg exited'))return sendJson(res,422,{error:`FFmpeg could not crop this audio: ${error.message}`});throw error;}
      finally{await Promise.all([keepSource?Promise.resolve():fs.promises.rm(sourcePath,{force:true}).catch(()=>{}),fs.promises.rm(clipPath,{force:true}).catch(()=>{})]);}
    }
    if(url.pathname.startsWith('/api/local-transcribe/')&&req.method==='GET'){
      const jobId=url.pathname.split('/').pop();if(!/^[a-f0-9]{32}$/.test(jobId||''))return sendJson(res,400,{error:'Invalid local STT job identifier.'});
      const poll=await fetchWithTimeout(`${LOCAL_STT}/api/jobs/${jobId}`,{},5000);if(!poll.ok)return sendJson(res,poll.status,{error:'Local STT job could not be checked.'});const status=await poll.json();
      if(status.status==='failed')return sendJson(res,422,{status:'failed',error:status.message||'Local diarization failed.'});
      if(status.status!=='completed')return sendJson(res,200,{status:status.status,message:status.message||'Local transcription is processing.',progress:null});
      const docRes=await fetchWithTimeout(`${LOCAL_STT}/recordings/${jobId}/transcript.json`,{},10000);if(!docRes.ok)throw new Error('Local STT completed but did not return its transcript.');const doc=await docRes.json();const backend=doc.default_backend||'nemo_native';const turns=doc.variants?.[backend]?.turns||[];
      const offset=Number(url.searchParams.get('offset')||0),from=0,to=Number(doc.duration||0);
      if(!Number.isFinite(offset)||offset<0)return sendJson(res,400,{error:'Invalid search-range offset.'});
      const mapped=turns.filter(t=>Number(t.end)>from&&Number(t.start)<to&&String(t.text||'').trim()).map((t,i)=>({id:String(t.id||`${backend}-${i}`),startSeconds:Number(t.start)+offset,endSeconds:Number(t.end)+offset,speakerId:String(t.speaker||'UNKNOWN'),text:String(t.text||''),uncertain:!!t.uncertain||t.speaker==='UNKNOWN',overlap:!!t.overlap,wordCount:String(t.text||'').trim().split(/\s+/).length,intersectsSearch:false}));
      return sendJson(res,200,{status:'completed',progress:100,turns:mapped,backend,duration:Number(doc.duration||0),notice:doc.notice||''});
    }
    if(url.pathname==='/api/voice-clone'&&req.method==='POST'){
      if(!process.env.GNANI_API_KEY)return sendJson(res,424,{error:'GNANI_API_KEY is not configured on this server.'});const payload=await readJson(req);
      if(payload.consent!==true)return sendJson(res,400,{error:'Confirm that this is your voice and you consent to Gnani processing it.'});if(typeof payload.audioBase64!=='string')return sendJson(res,400,{error:'Choose a clean voice sample.'});
      const bytes=Buffer.from(payload.audioBase64,'base64');if(bytes.length>12*1024*1024)return sendJson(res,413,{error:'Choose a voice sample under 12 MB.'});const filename=String(payload.filename||'voice.wav').replace(/[^\w.-]/g,'_');const mime=String(payload.mimeType||'audio/wav');
      const form=new FormData();form.append('audio_file',new Blob([bytes],{type:mime}),filename);const response=await fetchWithTimeout('https://api.vachana.ai/api/v1/tts/voice-clone/embeddings',{method:'POST',headers:{'X-API-Key-ID':process.env.GNANI_API_KEY},body:form},90000);
      if(!response.ok)return sendJson(res,response.status,{error:await upstreamError(response,'Gnani voice clone')});const data=await response.json();const embedding=data.data?.voice_clone_embedding;if(!embedding?.embedding||!Array.isArray(embedding.shape))return sendJson(res,502,{error:'Gnani returned an invalid voice embedding.'});hostVoiceEmbedding=embedding;return sendJson(res,200,{ready:true});
    }
    if(url.pathname==='/api/voice-clone/forget'&&req.method==='POST'){hostVoiceEmbedding=null;return sendJson(res,200,{ready:false});}
    if (url.pathname === '/api/ads' && req.method === 'POST') {
      const payload = await readJson(req); const config = textModelConfig();
      const llmRequests=[];
      if (!config.key || !config.a || !config.b) return sendJson(res, 424, { error: 'Add a Groq API key and choose two different model IDs in Airwave settings to enable live ad matching.' });
      if (config.a === config.b) return sendJson(res, 400, { error: 'Model A and Model B need different model IDs so they can evaluate the moment independently.' });
      if (!Array.isArray(payload.advertisers) || !payload.advertisers.length) return sendJson(res, 400, { error: 'Add at least one advertiser to the catalog.' });
      let first, second;
      try {
        if(!Array.isArray(payload.transcriptSegments)||!payload.transcriptSegments.length)return sendJson(res,400,{error:'Transcribe the search area or enter a timestamped transcript before matching.'});
        if(!payload.selectedHostSpeakerId)return sendJson(res,400,{error:'Choose which detected speaker is you before matching.'});
        const audioAsset=audioAssets.get(String(payload.audioAssetId||''));
        if(audioAsset?.audioPath){try{audioAsset.silenceIntervals=await detectSilenceIntervals(audioAsset.audioPath,Number(payload.searchStart),Number(payload.searchEnd));}catch(error){console.warn('Local silence analysis unavailable:',error.message);audioAsset.silenceIntervals=[];}}
        payload.audioSilenceIntervals=audioAsset?.silenceIntervals||[];
        payload.eligibleAdMoments=buildEligibleAdMoments(payload);
        if(!payload.eligibleAdMoments.length)return sendJson(res,422,{error:'No clean completed host turn with enough surrounding context was found. No transcript was sent to Groq. Expand the search area or check the selected speaker.',llmRequests});
        const candidateA=await completion(config.a, candidatePrompt(payload, 'Perspective A'), 400,llmRequests);
        if(candidateA.noFit)return sendJson(res,200,{candidates:[],winnerIndex:-1,eligibleMomentCount:payload.eligibleAdMoments.length,audioEvidence:audioAsset?.audioPath?'ffmpeg_silencedetect':'turn_timing_only',topic:payload.topic,debate:true,rounds:1,summary:'Perspective A found no natural, safe sponsor position; the episode will play without an ad.',llmRequests});
        first = cleanCandidate(candidateA, config.a, payload.advertisers, 'Perspective A',payload);
        const counter = await completion(config.b, counterPrompt(payload, first), 600,llmRequests);
        if(counter.candidateB?.noFit)return sendJson(res,200,{candidates:[],winnerIndex:-1,eligibleMomentCount:payload.eligibleAdMoments.length,audioEvidence:audioAsset?.audioPath?'ffmpeg_silencedetect':'turn_timing_only',topic:payload.topic,debate:true,rounds:1,summary:'Perspective B found no natural, safe sponsor position; no ad was selected.',llmRequests});
        second = cleanCandidate(counter.candidateB || {}, config.b, payload.advertisers, 'Perspective B',payload);
        const reviews = Array.isArray(counter.reviews) ? counter.reviews : [];
        const reviewA=reviews.filter(x=>Number(x.candidate)===0),reviewB=reviews.filter(x=>Number(x.candidate)===1);if(reviewA.length!==1||reviewB.length!==1)throw new Error('Perspective B must return exactly one review for each proposal.');
        for(const [i,review] of [[0,reviewA[0]],[1,reviewB[0]]]){requireScoreFields(review,`Perspective B review of proposal ${i?'B':'A'}`);if(!Number.isFinite(Number(review.confidence)))throw new Error(`Perspective B omitted confidence for proposal ${i?'B':'A'}.`);}
        const roundOneA=scoreDebateRound({...first},reviewA[0]);const roundOneB=scoreDebateRound({...second},reviewB[0]);
        const revisedA=await completion(config.a,revisionPrompt(payload,candidateA,counter.candidateB,'Perspective A',reviewA[0].critique),400,llmRequests);
        const revisedB=await completion(config.b,revisionPrompt(payload,counter.candidateB,candidateA,'Perspective B',reviewB[0].critique),400,llmRequests);
        const finalA=scoreDebateRound(cleanCandidate(revisedA,config.a,payload.advertisers,'Perspective A · revised',payload),revisedA.scores);
        const finalB=scoreDebateRound(cleanCandidate(revisedB,config.b,payload.advertisers,'Perspective B · revised',payload),revisedB.scores);
        const candidates=[roundOneA,roundOneB,finalA,finalB];
        for(const candidate of candidates)candidate.safeToPreview=candidate.breakdown.safety>=60&&candidate.breakdown.relevance>=55&&candidate.breakdown.slotFit>=48;
        const winner=candidates.filter(c=>c.breakdown.safety>=60&&c.breakdown.relevance>=55&&c.breakdown.slotFit>=48).sort((a,b)=>b.score-a.score)[0]||null;
        return sendJson(res,200,{candidates,winnerIndex:winner?candidates.indexOf(winner):-1,eligibleMomentCount:payload.eligibleAdMoments.length,audioEvidence:audioAsset?.audioPath?'ffmpeg_silencedetect':'turn_timing_only',topic:payload.topic,reviewer:config.b,debate:true,rounds:2,summary:winner?`Round two selected ${winner.label} at ${winner.momentEvidence}`:'No candidate cleared minimum fit and safety thresholds.',llmRequests});
      } catch (error) {
        return sendJson(res,502,{error:`Two-model ad review failed: ${error.message}`,llmRequests});
      }
    }
    if (url.pathname === '/api/shorten' && req.method === 'POST') {
      const payload = await readJson(req); const config = textModelConfig();
      if (!config.key || !config.b) return sendJson(res, 424, { error: 'A text model is required to shorten this script.' });
      const result = await completion(config.b,[{role:'system',content:`Shorten this spoken advertisement to fit within ${Math.max(8,Math.floor(Number(payload.slotSeconds)*2.15))} words. Preserve campaign facts and the offer. Do not add claims. Return JSON only: {"script":"..."}.`},{role:'user',content:JSON.stringify({script:payload.script,advertiser:payload.advertiser,language:payload.locale})}],350);
      return sendJson(res, 200, { script: String(result.script || '').trim() });
    }
    if(url.pathname==='/api/stream-sessions'&&req.method==='POST'){
      const payload=await readJson(req),insertAt=Number(payload.insertAt),duration=Number(payload.duration),filename=String(payload.filename||'episode.mp3');
      if(!Number.isFinite(insertAt)||insertAt<0)return sendJson(res,400,{error:'A verified turn-end timestamp is required.'});
      if(!Number.isFinite(duration)||duration<=insertAt||duration>8*60*60)return sendJson(res,400,{error:'The episode duration must be valid and no longer than eight hours.'});
      const audioAsset=audioAssets.get(String(payload.audioAssetId||''));if(!audioAsset?.audioPath)return sendJson(res,404,{error:'The local episode upload expired. Scan or upload the episode again.'});
      const anchor=(Array.isArray(payload.transcriptSegments)?payload.transcriptSegments:[]).find(turn=>turn.id===String(payload.anchorTurnId||''));
      if(!anchor||anchor.speakerId!==String(payload.hostSpeakerId||'')||anchor.uncertain||anchor.overlap||!Number.isFinite(Number(anchor.endSeconds))||Math.abs(Number(anchor.endSeconds)-insertAt)>.05)return sendJson(res,400,{error:'The ad must be anchored to the exact end of a clean turn from the selected host in this episode.'});
      const eligibleMoments=buildEligibleAdMoments({transcriptSegments:payload.transcriptSegments,selectedHostSpeakerId:payload.hostSpeakerId,searchStart:payload.searchStart,searchEnd:payload.searchEnd,audioSilenceIntervals:audioAsset.silenceIntervals||[]},{limit:false});
      if(!eligibleMoments.some(moment=>moment.anchorTurnId===String(anchor.id)))return sendJson(res,400,{error:'This ad moment no longer passes the clean-boundary checks. Run matching again with the current transcript.'});
      if(insertAt>=duration)return sendJson(res,400,{error:'The selected host turn must end before the episode does.'});
      if(!/^[^\\/:\x00-\x1f]+\.(mp3|wav|m4a|mp4|flac|ogg|webm)$/i.test(filename))return sendJson(res,400,{error:'Choose a supported podcast audio file.'});
      if(typeof payload.adAudioBase64!=='string')return sendJson(res,400,{error:'Generate the sponsor read before starting the stream.'});
      const ad=Buffer.from(payload.adAudioBase64,'base64');if(ad.length<44||ad.subarray(0,4).toString()!=='RIFF')return sendJson(res,400,{error:'Sponsor audio must be a WAV file.'});
      const id=randomUUID(),adPath=path.join(os.tmpdir(),`airwave-${id}-ad.wav`);await fs.promises.writeFile(adPath,ad);
      const session={id,insertAt,duration,filename,adPath,audioPath:'',audioAssetId:audioAsset.id,createdAt:Date.now(),active:false,anchorTurnId:String(anchor.id),hostSpeakerId:String(anchor.speakerId)};streamSessions.set(id,session);
      const expiry=setTimeout(()=>cleanupStreamSession(id),(duration+30*60)*1000);expiry.unref?.();
      return sendJson(res,201,{sessionId:id,status:'ready'});
    }
    const streamUpload=url.pathname.match(/^\/api\/stream-sessions\/([\da-f-]+)\/audio$/i);
    if(streamUpload&&req.method==='PUT'){
      const session=streamSessions.get(streamUpload[1]);if(!session)return sendJson(res,404,{error:'Stream session expired.'});
      const filename=decodeURIComponent(String(req.headers['x-filename']||session.filename));if(!/^[^\\/:\x00-\x1f]+\.(mp3|wav|m4a|mp4|flac|ogg|webm)$/i.test(filename))return sendJson(res,400,{error:'Unsupported podcast filename.'});
      const audioPath=path.join(os.tmpdir(),`airwave-${session.id}-original${path.extname(filename).toLowerCase()}`);let bytes=0;const limiter=new Transform({transform(chunk,encoding,callback){bytes+=chunk.length;if(bytes>134217728){callback(new Error('Podcast file exceeds the 128 MB stream limit.'));return;}callback(null,chunk);}});
      try{await pipeline(req,limiter,fs.createWriteStream(audioPath));}catch(error){await fs.promises.rm(audioPath,{force:true});if(bytes>134217728)return sendJson(res,413,{error:'Podcast file exceeds the 128 MB stream limit.'});throw error;}
      if(!bytes)return sendJson(res,400,{error:'Podcast upload was empty.'});session.audioPath=audioPath;session.filename=filename;
      return sendJson(res,200,{status:'ready',sessionId:session.id});
    }
    const streamRequest=url.pathname.match(/^\/api\/stream-sessions\/([\da-f-]+)$/i);
    if(streamRequest&&req.method==='DELETE'){cleanupStreamSession(streamRequest[1]);return sendJson(res,200,{deleted:true});}
    if(streamRequest&&req.method==='GET'){
      const session=streamSessions.get(streamRequest[1]),audioAsset=session?.audioAssetId?audioAssets.get(session.audioAssetId):null,sourcePath=audioAsset?.audioPath||session?.audioPath;if(!session||!sourcePath)return sendJson(res,404,{error:'Stream is not ready. Upload the podcast audio first.'});
      if(session.active)return sendJson(res,409,{error:'This stream session is already playing.'});session.active=true;
      const preview=url.searchParams.get('mode')==='preview',beforeSeconds=Math.max(3,Math.min(20,Number(url.searchParams.get('before')||10))),afterSeconds=Math.max(3,Math.min(20,Number(url.searchParams.get('after')||10))),previewStart=preview?Math.max(0,session.insertAt-beforeSeconds):0,previewEnd=preview?Math.min(session.duration,session.insertAt+afterSeconds):session.duration,insertRelative=session.insertAt-previewStart,sourceEnd=previewEnd-previewStart,ffmpeg=process.env.FFMPEG_PATH||'ffmpeg',t=Number(session.insertAt).toFixed(3);
      const filter=`[0:a]atrim=start=0:end=${insertRelative.toFixed(3)},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[before];[1:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,loudnorm=I=-18:TP=-2:LRA=7,afade=t=in:st=0:d=0.12,areverse,afade=t=in:st=0:d=0.18,areverse[ad];[0:a]atrim=start=${insertRelative.toFixed(3)}:end=${sourceEnd.toFixed(3)},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[after];[before][ad][after]concat=n=3:v=0:a=1[out]`;
      const inputArgs=preview?['-ss',previewStart.toFixed(3),'-re','-i',sourcePath]:['-re','-i',sourcePath];
      const child=spawn(ffmpeg,['-hide_banner','-loglevel','error',...inputArgs,'-i',session.adPath,'-filter_complex',filter,'-map','[out]','-c:a','libmp3lame','-q:a','3','-f','mp3','-flush_packets','1','pipe:1'],{windowsHide:true,stdio:['ignore','pipe','pipe']});session.child=child;let stderr='',started=false;
      child.stderr.on('data',chunk=>{stderr=(stderr+chunk.toString()).slice(-1200);});
      child.stdout.on('data',chunk=>{if(!started){started=true;res.writeHead(200,{'content-type':'audio/mpeg','cache-control':'no-store','transfer-encoding':'chunked','x-airwave-insert-at':t});}if(!res.write(chunk)){child.stdout.pause();res.once('drain',()=>child.stdout.resume());}});
      child.on('error',error=>{session.active=false;if(!started)sendJson(res,503,{error:error.code==='ENOENT'?'FFmpeg is required for continuous podcast streaming. Set FFMPEG_PATH to ffmpeg.exe or add it to PATH.':error.message});});
      child.on('close',code=>{session.active=false;if(!started&&!res.headersSent)sendJson(res,502,{error:`FFmpeg could not start the audio stream${stderr?`: ${stderr}`:''}`});else if(!res.writableEnded)res.end();if(!preview)cleanupStreamSession(session.id);});
      res.on('close',()=>{if(child.exitCode===null)child.kill();});return;
    }
    if (url.pathname === '/api/tts' && req.method === 'POST') {
      if (!process.env.GNANI_API_KEY) return sendJson(res, 424, { error: 'GNANI_API_KEY is not configured on this server.' });
      const payload=await readJson(req);const text=String(payload.text||'').trim();if(!text||text.length>3000)return sendJson(res,400,{error:'Add an ad script under 3,000 characters.'});
      const language=String(payload.language||'en-IN'),model=process.env.GNANI_TTS_MODEL||'timbre-v2.5';
      const v25Voices={'en-IN':'Kaveri','hi-IN':'Nalini','hi-en':'Poorvi'},v30Voices={'en-IN':'Shivani','hi-IN':'Shivani','hi-en':'Shivani'};
      const selectedVoice=process.env.GNANI_TTS_VOICE||((model==='timbre-v3.0'?v30Voices:v25Voices)[language]||'Kaveri');
      const useClone=payload.useHostVoice===true&&!!hostVoiceEmbedding;
      const body=useClone?{text,model:'vachana-vc-v1',speaker_embedding:hostVoiceEmbedding,audio_config:{sample_rate:24000,num_channels:1,sample_width:2,encoding:'linear_pcm',container:'wav'}}:{text,voice:selectedVoice,model,language:language==='hi-en'?'hi-en':language,speed:1.0,audio_config:{sample_rate:24000,num_channels:1,sample_width:2,encoding:'linear_pcm',container:'wav'}};
      const response=await fetchWithTimeout('https://api.vachana.ai/api/v1/tts/inference',{method:'POST',headers:{'X-API-Key-ID':process.env.GNANI_API_KEY,'Content-Type':'application/json'},body:JSON.stringify(body)},90000);
      if(!response.ok)return sendJson(res,response.status,{error:await upstreamError(response,'Gnani TTS')});const audio=Buffer.from(await response.arrayBuffer());
      res.writeHead(200,{'content-type':'audio/wav','content-length':audio.length,'cache-control':'no-store'});return res.end(audio);
    }
    if (req.method === 'GET' || req.method === 'HEAD') {
      let file = url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname.slice(1));
      const resolved = path.resolve(ROOT, file);
      if (!resolved.startsWith(ROOT + path.sep) && resolved !== path.join(ROOT,'index.html')) return sendJson(res,403,{error:'Forbidden.'});
      if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) return sendJson(res,404,{error:'Not found.'});
      const ext=path.extname(resolved);const types={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.json':'application/json'};
      res.writeHead(200,{'content-type':types[ext]||'application/octet-stream','cache-control':ext==='.html'?'no-store':'public, max-age=300'});if(req.method==='HEAD')return res.end();return fs.createReadStream(resolved).pipe(res);
    }
    return sendJson(res,404,{error:'Unknown endpoint.'});
  } catch(error) {
    console.error(error); if(!res.headersSent)sendJson(res,error.status||500,{error:error.name==='AbortError'?'The provider took too long to respond.':error.message||'Unexpected server error.'});else res.destroy();
  }
});

server.listen(PORT,'127.0.0.1',()=>console.log(`Airwave demo is ready at http://localhost:${PORT}`));
