const DEFAULT_ADVERTISERS = [
  { id: 'daybreak', name: 'Daybreak Coffee', product: 'Small-batch coffee made for a slower start', audience: 'People who enjoy thoughtful morning rituals', offer: 'A first bag for 15% off with code FIRSTLIGHT', facts: ['roasted in small batches', 'ships in recyclable packaging'], avoid: ['do not claim health benefits', 'do not claim it improves focus'], tags: ['coffee', 'morning ritual', 'slow living'], color: '#e9dfce', mark: 'D' },
  { id: 'clearspace', name: 'Clearspace', product: 'Guided audio pauses for a little breathing room', audience: 'People looking for simple moments of calm', offer: 'Try the first seven days free', facts: ['sessions are guided audio', 'most sessions take five to ten minutes'], avoid: ['do not claim to treat anxiety', 'do not promise better sleep'], tags: ['wellness', 'calm', 'mindfulness'], color: '#dfeade', mark: 'C' },
  { id: 'fieldnotes', name: 'Field Notes Club', product: 'A pocket journal for everyday observations', audience: 'Curious people who like to reflect and write', offer: 'Get a free pocket notebook with your first set', facts: ['recycled paper', 'designed to fit in a jacket pocket'], avoid: ['do not claim journaling improves mental health'], tags: ['journaling', 'books', 'creativity'], color: '#e7e1d7', mark: 'F' },
  { id: 'mossandbloom', name: 'Moss & Bloom', product: 'A simple, fragrance-light daily skincare set', audience: 'People who prefer a short, uncomplicated routine', offer: 'Save 10% on the starter set', facts: ['three step routine', 'fragrance-light formulas'], avoid: ['do not claim to cure skin conditions', 'do not promise specific results'], tags: ['skincare', 'self care', 'slow living'], color: '#e5eadc', mark: 'M' },
  { id: 'openroad', name: 'Open Road Outdoors', product: 'Easy day hikes and local outdoor guides', audience: 'People who want to spend more unhurried time outside', offer: 'Explore the free weekend trail guide', facts: ['guides are free to browse', 'covers beginner-friendly routes'], avoid: ['do not claim every route is accessible', 'do not guarantee conditions'], tags: ['outdoors', 'hiking', 'travel'], color: '#dce6e7', mark: 'O' },
  { id: 'turnleaf', name: 'Turnleaf Tea', product: 'Loose leaf blends for a gentle afternoon pause', audience: 'Tea drinkers and people who enjoy quiet rituals', offer: 'A sampler tin with your first order', facts: ['four caffeine-free blends in the sampler', 'blended in small batches'], avoid: ['do not claim to help people sleep', 'do not claim medical benefits'], tags: ['tea', 'slow living', 'wellness'], color: '#e2eadc', mark: 'T' }
];

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = {
  advertisers: loadStore('airwave-advertisers', DEFAULT_ADVERTISERS),
  interests: loadStore('airwave-interests', ['slow living', 'wellness', 'good coffee']),
  file: null, audioBuffer: null, objectUrl: '', duration: 120, slotStart: 20, slotEnd: 35, insertAt: 23, transcriptSegments: [], momentChosen:false,
  liveModels: false, gnaniReady: false, localSttReady:false,ffmpegReady:false, hostVoiceReady:false, useHostVoice:false, hostVoiceFile:null, transcriptSearchRange:null, transcriptCacheKey:'',hostSpeakerId:'',audioAssetId:'',audioAssetKey:'',preparedSessions:{},resultPayload:null,candidates: [], winner: null, debateWinnerIndex:null, renderedUrl: '',
  sampleContext: true, sampleEpisode: false, audioContext: null,episodeRevision:0
};

function loadStore(key, fallback) {
  try { const value = JSON.parse(localStorage.getItem(key)); return Array.isArray(value) && value.length ? value : structuredClone(fallback); }
  catch { return structuredClone(fallback); }
}
function saveStore(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} }
function esc(value = '') { return String(value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch])); }
function secondsLabel(value) { const seconds = Math.max(0, Math.floor(Number(value) || 0)); return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`; }
function tokenSet(value) { return new Set(String(value).toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, ' ').split(/\s+/).filter(x => x.length > 2)); }
function notify(message) { const el = $('#toast'); el.textContent = message; el.classList.add('show'); clearTimeout(notify.timer); notify.timer = setTimeout(() => el.classList.remove('show'), 3500); }
function setBusy(button, busy, label) { button.disabled = busy; if (busy) { button.dataset.oldHtml = button.innerHTML; button.innerHTML = `<span class="button-sparkle">✳</span> ${esc(label)}`; } else if (button.dataset.oldHtml) { button.innerHTML = button.dataset.oldHtml; delete button.dataset.oldHtml; } }

async function init() {
  renderAdvertisers(); renderInterestTags(); drawPlaceholderWave(); setupRanges(); setupUpload(); setupActions(); setupLlmHistory(); setupGroqSettings();
  $('#advertiser-count').textContent = String(state.advertisers.length).padStart(2, '0');
  $('#eligible-count').textContent = `${state.advertisers.length} advertisers available`;
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const cfg = await res.json(); state.liveModels = !!cfg.models; state.gnaniReady = !!cfg.gnani;state.localSttReady=!!cfg.localStt;state.ffmpegReady=!!cfg.ffmpegReady;state.hostVoiceReady=!!cfg.voiceCloneReady;state.useHostVoice=state.hostVoiceReady;
      $('#use-host-voice').disabled=!state.hostVoiceReady;$('#use-host-voice').checked=state.useHostVoice;
      $('#voice-clone-status').textContent=state.hostVoiceReady?'CLONE READY':'STOCK VOICE';$('#forget-voice-clone').classList.toggle('hidden',!state.hostVoiceReady);
      $('#transcript-status').textContent=!state.ffmpegReady?'Set FFMPEG_PATH in .env and restart Airwave to crop and stream audio':state.localSttReady?'Local speaker diarization ready · transcribe the episode to choose the host voice':'Start your local STT project at 127.0.0.1:5095 to get speaker timestamps';
      const status = $('.api-status'); const mode = $('#mode-pill');
      if (state.liveModels || state.gnaniReady) {
        status.classList.add('live'); $('#api-status-label').textContent = state.liveModels && state.gnaniReady ? 'Connected' : 'Partially connected';
        $('#api-status-copy').textContent = state.liveModels && state.gnaniReady ? (state.ffmpegReady?'AI + Gnani ready':'AI + Gnani ready · FFmpeg needed') : state.liveModels ? 'AI ready · Gnani needed' : 'Gnani ready · model keys needed';
        $('#api-dot').style.background = '#51a171'; mode.textContent = 'API CONNECTED'; $('.demo-badge').classList.add('live');
      }
    }
  } catch { /* Opening the HTML directly still allows the demo UI to be explored. */ }
}

function renderAdvertisers() {
  const root = $('#advertiser-list');
  root.innerHTML = state.advertisers.map(ad => `<article class="advertiser-card" data-id="${esc(ad.id)}"><span class="advertiser-match">POSSIBLE FIT</span><div class="advertiser-top"><span class="advertiser-logo" style="background:${esc(ad.color || '#e7eadf')};color:#50644e">${esc(ad.mark || ad.name.slice(0,1))}</span><span><b>${esc(ad.name)}</b><small>${esc(ad.audience)}</small></span></div><p class="advertiser-desc">${esc(ad.product)}</p><div class="advertiser-tags">${(ad.tags || []).slice(0,3).map(tag => `<span>${esc(tag)}</span>`).join('')}</div></article>`).join('');
  $('#advertiser-count').textContent = String(state.advertisers.length).padStart(2, '0');
  $('#eligible-count').textContent = `${state.advertisers.length} advertisers available`;
}
function renderInterestTags() {
  const holder = $('#interest-tags');
  $$('.interest-tag', holder).forEach(tag => tag.remove());
  const input = $('#interest-input');
  state.interests.forEach((interest, index) => {
    const tag = document.createElement('button'); tag.type = 'button'; tag.className = 'interest-tag'; tag.innerHTML = `${esc(interest)} <span>×</span>`;
    tag.addEventListener('click', () => { state.interests.splice(index, 1); saveStore('airwave-interests', state.interests); renderInterestTags(); });
    holder.insertBefore(tag, input);
  });
}

function drawPlaceholderWave() {
  const canvas = $('#waveform'); const rect = canvas.getBoundingClientRect(); const dpr = devicePixelRatio || 1;
  canvas.width = Math.max(300, Math.round((rect.width || 620) * dpr)); canvas.height = Math.max(40, Math.round((rect.height || 67) * dpr));
  const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height); const mid=canvas.height/2; const bars=Math.floor(canvas.width/(3*dpr));
  ctx.fillStyle = '#c5d2bd';
  for(let i=0;i<bars;i++){ const pattern=(Math.sin(i*.32)+Math.sin(i*.071+1.3)+Math.cos(i*.19+2))*.17+.37; const h=(.12+Math.abs(pattern)*.76)*canvas.height*.68; ctx.fillRect(i*3*dpr,mid-h/2,1.3*dpr,h); }
  updateSlotVisual();
}
function drawWave(buffer) {
  const canvas=$('#waveform'), rect=canvas.getBoundingClientRect(), dpr=devicePixelRatio||1;
  canvas.width=Math.max(300,Math.round(rect.width*dpr)); canvas.height=Math.max(40,Math.round(rect.height*dpr));
  const ctx=canvas.getContext('2d'), channel=buffer.getChannelData(0), bars=Math.floor(canvas.width/(3*dpr)), step=Math.max(1,Math.floor(channel.length/bars)), mid=canvas.height/2;
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#b8cbb0';
  for(let i=0;i<bars;i++){ let peak=0; for(let j=0;j<step;j++) peak=Math.max(peak,Math.abs(channel[i*step+j]||0)); const h=Math.max(2*dpr,peak*canvas.height*.82); ctx.fillRect(i*3*dpr,mid-h/2,1.35*dpr,h); }
  updateSlotVisual();
}
function setupRanges() {
  const start=$('#start-range'), end=$('#end-range');
  const update = (which) => {
    let s=Number(start.value), e=Number(end.value); const gap=Math.min(2, state.duration/10);
    if(which==='start' && s>=e-gap){s=Math.max(0,e-gap);start.value=s;}
    if(which==='end' && e<=s+gap){e=Math.min(state.duration,s+gap);end.value=e;}
    const maxSpan=Math.min(1800,state.duration);if(e-s>maxSpan){if(which==='start')s=Math.max(0,e-maxSpan);else e=Math.min(state.duration,s+maxSpan);start.value=s;end.value=e;$('#transcript-status').textContent='Search range shortened to the local STT 30-minute limit.';}
    Object.values(state.preparedSessions).forEach(id=>fetch(`/api/stream-sessions/${id}`,{method:'DELETE'}).catch(()=>{}));state.preparedSessions={};state.resultPayload=null;state.slotStart=s; state.slotEnd=e; state.insertAt=Math.max(s,Math.min(e,state.insertAt));state.episodeRevision++;state.transcriptSegments=[];state.transcriptSearchRange=null;state.transcriptCacheKey='';state.hostSpeakerId='';state.winner=null;state.candidates=[];$('#host-speaker-row').classList.add('hidden');$('#results-panel').classList.add('hidden');state.momentChosen=false;updateSlotVisual();
  };
  start.addEventListener('input',()=>update('start')); end.addEventListener('input',()=>update('end'));
}
function updateSlotVisual() {
  const dur=Math.max(1,state.duration), start=$('#start-range'),end=$('#end-range');
  start.max=dur; end.max=dur; start.value=Math.min(state.slotStart,dur);end.value=Math.min(state.slotEnd,dur);
  const left=100*state.slotStart/dur, width=100*(state.slotEnd-state.slotStart)/dur;
  const overlay=$('#slot-overlay');overlay.style.left=`${left}%`;overlay.style.width=`${width}%`;
  const point=$('#point-label');point.style.left=`${100*state.insertAt/dur}%`;point.classList.toggle('hidden',!state.momentChosen);
  $('#insert-time-label').textContent=state.momentChosen?secondsLabel(state.insertAt):'After transcript scan';
  $('#slot-start-label').textContent=secondsLabel(state.slotStart);$('#slot-end-label').textContent=secondsLabel(state.slotEnd);
}
function findQuietPoint(buffer, lower, upper) {
  const samples=buffer.getChannelData(0),rate=buffer.sampleRate,windowFrames=Math.min(Math.floor(rate*.55),Math.max(1,Math.floor((upper-lower)*rate*.45)));
  const margin=Math.min(.45,Math.max(0,(upper-lower)*.1)),first=Math.max(lower+margin,0),last=Math.max(first,upper-margin-windowFrames/rate);
  let best=first,bestEnergy=Infinity;const sampleStride=Math.max(1,Math.floor(rate*.012));
  for(let time=first;time<=last;time+=.1){const from=Math.floor(time*rate),to=Math.min(samples.length,from+windowFrames);let sum=0,count=0;for(let i=from;i<to;i+=sampleStride){const x=samples[i]||0;sum+=x*x;count++;}const energy=count?sum/count:0;if(energy<bestEnergy){bestEnergy=energy;best=time+windowFrames/rate/2;}}
  return Math.max(lower,Math.min(upper,best));
}
function setupUpload() {
  const input=$('#audio-file'), drop=$('#dropzone');
  $('#choose-file').addEventListener('click',()=>input.click());
  $('#load-example').addEventListener('click',()=>loadExampleEpisode());
  input.addEventListener('change',()=>{if(input.files?.[0]) loadAudioFile(input.files[0]);});
  drop.addEventListener('click',(event)=>{if(!event.target.closest('button'))input.click();});
  for(const event of ['dragenter','dragover']) drop.addEventListener(event,e=>{e.preventDefault();drop.classList.add('dragover');});
  for(const event of ['dragleave','drop']) drop.addEventListener(event,e=>{e.preventDefault();drop.classList.remove('dragover');});
  drop.addEventListener('drop',e=>{const f=e.dataTransfer.files?.[0];if(f)loadAudioFile(f);});
  $('#clear-audio').addEventListener('click',clearAudio);
  window.addEventListener('resize',()=>state.audioBuffer?drawWave(state.audioBuffer):drawPlaceholderWave());
}
async function loadAudioFile(file) {
  if(!file.type.startsWith('audio/')) return notify('Choose an audio file to continue.');
  if(file.size>100*1024*1024) return notify('This demo accepts audio files up to 100 MB.');
  const loadRevision=++state.episodeRevision;let nextUrl='';
  try {
    $('#upload-title').textContent='Loading your episode…';
    nextUrl=URL.createObjectURL(file);const probe=new Audio();probe.preload='metadata';probe.src=nextUrl;
    const duration=await new Promise((resolve,reject)=>{probe.onloadedmetadata=()=>resolve(probe.duration);probe.onerror=()=>reject(new Error('This browser cannot read the audio metadata. Try MP3 or WAV.'));probe.load();});
    if(!Number.isFinite(duration)||duration<=0)throw new Error('The audio duration could not be read.');
    if(loadRevision!==state.episodeRevision){URL.revokeObjectURL(nextUrl);return;}
    if(state.objectUrl)URL.revokeObjectURL(state.objectUrl);state.objectUrl=nextUrl;
    if(state.audioAssetId)fetch(`/api/audio-assets/${state.audioAssetId}`,{method:'DELETE'}).catch(()=>{});state.audioAssetId='';state.audioAssetKey='';state.preparedSessions={};state.resultPayload=null;
    state.file=file;state.audioBuffer=null;state.duration=duration;state.sampleEpisode=false;state.transcriptSegments=[];state.transcriptSearchRange=null;state.transcriptCacheKey='';state.hostSpeakerId='';state.momentChosen=false;state.candidates=[];state.winner=null;
    state.slotStart=Math.min(20,Math.max(0,duration*.2));state.slotEnd=Math.min(duration,state.slotStart+Math.min(120,Math.max(25,duration*.35)));state.insertAt=state.slotStart;
    const player=$('#episode-audio');player.src=state.objectUrl;player.preload='metadata';player.classList.remove('hidden');
    $('#audio-card').classList.remove('hidden');$('#dropzone').classList.add('hidden');$('#audio-name').textContent=file.name;
    $('#audio-info').textContent=`${secondsLabel(buffer.duration)} · ${Math.round(file.size/1024/1024*10)/10} MB · Search area ${secondsLabel(state.slotStart)}–${secondsLabel(state.slotEnd)}`;
    $('#upload-title').textContent=file.name;$('#upload-subtitle').textContent='Episode loaded · choose or drag a different file';
    $('#recent-title').textContent=file.name.length>20?file.name.slice(0,18)+'…':file.name;drawPlaceholderWave();updateSlotVisual();
    state.sampleContext=false;$('#episode-topic').value='';$('#context-text').value='';$('#transcript-status').textContent=`Suggested lower-energy point at ${secondsLabel(state.insertAt)} · transcribe around the point or add a note`;
    $('#host-speaker-row').classList.add('hidden');$('#results-panel').classList.add('hidden');$('#render-result').classList.add('hidden');
    notify('Episode ready. Mark the moment that needs an ad break.');
  } catch(error) { if(nextUrl&&nextUrl!==state.objectUrl)URL.revokeObjectURL(nextUrl);if(loadRevision===state.episodeRevision){console.error(error);$('#upload-title').textContent=state.file?.name||'Drop a podcast episode here';notify(error.message||'This audio format could not be opened. Try an MP3 or WAV file.');} }
}
function clearAudio() {
  if(state.objectUrl){URL.revokeObjectURL(state.objectUrl);state.objectUrl='';}
  if(state.audioAssetId)fetch(`/api/audio-assets/${state.audioAssetId}`,{method:'DELETE'}).catch(()=>{});state.audioAssetId='';state.audioAssetKey='';state.preparedSessions={};state.resultPayload=null;
  state.episodeRevision++;state.file=null;state.audioBuffer=null;state.sampleEpisode=false;state.duration=120;state.slotStart=20;state.slotEnd=35;state.insertAt=23;state.transcriptSegments=[];state.transcriptSearchRange=null;state.transcriptCacheKey='';state.hostSpeakerId='';state.candidates=[];state.winner=null;state.momentChosen=false;
  const p=$('#episode-audio');p.pause();p.removeAttribute('src');p.classList.add('hidden');$('#audio-card').classList.add('hidden');$('#dropzone').classList.remove('hidden');$('#upload-title').textContent='Drop a podcast episode here';$('#upload-subtitle').innerHTML='or <button class="inline-link" id="choose-file">browse files</button> · MP3, WAV, M4A';
  $('#host-speaker-row').classList.add('hidden');$('#results-panel').classList.add('hidden');$('#render-result').classList.add('hidden');$('#choose-file').addEventListener('click',()=>$('#audio-file').click());$('#episode-topic').value='';$('#context-text').value='';$('#transcript-status').textContent='Add an episode to transcribe its moment';drawPlaceholderWave();updateSlotVisual();
}
async function loadExampleEpisode() {
  // A low-key locally synthesized sample lets the matching/review flow be explored without a podcast file.
  const seconds=74,sampleRate=24000,Offline=window.OfflineAudioContext||window.webkitOfflineAudioContext;
  if(!Offline) return notify('Your browser does not support the sample audio. Upload an episode instead.');
  try {
    const offline=new Offline(1,seconds*sampleRate,sampleRate);const master=offline.createGain();master.gain.value=.035;master.connect(offline.destination);
    const freqs=[110,164.8,220,261.6];
    freqs.forEach((freq,i)=>{const osc=offline.createOscillator(),gain=offline.createGain();osc.type=i%2?'triangle':'sine';osc.frequency.value=freq;gain.gain.value=i===0?.42:.14;osc.connect(gain);gain.connect(master);osc.start();});
    const buffer=await offline.startRendering();state.episodeRevision++;state.audioBuffer=buffer;state.file=null;state.duration=seconds;state.slotStart=12;state.slotEnd=58;state.insertAt=findQuietPoint(buffer,state.slotStart,state.slotEnd);state.transcriptSegments=[];state.transcriptCacheKey='';state.hostSpeakerId='';state.momentChosen=false;state.sampleEpisode=true;
    const player=$('#episode-audio');player.removeAttribute('src');player.classList.add('hidden');$('#audio-card').classList.remove('hidden');$('#dropzone').classList.add('hidden');$('#audio-name').textContent='The little things · sample';$('#audio-info').textContent=`01:14 · generated sample audio · Search area 00:12–00:58`;$('#upload-title').textContent='Sample episode is loaded';$('#upload-subtitle').textContent='A local audio bed for exploring the flow';$('#recent-title').textContent='The little things';
    $('#episode-topic').value='Morning routines and small habits';$('#context-text').value='A conversation about small, repeatable habits that make mornings calmer — leaving your phone alone, taking a quiet moment, and starting the day with intention.';$('#transcript-status').textContent='Sample transcript context · not from a real recording';state.sampleContext=true;drawWave(buffer);updateSlotVisual();$('#results-panel').classList.add('hidden');notify('Sample episode loaded. Try a topic-aware advertiser match.');
  } catch(e){console.error(e);notify('Could not load the sample episode. Upload an audio file instead.');}
}

function setupActions() {
  $('#transcribe-button').addEventListener('click',()=>transcribeSearchArea().then(()=>notify('Episode analyzed locally. Choose which speaker is you.')).catch(error=>notify(error.message||'Could not transcribe the search area.')));
  $('#host-speaker-select').addEventListener('change',event=>{state.episodeRevision++;state.hostSpeakerId=event.target.value;state.momentChosen=false;state.winner=null;$('#results-panel').classList.add('hidden');if(state.transcriptCacheKey)cacheTranscript(state.transcriptCacheKey,{turns:state.transcriptSegments,hostSpeakerId:state.hostSpeakerId});});
  $('#choose-voice-reference').addEventListener('click',()=>$('#voice-reference-file').click());
  $('#voice-reference-file').addEventListener('change',event=>{state.hostVoiceFile=event.target.files?.[0]||null;$('#voice-clone-statusline').textContent=state.hostVoiceFile?`${state.hostVoiceFile.name} · confirm consent to prepare clone.`:'No voice sample uploaded.';$('#create-voice-clone').disabled=!state.hostVoiceFile||!$('#voice-consent').checked;});
  $('#voice-consent').addEventListener('change',event=>$('#create-voice-clone').disabled=!state.hostVoiceFile||!event.target.checked);
  $('#create-voice-clone').addEventListener('click',prepareHostVoice);
  $('#forget-voice-clone').addEventListener('click',forgetHostVoice);
  $('#use-host-voice').addEventListener('change',event=>state.useHostVoice=event.target.checked);
  $('#find-match').addEventListener('click',findMatch);
  $('#render-ad').addEventListener('click',streamEpisodeWithAd);
  $('#play-preview').addEventListener('click',previewEpisode);
  $('#add-advertiser').addEventListener('click',()=>$('#advertiser-dialog').showModal());
  $('#advertiser-form').addEventListener('submit',addAdvertiser);
  $('#edit-listener').addEventListener('click',()=>{$('#profile-edit-input').value=state.interests.join(', ');$('#profile-dialog').showModal();});
  $('#save-profile').addEventListener('click',event=>{event.preventDefault();state.interests=$('#profile-edit-input').value.split(',').map(x=>x.trim()).filter(Boolean).slice(0,8);saveStore('airwave-interests',state.interests);renderInterestTags();$('#profile-dialog').close();});
  $('#interest-input').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===','){event.preventDefault();const value=event.target.value.trim().replace(/,$/,'');if(value&&!state.interests.some(x=>x.toLowerCase()===value.toLowerCase())){state.interests.push(value);saveStore('airwave-interests',state.interests);renderInterestTags();}event.target.value='';}});
  $('#listener-name').addEventListener('input',()=>{$('.listener-card b').textContent=`${$('#listener-name').value||'Listener'}’s listening profile`;});
}
async function prepareHostVoice(){
  const button=$('#create-voice-clone');if(!state.hostVoiceFile||!$('#voice-consent').checked)return notify('Choose your voice sample and confirm consent first.');if(state.hostVoiceFile.size>12*1024*1024)return notify('Choose a clean sample under 12 MB.');
  setBusy(button,true,'Preparing clone…');$('#voice-clone-statusline').textContent='Sending your voice sample to Gnani for voice embedding…';
  try{const data=await decodeBlob(state.hostVoiceFile);if(data.duration<5||data.duration>30)throw new Error('Use a clean single-speaker sample between 5 and 30 seconds.');const response=await fetch('/api/voice-clone',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:state.hostVoiceFile.name,mimeType:state.hostVoiceFile.type||'audio/wav',audioBase64:await blobBase64(state.hostVoiceFile),consent:true})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Gnani could not prepare the voice clone.');state.hostVoiceReady=true;state.useHostVoice=true;$('#use-host-voice').disabled=false;$('#use-host-voice').checked=true;$('#voice-clone-status').textContent='CLONE READY';$('#forget-voice-clone').classList.remove('hidden');$('#voice-clone-statusline').textContent='Clone ready · held in server memory until it stops.';notify('Your voice clone is ready for the sponsor read.');}
  catch(error){$('#voice-clone-statusline').textContent=error.message;notify(error.message);}
  finally{setBusy(button,false);button.disabled=!state.hostVoiceFile||!$('#voice-consent').checked;}
}
async function forgetHostVoice(){try{await fetch('/api/voice-clone/forget',{method:'POST'});}catch{}state.hostVoiceReady=false;state.useHostVoice=false;$('#use-host-voice').checked=false;$('#use-host-voice').disabled=true;$('#voice-clone-status').textContent='STOCK VOICE';$('#forget-voice-clone').classList.add('hidden');$('#voice-clone-statusline').textContent='Clone forgotten.';}
function addAdvertiser(event) {
  event.preventDefault();const form=event.currentTarget, data=new FormData(form);const name=data.get('name').trim();
  const words=(value)=>String(value).split(/[;,]/).map(x=>x.trim()).filter(Boolean);
  const advertiser={id:`custom-${Date.now()}`,name,product:data.get('product').trim(),audience:data.get('audience').trim(),offer:data.get('offer').trim(),facts:words(data.get('facts')),avoid:words(data.get('avoid')),tags:[...words(data.get('audience')), ...words(data.get('product'))].flatMap(x=>x.toLowerCase().split(/[^\p{L}\p{N}]+/u)).filter(x=>x.length>3).slice(0,3),color:'#e9e8df',mark:name.split(/\s+/).map(x=>x[0]).join('').slice(0,2).toUpperCase()};
  state.advertisers.push(advertiser);saveStore('airwave-advertisers',state.advertisers);renderAdvertisers();form.reset();$('#advertiser-dialog').close();notify(`${name} added to the advertiser marketplace.`);
}

async function blobBase64(blob) { const buf=await blob.arrayBuffer();let binary='';const bytes=new Uint8Array(buf);for(let i=0;i<bytes.length;i+=0x8000)binary+=String.fromCharCode(...bytes.subarray(i,Math.min(i+0x8000,bytes.length)));return btoa(binary); }
let transcriptDbPromise;
function openTranscriptDb(){if(!('indexedDB'in window))return Promise.resolve(null);if(!transcriptDbPromise)transcriptDbPromise=new Promise(resolve=>{const request=indexedDB.open('airwave-transcript-cache',2);request.onupgradeneeded=()=>{const db=request.result;if(!db.objectStoreNames.contains('runs'))db.createObjectStore('runs');if(!db.objectStoreNames.contains('llm-requests'))db.createObjectStore('llm-requests',{keyPath:'id'});};request.onsuccess=()=>resolve(request.result);request.onerror=()=>resolve(null);});return transcriptDbPromise;}
async function saveLlmLog(entry){try{const db=await openTranscriptDb();if(!db)return;await new Promise(resolve=>{const req=db.transaction('llm-requests','readwrite').objectStore('llm-requests').put(entry);req.onsuccess=req.onerror=()=>resolve();});}catch{}}
async function readLlmLogs(){const db=await openTranscriptDb();if(!db)return[];return new Promise(resolve=>{const req=db.transaction('llm-requests','readonly').objectStore('llm-requests').getAll();req.onsuccess=()=>resolve((req.result||[]).sort((a,b)=>b.startedAt-a.startedAt));req.onerror=()=>resolve([]);});}
function setupLlmHistory(){const dialog=$('#llm-history-dialog'),list=$('#llm-history-list');const refresh=async()=>{const logs=await readLlmLogs();list.innerHTML=logs.length?logs.map((x,i)=>`<details style="border:1px solid #e8eae3;border-radius:6px;padding:9px;margin:7px 0"><summary>${esc(new Date(x.startedAt).toLocaleString())} · ${esc(x.kind)} · ${esc(x.status||'pending')}</summary><p><b>${x.kind==='ads'?'Local request summary (full transcript omitted)':'Browser request to /api/'+esc(x.kind)}</b></p><pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow:auto;background:#f7f6f1;padding:9px">${esc(JSON.stringify(x.request,null,2))}</pre><p><b>Actual model requests (prompts, retries, provider replies)</b></p><pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;background:#f7f6f1;padding:9px">${esc(JSON.stringify(x.response?.llmRequests??[],null,2))}</pre><p><b>Application response</b></p><pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:300px;overflow:auto;background:#f7f6f1;padding:9px">${esc(JSON.stringify(x.response??x.error??null,null,2))}</pre></details>`).join(''):'<p>No LLM requests recorded yet. New requests are saved as they are sent.</p>';};
  $('#llm-history-open').addEventListener('click',async()=>{await refresh();dialog.showModal();});$('#llm-history-close').addEventListener('click',()=>dialog.close());$('#llm-history-refresh').addEventListener('click',refresh);$('#llm-history-clear').addEventListener('click',async()=>{const db=await openTranscriptDb();if(db)await new Promise(resolve=>{const req=db.transaction('llm-requests','readwrite').objectStore('llm-requests').clear();req.onsuccess=req.onerror=()=>resolve();});await refresh();notify('LLM request history cleared.');});}
function setupGroqSettings(){const d=$('#groq-settings-dialog'),s=$('#groq-settings-status');$('#groq-settings-open').onclick=async()=>{const r=await fetch('/api/status'),x=await r.json();if(x.groqModels){$('#groq-model-a').value=x.groqModels.a;$('#groq-model-b').value=x.groqModels.b;}$('#groq-api-key').value='';s.textContent=x.groqKeySet?'Key set for this server session. Leave blank to keep.':'No Groq key set.';d.showModal();};$('#groq-settings-close').onclick=()=>d.close();$('#groq-settings-form').onsubmit=async e=>{e.preventDefault();s.textContent='Saving…';try{const r=await fetch('/api/groq/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({apiKey:$('#groq-api-key').value||undefined,modelA:$('#groq-model-a').value,modelB:$('#groq-model-b').value})}),x=await r.json();if(!r.ok)throw Error(x.error);$('#groq-api-key').value='';s.textContent='Saved in local server memory.';await refreshApiStatus();}catch(x){s.textContent=x.message;}};$('#groq-load-models').onclick=async()=>{s.textContent='Loading model list…';try{const r=await fetch('/api/groq/models'),x=await r.json();if(!r.ok)throw Error(x.error);s.textContent=`Groq returned ${x.models.length} models. Enter two model IDs above.`;}catch(x){s.textContent=x.message;}};}
async function refreshApiStatus(){try{const r=await fetch('/api/status'),x=await r.json();state.liveModels=!!x.models;state.ffmpegReady=!!x.ffmpegReady;$('#mode-pill').textContent=state.liveModels?'LIVE MODE':'DEMO ENVIRONMENT';$('#api-status-label').textContent=state.liveModels?'Connected':'Needs setup';$('#api-status-copy').textContent=state.liveModels?(state.ffmpegReady?'Groq ready':'Groq ready · FFmpeg needed'):'Add Groq model key';}catch{}}
async function loggedJsonRequest(kind,request){const historyRequest=kind==='ads'?{topic:request.topic,selectedHostSpeakerId:request.selectedHostSpeakerId,searchStart:request.searchStart,searchEnd:request.searchEnd,transcriptTurnCount:request.transcriptSegments?.length||0,advertisers:request.advertisers,interests:request.interests,locale:request.locale,slotSeconds:request.slotSeconds,localAudioAssetUsed:Boolean(request.audioAssetId),filter:'ranked clean host turn ends; completed-thought and question/answer risk signals; local silence/audio gap evidence when available; at most five diverse moments with adjacent context'}:request;const entry={id:crypto.randomUUID(),kind,startedAt:Date.now(),request:historyRequest,status:'sending'};await saveLlmLog(entry);try{const response=await fetch(`/api/${kind}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});let body;try{body=await response.json();}catch{body={error:'Response was not valid JSON'};}entry.status=response.ok?'completed':'failed';entry.response=body;entry.finishedAt=Date.now();await saveLlmLog(entry);return{ok:response.ok,status:response.status,json:async()=>body};}catch(error){entry.status='failed';entry.error=String(error?.message||error);entry.finishedAt=Date.now();await saveLlmLog(entry);throw error;}}
async function getCachedTranscript(key,legacyPrefix=''){try{const response=await fetch(`/api/transcript-cache?key=${encodeURIComponent(key)}`);if(response.ok)return await response.json();}catch{}try{const db=await openTranscriptDb();if(!db)return null;const current=await new Promise(resolve=>{const request=db.transaction('runs','readonly').objectStore('runs').get(key);request.onsuccess=()=>resolve(request.result||null);request.onerror=()=>resolve(null);});if(current)return current;if(legacyPrefix){const keys=await new Promise(resolve=>{const request=db.transaction('runs','readonly').objectStore('runs').getAllKeys();request.onsuccess=()=>resolve(request.result||[]);request.onerror=()=>resolve([]);}),oldKey=keys.find(item=>typeof item==='string'&&item.startsWith(`${legacyPrefix}|`));if(oldKey)return await new Promise(resolve=>{const request=db.transaction('runs','readonly').objectStore('runs').get(oldKey);request.onsuccess=()=>resolve(request.result||null);request.onerror=()=>resolve(null);});}return null;}catch{return null;}}
async function cacheTranscript(key,value){try{const db=await openTranscriptDb();if(db)await new Promise(resolve=>{const request=db.transaction('runs','readwrite').objectStore('runs').put({...value,savedAt:Date.now()},key);request.onsuccess=request.onerror=()=>resolve();});}catch{}try{await fetch('/api/transcript-cache',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({key,transcript:value})});}catch{}}
async function makeTranscriptCacheKey(file,rangeStart,rangeEnd){let fingerprint=`${file.name}|${file.size}|${file.lastModified}|${rangeStart.toFixed(3)}|${rangeEnd.toFixed(3)}`;try{if(crypto.subtle&&file.size){const n=Math.min(65536,file.size),starts=[0,Math.max(0,Math.floor(file.size/2-n/2)),Math.max(0,file.size-n)],sample=new Blob(starts.map(start=>file.slice(start,start+n)));const digest=await crypto.subtle.digest('SHA-256',await sample.arrayBuffer());fingerprint+='|'+[...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('');}const keyDigest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(fingerprint));return [...new Uint8Array(keyDigest)].map(x=>x.toString(16).padStart(2,'0')).join('');}catch{return [...new TextEncoder().encode(fingerprint)].reduce((hash,byte)=>((hash*33)^byte)>>>0,5381).toString(16).padStart(8,'0').repeat(8);}}
function applyTranscriptResult(data,rangeStart,rangeEnd,{cached=false,elapsed=0,cacheKey=''}={}){
  state.transcriptSegments=(Array.isArray(data.turns)?data.turns:[]).filter(turn=>!turn.intersectsSearch);state.transcriptSearchRange={start:rangeStart,end:rangeEnd};state.transcriptCacheKey=cacheKey;state.hostSpeakerId='';if(data.audioAssetId){state.audioAssetId=data.audioAssetId;state.audioAssetKey=audioFileKey(state.file);}
  if(!state.transcriptSegments.length)throw new Error('No speech turns were found in the selected search area. Expand the range or check the audio.');
  const speakers=[...new Set(state.transcriptSegments.map(turn=>turn.speakerId).filter(Boolean))];$('#host-speaker-select').innerHTML='<option value="">Choose your detected speaker</option>'+speakers.map((speaker,index)=>`<option value="${esc(speaker)}">Speaker ${index+1} · ${esc(speaker)}</option>`).join('');const remembered=speakers.includes(data.hostSpeakerId)?data.hostSpeakerId:'';$('#host-speaker-select').value=remembered;state.hostSpeakerId=remembered;$('#host-speaker-row').classList.remove('hidden');
  $('#context-text').value=state.transcriptSegments.map(turn=>`[${secondsLabel(turn.startSeconds)}–${secondsLabel(turn.endSeconds)}] ${esc(turn.speakerId)}${turn.uncertain?' · uncertain':''}${turn.overlap?' · overlap':''}: ${turn.text}`).join('\n');
  $('#transcript-status').textContent=`${cached?'Cached local STT':'Local STT'} · ${state.transcriptSegments.length} timestamped speaker turns in the selected search area${data.notice?` · ${data.notice}`:''}`;state.sampleContext=false;
  setScanProgress(cached?'Using saved transcript':'Transcript ready',`${state.transcriptSegments.length} speaker turns found${cached?' in this browser’s local cache.':''}${remembered?' · remembered host speaker restored.':' · choose which speaker is you.'}`,{percent:100,elapsed,done:true});return state.transcriptSegments;
}
function setScanProgress(title,detail,{percent=null,elapsed=0,done=false,visible=true}={}){
  const panel=$('#scan-progress'),fill=$('#scan-progress-fill');panel.classList.toggle('hidden',!visible);$('#scan-progress-title').textContent=title;$('#scan-progress-detail').textContent=detail;$('#scan-progress-elapsed').textContent=secondsLabel(elapsed);fill.classList.toggle('indeterminate',percent===null&&!done);fill.classList.toggle('done',done);fill.style.width=percent===null?(done?'100%':'34%'):`${Math.max(0,Math.min(100,percent))}%`;
}
function uploadEpisodeForScan(file,url,onProgress){return new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('POST',url);xhr.setRequestHeader('Content-Type','application/octet-stream');xhr.setRequestHeader('X-Filename',encodeURIComponent(file.name));xhr.upload.onprogress=event=>{if(event.lengthComputable)onProgress(event.loaded/event.total*100);};xhr.onerror=()=>reject(new Error('Could not reach Airwave while uploading the episode.'));xhr.onload=()=>{let data={};try{data=JSON.parse(xhr.responseText||'{}');}catch{}if(xhr.status<200||xhr.status>=300)reject(new Error(data.error||'Local STT could not accept the episode.'));else resolve(data);};xhr.send(file);});}
function audioFileKey(file){return file?`${file.name}|${file.size}|${file.lastModified}`:'';}
async function ensureAudioAsset(){
  const file=state.file;if(!file)throw new Error('Choose a podcast episode first.');const key=audioFileKey(file);
  if(state.audioAssetId&&state.audioAssetKey===key){const check=await fetch(`/api/audio-assets/${state.audioAssetId}`,{cache:'no-store'}).catch(()=>null);if(check?.ok){const data=await check.json();if(data.ready)return state.audioAssetId;}state.audioAssetId='';state.audioAssetKey='';}
  const createdResponse=await fetch('/api/audio-assets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:file.name,duration:state.duration})}),created=await createdResponse.json();if(!createdResponse.ok)throw new Error(created.error||'Could not prepare the local episode for preview.');
  state.audioAssetId=created.audioAssetId;state.audioAssetKey=key;
  try{const ready=await uploadStreamEpisode(file,`/api/audio-assets/${state.audioAssetId}/audio`,percent=>{const detail=$('#render-stage-detail');if(detail)detail.textContent=`Uploading the episode once for local preview and playback · ${Math.round(percent)}%.`;});if(!ready.ok)throw new Error(ready.error||'Could not upload the episode for preview.');return state.audioAssetId;}
  catch(error){fetch(`/api/audio-assets/${state.audioAssetId}`,{method:'DELETE'}).catch(()=>{});state.audioAssetId='';state.audioAssetKey='';throw error;}
}
async function transcribeSearchArea() {
  const button=$('#transcribe-button');setBusy(button,true,'Transcribing…');
  const scanStarted=Date.now(),episodeRevision=state.episodeRevision,file=state.file;
  const stillCurrent=()=>state.episodeRevision===episodeRevision;
  try {
    state.transcriptSegments=[];state.transcriptSearchRange=null;state.hostSpeakerId='';$('#host-speaker-row').classList.add('hidden');
    if(state.sampleEpisode){
      if(!state.audioBuffer)throw new Error('Load the sample episode before transcribing it.');
      const start=Math.max(state.slotStart,state.insertAt-8),end=Math.min(state.slotEnd,state.insertAt+8);
      state.transcriptSegments=[{id:'sample-host-turn',startSeconds:start,endSeconds:end,speakerId:'Sample host',text:$('#context-text').value.trim(),uncertain:false,overlap:false}];
      state.hostSpeakerId='';state.transcriptSearchRange={start:state.slotStart,end:state.slotEnd};
      $('#host-speaker-select').innerHTML='<option value="Sample host">Sample host · demo context</option>';$('#host-speaker-select').value='';$('#host-speaker-row').classList.remove('hidden');state.hostSpeakerId='Sample host';
      $('#transcript-status').textContent='Sample context only · generated sample audio contains no real speech';setScanProgress('Sample context ready','This audio bed contains no actual speech.',{percent:100,done:true,visible:true});return state.transcriptSegments;
    }
    if(!file)throw new Error('Upload the original episode file before using local diarization.');
    const rangeStart=state.slotStart,rangeEnd=state.slotEnd,cacheKey=await makeTranscriptCacheKey(file,rangeStart,rangeEnd),legacyPrefix=`${file.name}|${file.size}|${file.lastModified}|${rangeStart.toFixed(3)}|${rangeEnd.toFixed(3)}`;if(!stillCurrent())throw new Error('The episode or search area changed during the scan. Start it again.');
    const cached=await getCachedTranscript(cacheKey,legacyPrefix);if(!stillCurrent())throw new Error('The episode or search area changed during the scan. Start it again.');if(cached&&Date.now()-cached.savedAt<30*24*60*60*1000){await cacheTranscript(cacheKey,cached);return applyTranscriptResult(cached,rangeStart,rangeEnd,{cached:true,elapsed:(Date.now()-scanStarted)/1000,cacheKey});}
    if(!state.localSttReady)throw new Error('Start your local STT project at 127.0.0.1:5095, then try again.');
    $('#transcript-status').textContent='Sending the selected search range to local speaker diarization…';
    const scanQuery=new URLSearchParams({offset:String(rangeStart),start:String(rangeStart),end:String(rangeEnd)});
    setScanProgress('Uploading episode · 0%',`Sending ${file.name} to the local Airwave server, which will crop only ${secondsLabel(rangeStart)}–${secondsLabel(rangeEnd)} before local diarization.`,{percent:0});
    const job=await uploadEpisodeForScan(file,`/api/local-transcribe?${scanQuery}`,percent=>{if(stillCurrent())setScanProgress(percent>=100?'Cropping selected range locally':'Uploading episode · '+Math.round(percent)+'%',percent>=100?'The file is uploaded. FFmpeg is extracting the selected passage before it is sent to local diarization.':`The selected ${secondsLabel(rangeStart)}–${secondsLabel(rangeEnd)} range will be cropped locally before it reaches the STT project.`,{percent:percent>=100?null:percent,elapsed:(Date.now()-scanStarted)/1000});});if(!stillCurrent())throw new Error('The episode or search area changed during the scan. Start it again.');
    if(job.audioAssetId){state.audioAssetId=job.audioAssetId;state.audioAssetKey=audioFileKey(file);}
    setScanProgress('Episode uploaded · queued',job.message||'Waiting for local speaker diarization to start.',{percent:null,elapsed:(Date.now()-scanStarted)/1000});
    const deadline=Date.now()+30*60*1000;let data=null;
    while(Date.now()<deadline){
      const elapsed=(Date.now()-scanStarted)/1000;if(!stillCurrent())throw new Error('The episode or search area changed during the scan. Start it again.');const pollUrl=`/api/local-transcribe/${encodeURIComponent(job.jobId)}?${scanQuery}`;const response=await fetch(pollUrl,{cache:'no-store'});const current=await response.json();if(!response.ok)throw new Error(current.error||'Local speaker diarization failed.');if(!stillCurrent())throw new Error('The episode or search area changed during the scan. Start it again.');
      if(current.status==='completed'){data={...current,audioAssetId:job.audioAssetId||''};break;}
      if(current.status==='failed')throw new Error(current.error||'Local speaker diarization failed.');
      const queued=current.status==='queued';setScanProgress(queued?'Queued for local analysis':'Analyzing selected range locally',`${current.message||'Detecting speakers and aligning word timestamps.'} Range: ${secondsLabel(rangeStart)}–${secondsLabel(rangeEnd)} (${secondsLabel(rangeEnd-rangeStart)}). This setup runs two diarization passes plus word recognition and optional overlap recovery. STT percentage isn’t available; elapsed ${secondsLabel(elapsed)}.`,{percent:null,elapsed});
      await new Promise(resolve=>setTimeout(resolve,1500));
    }
    if(!data)throw new Error('Local diarization is still running after 30 minutes. Check the local STT project and try again.');
    await cacheTranscript(cacheKey,data);if(!stillCurrent())throw new Error('The episode or search area changed during the scan. Start it again.');return applyTranscriptResult(data,rangeStart,rangeEnd,{elapsed:(Date.now()-scanStarted)/1000,cacheKey});
  } catch(error) {if(stillCurrent()){$('#transcript-status').textContent='Local diarization could not transcribe this episode';setScanProgress('Scan stopped',error.message,{percent:null,elapsed:(Date.now()-scanStarted)/1000});}throw error; }
  finally {setBusy(button,false);$('#transcribe-label').textContent='Transcribe & label speakers';}
}
async function previewEpisode() {
  if(state.sampleEpisode){
    if(!state.audioBuffer)return notify('Load the sample episode first.');const ctx=state.audioContext||new(window.AudioContext||window.webkitAudioContext)();state.audioContext=ctx;
    const src=ctx.createBufferSource();src.buffer=state.audioBuffer;const gain=ctx.createGain();gain.gain.value=.7;src.connect(gain);gain.connect(ctx.destination);src.start(0,Math.max(0,state.insertAt-8),Math.min(16,state.duration-Math.max(0,state.insertAt-8)));return;
  }
  const player=$('#episode-audio');if(!state.file)return notify('Upload an episode to preview it.');const previewStart=Math.max(0,state.insertAt-8),previewEnd=Math.min(state.duration,state.insertAt+8);player.currentTime=previewStart;player.play().catch(()=>notify('Press Play to preview this moment.'));const stop=()=>{if(player.currentTime>=previewEnd){player.pause();player.removeEventListener('timeupdate',stop);}};player.addEventListener('timeupdate',stop);
}

function localDraft(modelName, modelIndex, ads, context, interests, slotSeconds) {
  const combined=tokenSet(`${context} ${interests.join(' ')}`);
  const scored=ads.map(ad=>{const tags=tokenSet(`${(ad.tags||[]).join(' ')} ${ad.product} ${ad.audience}`);let overlap=0;for(const t of tags)if(combined.has(t))overlap+=1;const points=Math.min(42,overlap*10);const productWords=[...tokenSet(ad.product)];const contextual=productWords.filter(x=>combined.has(x)).length;const score=Math.min(95,52+points+Math.min(12,contextual*4)+(ad.facts?.length?2:0));return{ad,score,overlap};}).sort((a,b)=>b.score-a.score);
  const chosen=scored[Math.min(modelIndex,scored.length-1)]||scored[0];if(!chosen)return null;
  const name=chosen.ad.name, interest=interests[modelIndex%Math.max(1,interests.length)]||'little rituals';
  const first=chosen.ad.facts?.[0]||chosen.ad.product;const topic=deriveTopic(context);const script=`Speaking of ${topic}, this brief pause is sponsored by ${name}. ${first}. ${chosen.ad.offer}. If that sounds like your kind of thing, look for ${name}.`;
  const slotFit=Math.max(45,Math.min(96,100-Math.max(0,script.split(/\s+/).length/2.45-slotSeconds)*8));
  return {modelId:modelName,label:modelName,advertiserId:chosen.ad.id,advertiserName:name,script,score:Math.round(chosen.score*.34+88*.28+78*.18+slotFit*.2),confidence:Math.min(.94,.58+chosen.overlap*.07),critique:modelIndex===0?'Strongest connection to the conversation and listener. Keep the offer specific and the read relaxed.':'This advertiser is a plausible fit; the mention should feel like a natural aside, not a hard pivot.',breakdown:{relevance:chosen.score,safety:88,naturalness:78,slotFit:Math.round(slotFit)},source:'demo'};
}
async function findMatch() {
  const reviewRevision=state.episodeRevision;Object.values(state.preparedSessions).forEach(id=>fetch(`/api/stream-sessions/${id}`,{method:'DELETE'}).catch(()=>{}));state.preparedSessions={};
  state.momentChosen=false;updateSlotVisual();
  const button=$('#find-match');setBusy(button,true,'Finding the fit…');$('#results-panel').classList.remove('hidden');$('#results-panel').scrollIntoView({behavior:'smooth',block:'nearest'});$('#agent-grid').innerHTML='<div class="agent-card"><div class="agent-title"><span><i class="agent-avatar">A</i> Comparing the moment…</span><span class="agent-score">ANALYZING</span></div><p>Reading the conversation and listener interests against the advertiser catalog.</p></div><div class="agent-card"><div class="agent-title"><span><i class="agent-avatar">B</i> Checking for a natural fit…</span><span class="agent-score">ANALYZING</span></div><p>Considering which offer belongs in this particular break.</p></div>';
  $('#winner-card').classList.add('hidden');$('#no-match-card').classList.add('hidden');$('#render-controls').classList.add('hidden');$('#render-result').classList.add('hidden');
  let payload={advertisers:state.advertisers.map(({id,name,product,audience,offer,facts,avoid,tags})=>({id,name,product,audience,offer,facts,avoid,tags})),topic:deriveTopic($('#context-text').value),context:$('#context-text').value,transcriptSegments:state.transcriptSegments,interests:state.interests,locale:$('#locale-select').value,slotSeconds:28,searchStart:state.slotStart,searchEnd:state.slotEnd,insertAt:state.insertAt};
  try {
    if(state.file&&!state.sampleEpisode&&(state.transcriptSearchRange?.start!==state.slotStart||state.transcriptSearchRange?.end!==state.slotEnd)){
      $('#agent-grid').innerHTML='<div class="agent-card"><div class="agent-title"><span><i class="agent-avatar">STT</i> Scanning episode locally…</span><span class="agent-score">LOCAL</span></div><p>Airwave uploads this file to the local server, crops the selected range there, then sends only that crop to local diarization.</p></div>';
      await transcribeSearchArea();
    }
    if(state.episodeRevision!==reviewRevision)throw new Error('The episode or search area changed. Run the ad review again.');
    if(state.sampleEpisode&&!state.transcriptSegments.length)await transcribeSearchArea();
    if(state.file&&!state.sampleEpisode&&!state.hostSpeakerId)throw new Error('Choose which detected speaker is you, then find the ad fit.');
    const transcriptSegments=state.transcriptSegments.length?state.transcriptSegments:[{id:'manual-context',startSeconds:state.slotStart,endSeconds:state.slotEnd,speakerId:'Sample host',text:$('#context-text').value.trim(),uncertain:false,overlap:false}].filter(s=>s.text);
    payload={...payload,topic:deriveTopic($('#context-text').value),context:$('#context-text').value,transcriptSegments,selectedHostSpeakerId:state.hostSpeakerId||'Sample host'};
    if(state.file&&!state.sampleEpisode){$('#agent-grid').innerHTML='<div class="agent-card"><div class="agent-title"><span><i class="agent-avatar">LOCAL</i> Scoring audio boundaries…</span><span class="agent-score">LOCAL AUDIO</span></div><p>Checking silence and turn timing locally, then sending only the strongest moments for model review.</p></div>';payload.audioAssetId=await ensureAudioAsset();}
    let results;
    if(state.liveModels){const response=await loggedJsonRequest('ads',payload);const data=await response.json();if(state.episodeRevision!==reviewRevision)throw new Error('The episode or search area changed. Run the ad review again.');if(!response.ok)throw new Error(data.error||'Models are unavailable');results=data.candidates;state.debateWinnerIndex=Number(data.winnerIndex);state.audioEvidenceMode=data.audioEvidence||'turn_timing_only';$('#agent-grid').dataset.debateSummary=data.summary||'';$('#result-mode').textContent='TWO-ROUND MODEL DEBATE';}
    else {throw new Error('No configured model keys');}
    state.candidates=results.map((c,i)=>({...c, source:'live',finalist:i===state.debateWinnerIndex}));
  } catch(error) {
    if(state.episodeRevision!==reviewRevision){setBusy(button,false);return;}
    if(state.file&&!state.hostSpeakerId&&state.transcriptSegments.length){state.winner=null;$('#agent-grid').innerHTML=`<div class="no-match-card"><b>Choose the host speaker to continue.</b> The transcript is ready; select your detected speaker above. This transcript is cached, so choosing the speaker won’t repeat the local scan.</div>`;$('#winner-card').classList.add('hidden');$('#render-controls').classList.add('hidden');setBusy(button,false);return;}
    if(state.file&&!state.sampleEpisode&&(!state.transcriptSegments.length||!state.hostSpeakerId||!state.localSttReady||state.transcriptSearchRange?.start!==state.slotStart||state.transcriptSearchRange?.end!==state.slotEnd)){
      state.candidates=[];state.winner=null;$('#agent-grid').innerHTML=`<div class="no-match-card"><b>Search-area scan stopped.</b> ${esc(error.message)} No moment was selected because the whole range was not transcribed.</div>`;$('#winner-card').classList.add('hidden');$('#render-controls').classList.add('hidden');notify(error.message||'Could not scan the selected search area.');setBusy(button,false);return;
    }
    if(state.file&&!state.sampleEpisode){
      state.candidates=[];state.winner=null;state.debateWinnerIndex=null;$('#result-mode').textContent='MODEL REVIEW FAILED';$('#agent-grid').innerHTML=`<div class="no-match-card"><b>No live ad was selected.</b> The model review failed validation: ${esc(error.message)} No simulated result was substituted. Retry the model review after checking the selected host speaker and transcript turns.</div>`;$('#winner-card').classList.add('hidden');$('#render-controls').classList.add('hidden');notify(`Live model review failed: ${error.message}`);setBusy(button,false);return;
    }
    console.info('Using local demo evaluation:',error.message);
    const a=localDraft('Perspective A',0,state.advertisers,payload.context,state.interests,payload.slotSeconds),b=localDraft('Perspective B',1,state.advertisers,payload.context,state.interests,payload.slotSeconds);
    state.candidates=[a,b].filter(Boolean);state.debateWinnerIndex=null;$('#result-mode').textContent='DEMO REVIEW';
    if(state.liveModels)notify(`Live model review unavailable; showing demo evaluation. ${error.message}`);
  }
  chooseWinner();
  state.momentChosen=Boolean(state.winner&&Number.isFinite(Number(state.winner.momentSeconds))&&(state.winner.source==='live'||state.winner.source==='demo'));
  if(state.momentChosen){
    const modelTime=Math.max(state.slotStart,Math.min(state.slotEnd,Number(state.winner.momentSeconds)));
    state.insertAt=modelTime;
    updateSlotVisual();
  }else if(state.winner?.source==='demo'&&state.transcriptSegments.length){state.insertAt=Number(state.transcriptSegments.find(t=>t.id==='sample-host-turn')?.endSeconds||state.insertAt);state.momentChosen=true;updateSlotVisual();}
  payload={topic:deriveTopic($('#context-text').value)};state.resultPayload=payload;
  renderResults(payload);
  setBusy(button,false);
}
function deriveTopic(context) {
  const stated=$('#episode-topic')?.value.trim();if(stated)return stated;
  if(!context.trim())return 'Topic not provided';const normalized=context.toLowerCase();
  const groups=[['morning rituals',['morning','start the day','phone','calm','habit']],['wellbeing',['wellness','calm','breath','mindful','stress']],['food and drink',['coffee','tea','recipe','cook','food']],['outdoors',['hike','trail','outside','nature','walk']],['creative routines',['journal','write','book','creative','reflect']]];
  const scored=groups.map(([label,words])=>({label,score:words.reduce((n,w)=>n+(normalized.includes(w)?1:0),0)})).sort((a,b)=>b.score-a.score);
  return scored[0]?.score?scored[0].label:'the conversation in this episode';
}
function chooseWinner() {
  if(Number.isInteger(state.debateWinnerIndex)&&state.debateWinnerIndex>=0&&state.debateWinnerIndex<state.candidates.length){state.winner=state.candidates[state.debateWinnerIndex];state.debateWinnerIndex=null;return;}
  state.debateWinnerIndex=null;
  const pass=state.candidates.filter(c=>(Number(c.breakdown?.relevance??c.score??0)>=55)&&(Number(c.breakdown?.safety??c.safety??80)>=60)&&(Number(c.breakdown?.slotFit??c.slotFit??c.score)>=48));
  state.winner=pass.sort((a,b)=>Number(b.score||0)-Number(a.score||0))[0]||null;
}
function renderResults(payload) {
  const grid=$('#agent-grid');
  grid.innerHTML=state.candidates.map((c,i)=>{
    const ad=state.advertisers.find(a=>a.id===c.advertiserId)||state.advertisers.find(a=>a.name===c.advertiserName)||{};
    const chosen=state.winner===c;const score=Math.round(Number(c.score)||0), confidence=Math.round(Number(c.confidence||c.confidenceEstimate||.72)*100);
    const breakdown=c.breakdown||{};
    const canPreview=state.file&&!state.sampleEpisode&&c.source==='live'&&c.safeToPreview===true&&c.anchorTurnId&&Number.isFinite(Number(c.momentSeconds));
    const acoustic=c.audioEvidence==='ffmpeg_silence'?'audio silence measured locally':'transcript timing only';
    const localEvidence=Number(c.localMomentScore)>0?`<div class="local-moment-evidence">Local spot rank ${Math.round(c.localMomentScore)}/100 · ${esc(acoustic)}${Number.isFinite(Number(c.audioPauseSeconds))?` · ${Number(c.audioPauseSeconds).toFixed(2)}s pause`:''}</div>`:'';
    const previewButton=canPreview?`<button class="outline-button preview-candidate" data-candidate-index="${i}">${state.preparedSessions[candidateSessionKey(c)]?'Preview again':'Preview this placement'}</button>`:'';
    return `<article class="agent-card ${chosen?'winner':''}"><div class="agent-title"><span><i class="agent-avatar">${i%2?'B':'A'}</i> ${esc(c.label||c.modelId||`Perspective ${i?'B':'A'}`)}${c.finalist?' · FINAL PICK':''}</span><span class="agent-score">${score} FIT</span></div><p>${esc(c.critique||'This perspective weighed the moment, the listener profile, and the advertiser facts.')}</p><div class="agent-brand">${esc(ad.name||c.advertiserName||'Advertiser')} · completed host turn at ${Number.isFinite(Number(c.momentSeconds))?secondsLabel(c.momentSeconds):'not selected'}</div>${c.momentEvidence?`<div class="agent-moment">“${esc(c.momentEvidence)}”</div>`:''}<div class="agent-script">“${esc(c.script||'')}”</div><div class="agent-breakdown"><span>Relevance ${Math.round(breakdown.relevance??score)}%</span><span>Safety ${Math.round(breakdown.safety??84)}%</span><span>Slot fit ${Math.round(breakdown.slotFit??75)}%</span><span>Est. confidence ${confidence}%</span></div>${localEvidence}${previewButton}</article>`;
  }).join('');
  $$('.preview-candidate',grid).forEach(button=>button.addEventListener('click',()=>previewAdCandidate(state.candidates[Number(button.dataset.candidateIndex)])));
  const winner=state.winner;
  if(!winner){$('#winner-card').classList.add('hidden');$('#render-controls').classList.add('hidden');$('#no-match-card').classList.remove('hidden');$('#no-match-card').innerHTML=`<b>Let this one breathe.</b> Neither option scored well enough on safety and slot fit. There’s no ad inserted, so the original episode stays intact. <button class="inline-link" id="retry-match">Review again</button>`;$('#retry-match').addEventListener('click',findMatch);return;}
  const advertiser=state.advertisers.find(a=>a.id===winner.advertiserId)||state.advertisers.find(a=>a.name===winner.advertiserName)||{};
  const hasModelMoment=Number.isFinite(Number(winner.momentSeconds))&&(winner.source==='live'||winner.source==='demo');
  $('#no-match-card').classList.add('hidden');$('#winner-card').classList.remove('hidden');$('#winner-card').innerHTML=`<div class="winner-top"><b>✳ ${esc(advertiser.name||winner.advertiserName)} ${hasModelMoment?`fits this moment · ${secondsLabel(winner.momentSeconds)}`:'is the demo fit'}</b><span>${Math.round(winner.score)} / 100 · ESTIMATED FIT</span></div>${winner.momentEvidence?`<p class="chosen-evidence">Chosen completed host turn: “${esc(winner.momentEvidence)}”</p>`:''}<p>Selected for its connection to <b>${esc(payload.topic)}</b>, the nearby conversation, and ${esc($('#listener-name').value||'the listener')}’s interest in ${esc(state.interests.slice(0,2).join(' and ')||'the topic')}. ${esc(winner.critique||'')}</p>`;
  const readyToPlay=Boolean(state.preparedSessions[candidateSessionKey(winner)]);$('#render-controls').classList.toggle('hidden',!readyToPlay);$('#render-ad').textContent='Start full episode with this ad →';
}

function candidateSessionKey(candidate){return`${candidate?.anchorTurnId||''}|${candidate?.advertiserId||''}|${candidate?.script||''}`;}
function showCandidatePreview(candidate,sessionId){const advertiser=state.advertisers.find(item=>item.id===candidate.advertiserId)||{},result=$('#render-result');result.classList.remove('hidden');result.innerHTML=`<div class="render-ready"><h3>${esc(advertiser.name||candidate.advertiserName)} · insertion preview</h3><p>Hear 10 seconds before the host turn, the ad at ${secondsLabel(state.insertAt)}, then 10 seconds of the original episode after it. Preview playback does not consume the full stream session.</p><audio controls preload="none" src="/api/stream-sessions/${sessionId}?mode=preview&amp;before=10&amp;after=10"></audio><small>Listen for the host thought, sponsor handoff, and return to the conversation.</small></div>`;const player=$('audio',result);player.addEventListener('error',()=>notify('The insertion preview could not play. Check FFmpeg and the local server.'),{once:true});$('#render-controls').classList.remove('hidden');$('#render-ad').textContent='Start full episode with this ad →';}
async function previewAdCandidate(candidate){
  if(!candidate||!state.file||state.sampleEpisode)return notify('Upload and transcribe a real episode before previewing an insertion.');
  if(!candidate.anchorTurnId||!Number.isFinite(Number(candidate.momentSeconds)))return notify('This proposal has no verified host-turn boundary.');
  const revision=state.episodeRevision,key=candidateSessionKey(candidate),result=$('#render-result');state.winner=candidate;state.debateWinnerIndex=null;state.insertAt=Number(candidate.momentSeconds);state.momentChosen=true;updateSlotVisual();if(state.resultPayload)renderResults(state.resultPayload);
  const existingSession=state.preparedSessions[key];if(existingSession){showCandidatePreview(candidate,existingSession);return;}
  $$('.preview-candidate').forEach(button=>button.disabled=true);result.classList.remove('hidden');result.innerHTML='<div class="audio-render-progress"><div class="scan-progress-top"><b id="render-stage-title">Preparing this spot</b><span id="render-stage-time">00:00</span></div><div class="scan-progress-track"><div class="scan-progress-fill indeterminate"></div></div><small id="render-stage-detail">Preparing the sponsor voice and checking the exact turn end.</small></div>';
  const startedAt=Date.now(),timer=setInterval(()=>{const time=$('#render-stage-time');if(time)time.textContent=secondsLabel((Date.now()-startedAt)/1000);},1000),stage=(title,detail)=>{const heading=$('#render-stage-title'),copy=$('#render-stage-detail');if(heading)heading.textContent=title;if(copy)copy.textContent=detail;},ensureCurrent=()=>{if(state.episodeRevision!==revision)throw new Error('The episode or search area changed while preparing this preview.');};let sessionId='';
  try{
    const statusResponse=await fetch('/api/status'),status=await statusResponse.json();if(!status.ffmpegReady)throw new Error('FFmpeg is not available. Set FFMPEG_PATH in .env and restart Airwave.');if(!state.gnaniReady)throw new Error('Gnani TTS is not configured on the Airwave server.');
    stage('Preparing local episode','Reusing the local upload from scanning, or uploading it once for previews.');const audioAssetId=await ensureAudioAsset();ensureCurrent();
    stage('Generating sponsor audio','Creating the ad read for this candidate.');const voice=await requestVoice(candidate.script,$('#locale-select').value,state.useHostVoice);ensureCurrent();const adAudioBase64=await blobBase64(voice);
    stage('Checking the insertion point','The server verifies this candidate against the selected host turn and local boundary ranking.');
    const create=await fetch('/api/stream-sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:state.file.name,duration:state.duration,audioAssetId,insertAt:state.insertAt,searchStart:state.slotStart,searchEnd:state.slotEnd,anchorTurnId:candidate.anchorTurnId,hostSpeakerId:state.hostSpeakerId,transcriptSegments:state.transcriptSegments,adAudioBase64})});const session=await create.json();if(!create.ok)throw new Error(session.error||'Could not prepare this insertion preview.');sessionId=session.sessionId;ensureCurrent();state.preparedSessions[key]=sessionId;
    if(state.resultPayload)renderResults(state.resultPayload);showCandidatePreview(candidate,sessionId);
  }catch(error){if(sessionId){delete state.preparedSessions[key];fetch(`/api/stream-sessions/${sessionId}`,{method:'DELETE'}).catch(()=>{});}result.innerHTML=`<div class="no-match-card"><b>Preview isn’t ready.</b> ${esc(error.message)}</div>`;$('#render-controls').classList.add('hidden');notify(error.message||'Could not prepare the preview.');}
  finally{clearInterval(timer);$$('.preview-candidate').forEach(button=>button.disabled=false);}
}
async function streamEpisodeWithAd(){
  const candidate=state.winner,key=candidateSessionKey(candidate),sessionId=state.preparedSessions[key];if(!sessionId)return notify('Preview this candidate before starting the full episode.');
  const button=$('#render-ad'),result=$('#render-result');setBusy(button,true,'Starting full stream…');const oldPlayer=$('audio',result);oldPlayer?.pause();result.classList.remove('hidden');
  result.innerHTML=`<div class="render-ready"><h3>Continuous episode stream ready</h3><p>The original podcast plays through, with the clearly disclosed ad after the completed host turn at ${secondsLabel(state.insertAt)}. The original episode is not changed or saved over.</p><audio controls autoplay preload="none" src="/api/stream-sessions/${sessionId}"></audio><small>If playback doesn’t start automatically, press Play once.</small></div>`;
  const player=$('audio',result);player.addEventListener('error',()=>notify('The stream could not play. Check FFmpeg and the local server.'),{once:true});const playback=player.play();if(playback)playback.then(()=>notify('Continuous podcast stream started.')).catch(()=>notify('Stream is ready. Press Play in the player to begin.'));delete state.preparedSessions[key];setBusy(button,false);
}
function uploadStreamEpisode(file,url,onProgress){return new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('PUT',url);xhr.setRequestHeader('Content-Type',file.type||'application/octet-stream');xhr.setRequestHeader('X-Filename',encodeURIComponent(file.name));xhr.upload.onprogress=event=>{if(event.lengthComputable)onProgress(event.loaded/event.total*100);};xhr.onerror=()=>reject(new Error('Could not reach the local stream server while uploading the episode.'));xhr.onload=()=>{let data={};try{data=JSON.parse(xhr.responseText||'{}');}catch{}if(xhr.status<200||xhr.status>=300)reject(new Error(data.error||'Could not upload this episode.'));else resolve({...data,ok:true});};xhr.send(file);});}
async function requestVoice(script,language=$('#locale-select').value,useHostVoice=state.useHostVoice) {
  const response=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:script,language,useHostVoice})});
  if(!response.ok){let msg='Gnani TTS could not create the audio.';try{msg=(await response.json()).error||msg;}catch{}throw new Error(msg);}
  return await response.blob();
}
async function decodeBlob(blob) {const ctx=state.audioContext||new(window.AudioContext||window.webkitAudioContext)();state.audioContext=ctx;return await ctx.decodeAudioData(await blob.arrayBuffer());}
init();
