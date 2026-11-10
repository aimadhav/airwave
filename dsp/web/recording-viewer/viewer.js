/* Local-only recording viewer. User-supplied text is never inserted as HTML. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let data;
  try { data = JSON.parse($('recording-data').textContent); }
  catch { $('filename').textContent = 'This is the page template. Process a recording to generate the complete listening room.'; return; }
  const audio = $('audio');
  const buffered = data.web_playback?.buffer_before_seek === true && typeof window.createBufferedPlayback === 'function';
  const playback = buffered ? window.createBufferedPlayback({audio, source:data.audio,
    mimeType:data.web_playback.mime_type, timeoutMs:data.web_playback.load_timeout_seconds*1000,
    onStatus:message=>{$('now-speaking').textContent=message;}}) : {ensure:()=>Promise.resolve()};
  let playbackRequest = 0;
  const readOnly = data.read_only === true;
  if(readOnly){
    $('new-recording').hidden=true;
    const badge=document.querySelector('.private');if(badge)badge.textContent='Read-only shared demo';
    const report=document.createElement('a');report.className='button';report.textContent='Results & overlap examples ↗';report.href='examples.html';
    document.querySelector('.masthead-actions')?.prepend(report);
  }
  const labLink=document.createElement('a');labLink.className='button';labLink.textContent='Overlap lab ↗';
  const recordingId=location.pathname.match(/\/recordings\/([a-f0-9]{32})\//)?.[1]||'current';
  if(!readOnly&&(location.protocol==='http:'||location.protocol==='https:')){labLink.href=`/lab/?recording=${recordingId}`;document.querySelector('.masthead-actions')?.prepend(labLink);}
  else if(!readOnly){try{const app=new URL(data.local_app_url);if(app.protocol==='http:'&&['localhost','127.0.0.1'].includes(app.hostname)){labLink.href=new URL(`/lab/?recording=${recordingId}`,app).href;document.querySelector('.masthead-actions')?.prepend(labLink);}}catch{}}
  const palette = [ ['#7a62a7','#f0ebf7'], ['#b77d53','#f7eee7'], ['#558c7a','#eaf3ef'], ['#6d8faf','#eaf0f6'], ['#b56d86','#f9ecf1'], ['#92904d','#f3f2e6'], ['#9b6caa','#f4ebf7'], ['#648b9d','#eaf1f4'], ['#998677','#f3efeb'] ];
  const state = { model: 'sherpa', speaker: null, overlaps: false, search: '', active: new Set() };
  state.model = data.default_backend && data.variants[data.default_backend] ? data.default_backend : 'sherpa';
  $('model').replaceChildren(...Object.keys(data.variants).map(key=>{const option=document.createElement('option');option.value=key;option.textContent=({sherpa:'Sherpa · comparison',nemo_native:'Nemotron · original transcript',overlap_recovery:'Nemotron + overlap recovery'})[key]||key;return option;}));
  $('model').value=state.model;
  if($('processing-notice')) $('processing-notice').textContent=data.notice;
  if($('recovery-summary')&&data.recovery?.enabled){const r=data.recovery;$('recovery-summary').textContent=`Overlap windows: ${r.accepted||0} accepted · ${r.rejected||0} rejected · ${r.skipped||0} skipped · ${r.failed||0} failed. Acceptance means engineering checks passed, not human verification.`;
    $('recovery-details').hidden=false;for(const row of r.decisions||[]){const item=document.createElement('li');item.textContent=`${row.start.toFixed(2)}–${row.end.toFixed(2)}s: ${row.status} — ${row.reason.replaceAll('_',' ')}`;$('recovery-decisions').append(item);}}
  let turns = [], groups = [], byId = new Map(), speakerIds = [], cardMap = new Map(), bars = [];
  const fmt = seconds => { const n = Math.max(0, Math.floor(seconds)); return `${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`; };
  const precise = seconds => { const ticks=Math.max(0,Math.round(seconds*100)); return `${Math.floor(ticks/6000)}:${String(Math.floor(ticks/100)%60).padStart(2,'0')}.${String(ticks%100).padStart(2,'0')}`; };
  const el = (tag, className, text) => { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; };
  const name = id => id === 'UNKNOWN' ? 'Unassigned voice' : `Speaker ${speakerIds.indexOf(id)+1}`;
  const initials = id => id === 'UNKNOWN' ? '?' : String(speakerIds.indexOf(id)+1).padStart(2,'0');
  const decorate = (node, id) => { const i = speakerIds.indexOf(id); const colors = id === 'UNKNOWN' ? ['#99939f','#f0eef2'] : palette[Math.max(0,i)%palette.length]; node.style.setProperty('--speaker',colors[0]); node.style.setProperty('--speaker-bg',colors[1]); return node; };
  const avatar = id => decorate(el('span','speaker-avatar',initials(id)), id);
  function unionSeconds(rows) { let total=0,right=-1; for(const r of [...rows].sort((a,b)=>a.start-b.start)){ total+=Math.max(0,r.end-Math.max(r.start,right)); right=Math.max(right,r.end); } return total; }
  function selectModel() {
    const variant = data.variants[state.model];
    turns = variant.turns; groups = variant.groups; byId = new Map(turns.map(t=>[t.id,t]));
    $('words-stat').textContent=turns.reduce((sum,t)=>sum+t.word_count,0).toLocaleString();
    speakerIds = [...new Set(turns.map(t=>t.speaker).filter(id=>id!=='UNKNOWN'))].sort();
    if(turns.some(t=>t.speaker==='UNKNOWN')) speakerIds.push('UNKNOWN');
    state.speaker=null; state.active=new Set();
    $('voices-stat').textContent=speakerIds.filter(s=>s!=='UNKNOWN').length;
    $('voice-count').textContent=speakerIds.filter(s=>s!=='UNKNOWN').length;
    $('overlap-count').textContent=groups.filter(g=>g.overlap).length;
    $('first-overlap').disabled=!groups.some(g=>g.overlap);
    renderSpeakers(); renderTranscript(); updatePlayback();
  }
  function renderSpeakers() {
    const container=$('speakers'); container.replaceChildren();
    for(const id of speakerIds) {
      const rows=turns.filter(t=>t.speaker===id), seconds=unionSeconds(rows);
      const button=decorate(el('button','speaker-button'+(state.speaker===id?' selected':'')),id);
      button.setAttribute('aria-pressed',String(state.speaker===id));
      button.append(avatar(id));
      const detail=el('span'); detail.append(el('span','speaker-name',name(id)),el('span','speaker-meta',`${fmt(seconds)} activity · ${rows.length} turns`));
      const meter=el('span','speaker-meter'), fill=el('i'); fill.style.width=`${Math.min(100,seconds/data.duration*100)}%`; meter.append(fill); detail.append(meter); button.append(detail);
      button.addEventListener('click',()=>{state.speaker=state.speaker===id?null:id; renderSpeakers(); renderTranscript();});
      container.append(button);
    }
    $('all-voices').classList.toggle('selected',state.speaker===null);
  }
  function renderTranscript() {
    const root=$('transcript'); root.replaceChildren(); cardMap=new Map(); let visible=0, passages=0;
    const query=state.search.toLocaleLowerCase();
    for(const group of groups) {
      if(state.overlaps&&!group.overlap)continue;
      const rows=group.turn_ids.map(id=>byId.get(id)).filter(t=>(!state.speaker||t.speaker===state.speaker)&&(!query||t.text.toLocaleLowerCase().includes(query)));
      if(!rows.length)continue;
      const passage=el('section','passage'+(group.overlap?' overlap-passage':''));
      passage.dataset.start=String(group.start);
      if(group.overlap){const heading=el('div','overlap-heading');heading.append(el('span','', '↳ OVERLAPPING TURNS'),el('span','',`${fmt(group.start)} – ${fmt(group.end)}`));passage.append(heading);}
      for(const turn of rows) {
        const card=decorate(el('article','turn'),turn.speaker); card.dataset.turnId=turn.id;
        const stamp=el('button','timestamp',precise(turn.start));stamp.setAttribute('aria-label',`Play ${name(turn.speaker)} from ${fmt(turn.start)}`);stamp.append(el('small','',`→ ${precise(turn.end)}`));
        stamp.addEventListener('click',()=>seek(turn.start,true)); card.append(stamp);
        const body=el('div','turn-body'),head=el('div','turn-head');head.append(avatar(turn.speaker),el('span','turn-name',name(turn.speaker)));
        if(turn.uncertain&&turn.text){const badge=el('span','confidence-note','Attribution uncertain');badge.title='Overlapping activity or weak timing agreement makes the speaker assignment uncertain.';head.append(badge);}
        if(!turn.text)head.append(el('span','confidence-note','Words unresolved'));
        if(turn.transcript_source==='separated_asr'){const badge=el('span','confidence-note','Separated · review');badge.title='Independently recognized from a separated audio track and matched by voice embedding. Not human verified.';head.append(badge);}
        const text=turn.text||(turn.overlap?'Voice detected here. Its words could not be isolated from the overlapping speech.':'Voice activity detected; no words were recovered in this interval.');
        body.append(head,el('p','turn-text'+(!turn.text?' unresolved':''),text));card.append(body);passage.append(card);cardMap.set(turn.id,card);visible++;
      }
      root.append(passage);passages++;
    }
    $('result-count').textContent=`${visible} ${visible===1?'turn':'turns'} · ${passages} ${passages===1?'passage':'passages'}${state.search?' matching your search':''}`;
    $('empty').hidden=visible!==0;
    $('all-tab').classList.toggle('active',!state.overlaps);$('overlap-tab').classList.toggle('active',state.overlaps);
    updatePlayback();
  }
  function drawWaveform() {
    const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
    svg.setAttribute('viewBox',`0 0 ${data.waveform.length*3} 36`);svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('aria-hidden','true');
    data.waveform.forEach((level,i)=>{const h=Math.max(3,level*33),bar=document.createElementNS(ns,'rect');bar.setAttribute('x',String(i*3));bar.setAttribute('y',String((36-h)/2));bar.setAttribute('width','1.5');bar.setAttribute('height',String(h));bar.setAttribute('rx','.75');bar.setAttribute('fill','#d8d1e1');svg.append(bar);bars.push(bar);});
    $('waveform').append(svg);$('waveform').setAttribute('aria-valuemax',String(data.duration));
  }
  async function seek(seconds,play=false){
    const request=++playbackRequest;
    try{await playback.ensure();if(request!==playbackRequest)return;
      audio.currentTime=Math.max(0,Math.min(seconds,Number.isFinite(audio.duration)?audio.duration:data.duration));
      if(play)await audio.play();updatePlayback();
    }catch{if(!buffered)showPlaybackError();}
  }
  function showPlaybackError(){ $('now-speaking').textContent='Use the play button to start audio.'; }
  function updatePlayback() {
    const time=audio.currentTime||0,active=turns.filter(t=>t.start<=time&&time<t.end),ids=new Set(active.map(t=>t.id));
    let changed=false;
    for(const [id,card]of cardMap){const on=ids.has(id);card.classList.toggle('active',on);if(on&&!state.active.has(id))changed=true;}
    if(changed&&$('follow').checked){const first=active.map(t=>cardMap.get(t.id)).find(Boolean);first?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'center'});}
    state.active=ids;
    const voices=[...new Set(active.map(t=>name(t.speaker)))];
    $('now-speaking').textContent=voices.length?voices.join(' + '):(audio.paused?'Ready when you are':'Between turns');
    $('current').textContent=fmt(time);$('waveform').setAttribute('aria-valuenow',String(Math.round(time)));
    const position=time/data.duration*bars.length;
    bars.forEach((bar,i)=>bar.setAttribute('fill',i<=position?'#80709e':'#d8d1e1'));
  }
  $('filename').textContent=data.filename;$('duration-stat').textContent=fmt(data.duration);$('words-stat').textContent=data.word_count.toLocaleString();$('total').textContent=fmt(data.duration);if(!buffered)audio.src=data.audio;
  $('model').addEventListener('change',e=>{state.model=e.target.value;selectModel();});
  $('search').addEventListener('input',e=>{state.search=e.target.value;renderTranscript();});
  $('all-voices').addEventListener('click',()=>{state.speaker=null;renderSpeakers();renderTranscript();});
  $('all-tab').addEventListener('click',()=>{state.overlaps=false;renderTranscript();});
  $('overlap-tab').addEventListener('click',()=>{state.overlaps=true;renderTranscript();});
  $('first-overlap').addEventListener('click',()=>{state.search='';$('search').value='';state.speaker=null;state.overlaps=true;renderSpeakers();renderTranscript();const group=groups.find(g=>g.overlap);if(!group)return;const rows=group.turn_ids.map(id=>byId.get(id));let start=group.start;outer:for(let i=0;i<rows.length;i++)for(let j=i+1;j<rows.length;j++)if(rows[i].speaker!==rows[j].speaker&&Math.min(rows[i].end,rows[j].end)>Math.max(rows[i].start,rows[j].start)){start=Math.max(rows[i].start,rows[j].start)+.01;break outer;}seek(start);$('transcript').querySelector('.passage')?.scrollIntoView({block:'center'});});
  $('play').addEventListener('click',()=>audio.paused?seek(audio.currentTime,true):audio.pause());
  $('back').addEventListener('click',()=>seek(audio.currentTime-10));$('forward').addEventListener('click',()=>seek(audio.currentTime+10));
  audio.addEventListener('play',()=>{$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Pause recording');});
  audio.addEventListener('pause',()=>{$('play').textContent='▶';$('play').setAttribute('aria-label','Play recording');});
  audio.addEventListener('timeupdate',updatePlayback);audio.addEventListener('seeked',updatePlayback);
  audio.addEventListener('error',()=>{$('now-speaking').textContent='Audio unavailable. Keep the original audio file beside this page.';});
  $('speed').addEventListener('change',e=>{audio.playbackRate=Number(e.target.value);});
  $('waveform').addEventListener('click',e=>{const rect=e.currentTarget.getBoundingClientRect();seek((e.clientX-rect.left)/rect.width*data.duration);});
  $('waveform').addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();seek(audio.currentTime+(e.key==='ArrowLeft'?-5:5));}});
  $('about-toggle').addEventListener('click',()=>{const open=$('about').hidden;$('about').hidden=!open;$('about-toggle').setAttribute('aria-expanded',String(open));});
  $('export').addEventListener('click',()=>{const lines=[data.filename,'Machine-generated transcript. Speaker identities are unverified.',data.notice,'',...turns.map(t=>`[${fmt(t.start)}–${fmt(t.end)}] ${name(t.speaker)}${t.overlap?' [overlap]':''}${t.uncertain?' [uncertain]':''}: ${t.text||'[Voice detected; words unresolved]'}`)];const url=URL.createObjectURL(new Blob([lines.join('\n\n')],{type:'text/plain;charset=utf-8'}));const link=el('a');link.href=url;link.download='conversation-transcript.txt';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.key==='/'){e.preventDefault();$('search').focus();}});
  drawWaveform();selectModel();
})();
