/* Read-only, fully portable saved results. No fetches, uploads or local APIs. */
(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('demo-data').textContent);
  const node=(tag,text)=>{const element=document.createElement(tag);if(text!==undefined)element.textContent=text;return element;};
  const time=value=>`${Math.floor(value/60)}:${(value%60).toFixed(2).padStart(5,'0')}`;
  function table(headers,rows){
    const result=node('table'),head=node('thead'),body=node('tbody'),heading=node('tr');
    for(const title of headers){const th=node('th',title);th.scope='col';heading.append(th);}
    head.append(heading);result.append(head);
    for(const row of rows){const tr=node('tr');for(const value of row)tr.append(node('td',String(value)));body.append(tr);}
    result.append(body);return result;
  }
  document.getElementById('recording-summary').textContent=`${data.filename} · ${time(data.full_recording.duration_seconds)} · ${data.full_recording.raw_word_tokens.toLocaleString()} raw recognizer word tokens.`;
  document.getElementById('full-results').append(table(['Saved variant','Transcript rows','Estimated voice groups'],data.full_recording.variants.map(variant=>[variant.id,variant.turns,variant.estimated_voice_groups])));
  document.getElementById('gain-results').append(table(['Gain / context','Confirmed overlaps passing checks','Non-overlap controls incorrectly passing'],data.gain_comparison.map(row=>[
    `${row.gain_mode} / ${row.context_seconds}s`,`${row.confirmed_overlaps_accepted}/${row.confirmed_overlaps_tested}`,
    row.negative_controls_tested?`${row.negative_controls_incorrectly_accepted}/${row.negative_controls_tested}`:'not tested'])));
  function player(path,label){
    const audio=node('audio');audio.controls=true;audio.preload='metadata';audio.src=path;audio.setAttribute('aria-label',label);
    audio.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(other!==audio)other.pause();});
    return audio;
  }
  for(const example of data.examples){
    const card=node('article');card.className='report-card';
    card.append(node('h2',`${time(example.start)}–${time(example.end)}`));
    const gate=node('p',`${example.status} · ${example.reason.replaceAll('_',' ')}`);gate.className='gate';card.append(gate);
    const scope=node('p',`Playback: ${time(example.crop_start)}–${time(example.crop_end)}. Scoring core: ${(example.start-example.crop_start).toFixed(2)}–${(example.end-example.crop_start).toFixed(2)} seconds into each player. Text outside that core is not displayed.`);scope.className='scope-note';card.append(scope);
    card.append(node('h3','Original mixture'),player(`examples/${example.id}/mixture.wav`,`Original mixture at ${time(example.start)}`));
    const grid=node('div');grid.className='track-grid';
    for(const track of example.tracks){
      const box=node('section');box.className='track-box';
      const match=track.assigned_speaker?.match(/^SPEAKER_(\d+)$/);
      const label=match?`estimated Speaker ${Number(match[1])+1}`:'unassigned voice';
      box.append(node('h3',`Track ${track.index+1} · ${label}`),node('p',track.text||'[No words recovered in the core]'),player(`examples/${example.id}/track-${track.index+1}.wav`,`Separated track ${track.index+1} at ${time(example.start)}`));
      grid.append(box);
    }
    card.append(grid);document.getElementById('examples').append(card);
  }
})();
