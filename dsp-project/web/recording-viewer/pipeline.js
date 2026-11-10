/* Read-only explanation and saved evidence. No model calls or review writes. */
(() => {
  'use strict';
  /** @param {number} core @param {number} context @param {number} rate
   * @returns {{crop:number,samples:number,leftFraction:number,coreFraction:number}} */
  function chunkGeometry(core, context, rate) {
    if (![core, context, rate].every(Number.isFinite) || core <= 0 || context < 0 || rate <= 0) throw new Error('Invalid time scale');
    const crop = core + 2 * context;
    return {crop, samples:Math.round(crop*rate), leftFraction:context/crop, coreFraction:core/crop};
  }
  /** @param {Array<object>} reports @param {string} status @param {string} reason @returns {Array<object>} */
  function filterReports(reports, status, reason) {
    return reports.filter(row=>(status==='all'||row.status===status)&&(reason==='all'||row.reason===reason));
  }
  /** @param {number} seconds @returns {string} */
  function time(seconds) {
    const ticks=Math.round(seconds*100);
    return `${Math.floor(ticks/6000)}:${String(Math.floor(ticks/100)%60).padStart(2,'0')}.${String(ticks%100).padStart(2,'0')}`;
  }
  window.PipelineNotebook = {chunkGeometry, filterReports, time};

  document.addEventListener('DOMContentLoaded', () => {
    const $=id=>document.getElementById(id);
    let audit;
    try {audit=JSON.parse($('pipeline-data').textContent);} catch { $('run-stats').textContent='Audit evidence is unavailable. Rebuild the saved page before using the numbers.';return; }
    const f=audit.runtime_facts, p=audit.current_recovery_policy, saved=audit.saved_recovery_policy;
    const node=(tag,text,className)=>{const el=document.createElement(tag);if(text!==undefined)el.textContent=String(text);if(className)el.className=className;return el;};
    const row=(table,values)=>{const tr=node('tr');for(const value of values)tr.append(node('td',value));$(table).append(tr);};
    const readable=value=>value.replaceAll('_',' ');
    for(const [value,label] of [[time(audit.duration_seconds),'full recording'],[audit.raw_word_tokens.toLocaleString(),'raw recognizer word tokens'],[audit.reports.length,'overlap recovery candidates'],[audit.counts.accepted||0,'older full-run acceptances—not accuracy']]){
      const item=node('div',undefined,'stat');item.append(node('strong',value),node('span',label));$('run-stats').append(item);
    }
    const nativeSeconds=f.native_chunk_frames*f.native_coarse_frame_seconds;
    const chunks=[
      ['Energy VAD',`${f.vad_frame_ms} ms frame / ${f.vad_hop_ms} ms hop`,'480 / 160 samples at 16 kHz. Diagnostic here, not a hard neural gate.'],
      ['Whisper frequency features',`${f.asr_fft_samples/f.canonical_sample_rate*1000} ms FFT / ${f.asr_hop_samples/f.canonical_sample_rate*1000} ms hop`,`${f.asr_fft_samples} / ${f.asr_hop_samples} samples; ${f.asr_mel_bins} log-mel bands. Installed feature-extractor defaults.`],
      ['Whisper model context',`${f.asr_window_seconds} s maximum window`,'480,000 input samples. Internal decoding advances dynamically; returned segments are not all 30 s.'],
      ['Nemotron model chunks',`${nativeSeconds.toFixed(2)} s / 80 ms context each side`,`${f.native_chunk_frames} × 80 ms coarse frames. v0.2.0 preset source-verified; geometry not independently saved per run.`],
      ['Nemotron prediction grid',`${f.native_output_frame_seconds*1000} ms`,'Activity timing resolution, not guaranteed boundary accuracy. Different from the 80 ms cache grid.'],
      ['Clean voice references',`${p.reference_seconds} s each; ${p.minimum_references}–${p.max_references} references`,`${p.reference_guard_seconds} s guard at either edge of clean spans; require two successful embeddings, keep at most five.`],
      ['Recovery ownership core',`${p.minimum_overlap_seconds}–${p.max_core_seconds} s for eligible two-voice jobs`,'Variable overlap intervals. Cores shorter than the minimum are recorded but skipped. Long overlap is split.'],
      ['Separator input crop',`core + up to ${p.context_seconds} s per side`, `At most ${p.max_core_seconds+2*p.context_seconds} s under the current 1 s policy; third speakers and file edges can shorten it.`],
      ['Embedding crop',`at least ${p.minimum_embedding_seconds} s`,'The separated track plus allowed context; signal gates and word ownership still use the core.'],
      ['SepFormer internal feature chunks',`${f.separator_internal_chunk_steps} encoded steps / ${f.separator_internal_hop_steps} step hop`,'Inside the 8 kHz separator: 1 ms encoder stride, about 250 ms feature chunks with a 125 ms hop. Not the 0.3–6 s ownership cores. Encoder receptive-field edges add about 1 ms.'],
      ['Waveform display',`${(audit.duration_seconds/audit.asr.waveform_bins).toFixed(3)} s per bin`,`${audit.asr.waveform_bins} RMS bins in the saved run. Display only—not an ASR chunk, VAD trace or diarizer output.`],
    ];
    for(const values of chunks)row('chunk-table',values);
    const asrDescriptions={model_repository:'Installed English speech recognizer.',device:'Local inference; no cloud transcription.',compute_type:'Quantized inference to reduce CPU/memory cost.',cpu_threads:'CPU work threads, not simultaneous speakers.',beam_size:'Candidate text search width.',vad_filter:'No built-in VAD removal enabled for this run.',condition_on_previous_text:'No previous-window transcript conditioning.',minimum_word_coverage:'Timing-alignment uncertainty threshold, not ASR confidence.',maximum_no_speech_probability:'Segments above this no-speech score are dropped.',minimum_segment_log_probability:'Segments below this average log probability are dropped.',maximum_compression_ratio:'Segments above this text-repetition/compression ratio are dropped.',language:'English fixed; not automatic language identification.'};
    for(const [key,description] of Object.entries(asrDescriptions))row('asr-settings',[key,String(audit.asr[key]),description]);
    const gates=[
      ['Eligible overlap',`exactly 2 speakers; core ≥ ${p.minimum_overlap_seconds} s`,'A model flag is a candidate, not proof that two people are audible.'],
      ['Attempt budget',`at most ${p.max_windows} chronological eligible attempts`,'Skipped later cores do not mean separation failed. Rejections and failures consume attempts.'],
      ['Clean references',`${p.minimum_references}–${p.max_references} successful × ${p.reference_seconds} s`,'Needs exclusive diarization spans, guarded by 0.15 s at each edge; incorrect diarization can contaminate them.'],
      ['Output contract','2 finite mono tracks; exact crop sample length / 16 kHz','Protects timing/shape, not semantic correctness.'],
      ['Track RMS',`each ≥ ${p.minimum_stream_rms}`,'Reject silence. Loud leakage can still pass.'],
      ['Weak/strong energy ratio',`≥ ${p.minimum_energy_ratio}`,'Reject a negligible secondary stream; not a speech-probability measure.'],
      ['Absolute stream correlation',`≤ ${p.maximum_stream_correlation}`,'Reject nearly duplicated waveforms. Distinct waveforms may still contain the same voice.'],
      ['Least-squares reconstruction',`≥ ${p.minimum_reconstruction_db} dB`,'Checks whether scaled outputs can reconstruct the mix; not SI-SDR or word accuracy.'],
      ['Embedding duration',`crop ≥ ${p.minimum_embedding_seconds} s`,'Short voice samples are unstable. Having enough duration is not enough clean speech.'],
      ['One-to-one voice match',`each cosine ≥ ${p.minimum_cosine}; margin ≥ ${p.minimum_cosine_margin}`,'Both track assignments must independently beat their alternate speaker; similarity is not confidence probability.'],
      ['Core ownership','at least one recognized word on EACH track','Only word midpoints inside the core count. Context speech alone cannot validate it.'],
      ['Word confidence',`minimum ≥ ${p.minimum_word_probability}; mean ≥ ${p.minimum_mean_probability}`,'All owned words need probabilities. Models can still be confidently wrong.'],
      ['Duplicated text',`reject similarity ≥ ${p.maximum_duplicate_similarity} if each has ≥ ${p.minimum_duplicate_words} words`,'Token SequenceMatcher heuristic; short duplicates and nonidentical leakage may pass.'],
      ['Provider failure bounds',`${p.separator_timeout_seconds} s separation timeout; ${p.separator_threads} separator threads`,'Typed failure retains baseline. An operational bound, not an audio chunk length.'],
      ['Current gain policy',`${p.separator_gain_mode}; common peak limit ${p.separator_peak_limit}`,'Shared attenuation only, not independent track gain or denoising. Absent from the older full-run artifact.'],
    ];
    for(const values of gates)row('gate-table',values);
    $('saved-policy').textContent=JSON.stringify(saved,null,2);
    $('current-policy').textContent=JSON.stringify(p,null,2);
    function updateChunk() {
      const core=Number($('core-length').value),context=Number($('context-length').value),g=chunkGeometry(core,context,f.canonical_sample_rate);
      $('core-output').textContent=`${core.toFixed(2)} s`;$('context-output').textContent=`${context.toFixed(1)} s`;
      const left=20+760*g.leftFraction,width=760*g.coreFraction;
      $('chunk-core').setAttribute('x',String(left));$('chunk-core').setAttribute('width',String(width));
      for(const [id,x,label] of [['left-label',20+760*g.leftFraction/2,`context ${context.toFixed(1)} s`],['middle-label',left+width/2,`core ${core.toFixed(2)} s`],['right-label',left+width+760*g.leftFraction/2,`context ${context.toFixed(1)} s`]]){
        $(id).setAttribute('x',String(x));$(id).textContent=label;
      }
      $('chunk-calculation').textContent=`${core.toFixed(2)} + ${context.toFixed(1)} + ${context.toFixed(1)} = ${g.crop.toFixed(2)} s crop · ${g.samples.toLocaleString()} samples at 16 kHz · only the core owns words. ${context>p.context_seconds?'This exceeds the current 1 s default; it is a what-if illustration.':''}`;
    }
    $('core-length').addEventListener('input',updateChunk);$('context-length').addEventListener('input',updateChunk);updateChunk();
    const interpretations={ambiguous_speaker_match:'Track cosine and/or margin failed; not a verified identity.',low_recognition_confidence:'At least one track failed owned-word probability checks.',no_secondary_words_recovered:'At least one track had no owned words inside the core.',overlap_too_short:'Core was shorter than the minimum.',quality_gates_passed_not_human_verified:'All automatic gates passed; no human word verification implied.',unsupported_simultaneous_speaker_count:'The active set was not exactly two speakers.',window_budget_exceeded:'Not attempted because the chronological budget was exhausted.'};
    const attempts=audit.reports.filter(r=>r.status!=='skipped');
    $('audit-summary').append(node('strong',`${attempts.length}/${audit.reports.length} attempted · ${audit.counts.skipped||0} skipped · last attempt ended ${time(audit.last_attempt_end)}`),node('p','Budget exhaustion, unsupported speaker count and short intervals are separate limitations. The later corrected examples were not a rerun of these historical decisions.'));
    for(const [reason,count] of Object.entries(audit.reason_counts).sort((a,b)=>b[1]-a[1])){
      const option=node('option',`${readable(reason)} (${count})`);option.value=reason;$('reason-filter').append(option);
      row('reason-counts',[reason,count,interpretations[reason]||'Inspect the saved reason; do not infer correctness from status alone.']);
    }
    let filtered=[];
    function renderDecision() {
      const report=filtered.find(r=>r.id===$('window-select').value),root=$('window-detail');root.replaceChildren();
      if(!report){root.append(node('p','No saved decisions match these filters. Change a filter to inspect another core.'));return;}
      const duration=report.end-report.start,crop=report.crop_end-report.crop_start;
      root.append(node('span',report.status,`decision ${report.status}`),node('h3',`${report.id} · ${time(report.start)}–${time(report.end)}`),node('p',report.reason),
        node('p',`Estimated active set: ${report.speakers.map(s=>`Speaker ${Number(s.split('_')[1])+1}`).join(' + ')}. Core ${duration.toFixed(3)} s; crop ${crop.toFixed(3)} s (${time(report.crop_start)}–${time(report.crop_end)}).`),
        node('p',`Actual context: ${(report.start-report.crop_start).toFixed(3)} s before; ${(report.crop_end-report.end).toFixed(3)} s after. Crop uses ${Math.round(report.crop_end*16000)-Math.round(report.crop_start*16000)} samples at 16 kHz.`, 'formula'),
        node('p',interpretations[report.reason]||'This decision does not certify correct words.'));
      const metrics=node('div',undefined,'metrics');
      for(const [key,value] of Object.entries(report.metrics)){const item=node('p',readable(key));item.append(node('strong',value.toFixed(5)));metrics.append(item);}
      root.append(metrics,node('p',Object.keys(report.metrics).length?'These are saved, historical metrics. Later checks can be absent because the pipeline stopped at its first failed gate.':'No metric values were saved for this decision. Skipped windows never ran signal/identity/recognition gates.','small'));
    }
    function updateFilter() {
      filtered=filterReports(audit.reports,$('decision-filter').value,$('reason-filter').value);
      $('window-select').replaceChildren(...filtered.map(r=>{const option=node('option',`${time(r.start)} · ${r.id} · ${r.status}`);option.value=r.id;return option;}));
      $('audit-count').textContent=`${filtered.length} matching saved decisions (out of ${audit.reports.length}).`;
      renderDecision();
    }
    $('decision-filter').addEventListener('change',updateFilter);$('reason-filter').addEventListener('change',updateFilter);$('window-select').addEventListener('change',renderDecision);updateFilter();
    $('evidence-summary').append(node('p',`Audit verified ${audit.verified_on}; algorithm/source baseline ${audit.code_baseline}. Source fingerprint:`, 'evidence-line'),node('code',audit.source_sha256));
    const details=node('details');details.append(node('summary','Saved evidence fingerprints (SHA-256)'),node('pre',JSON.stringify(audit.evidence_sha256,null,2)));$('evidence-summary').append(details);
    for(const source of audit.primary_sources){const li=node('li'),link=node('a',source.label);link.href=source.url;link.rel='noopener';li.append(link);$('primary-sources').append(li);}
  });
})();
