/* Uploads stay on the local HTTP app. Saved file:// pages never POST across origins. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  let documentData;
  try{documentData=JSON.parse($('recording-data').textContent);}catch{return;}
  if(documentData.read_only===true)return;
  const dialog=$('upload-dialog'),input=$('recording-file'),submit=$('upload-submit');
  let policy=null,busy=false,job=null,pollTimer=null;
  const error=message=>{$('upload-error').textContent=message;$('upload-error').hidden=false;};
  const clearError=()=>{$('upload-error').hidden=true;$('upload-error').textContent='';};
  function validate(){
    clearError();const file=input.files[0];submit.disabled=busy||!file||!policy;
    $('upload-selected').textContent=file?`${file.name} · ${(file.size/1048576).toFixed(1)} MB`:'No file selected';
    if(!file||!policy)return;
    const suffix='.'+file.name.split('.').pop().toLowerCase();
    if(!policy.extensions.includes(suffix)){error('That file format is not supported. Choose one of the formats shown above.');submit.disabled=true;}
    if(file.size===0||file.size>policy.max_bytes){error(`Choose a non-empty recording under ${Math.round(policy.max_bytes/1048576)} MB.`);submit.disabled=true;}
  }
  async function poll(){
    try{
      const response=await fetch(`/api/jobs/${job}`,{cache:'no-store'}),state=await response.json();
      if(!response.ok)throw new Error(state.error||'The processing job is unavailable. The local app may have restarted.');
      $('upload-status').textContent=state.message;
      if(state.status==='completed'){
        if(!/^\/recordings\/[a-f0-9]{32}\/index\.html$/.test(state.result_url))throw new Error('The app returned an invalid result address.');
        window.location.assign(state.result_url);return;
      }
      if(state.status==='failed')throw new Error(state.message);
      pollTimer=setTimeout(poll,policy.poll_interval_ms);
    }catch(problem){busy=false;submit.disabled=false;input.disabled=false;$('upload-progress-box').hidden=true;error(problem.message||'Could not reach the local app. Check that it is still running.');}
  }
  $('new-recording').addEventListener('click',async()=>{
    dialog.showModal();
    if(!['http:','https:'].includes(location.protocol)){
      $('upload-offline').hidden=false;$('upload-form').hidden=true;
      const url=new URL(documentData.local_app_url);
      if(url.protocol==='http:'&&['127.0.0.1','localhost'].includes(url.hostname))$('upload-open-app').href=url.href;
      return;
    }
    $('upload-form').hidden=false;$('upload-offline').hidden=true;
    if(busy)return;
    try{
      const response=await fetch('/api/upload-config',{cache:'no-store'});policy=await response.json();
      if(!response.ok||!policy.enabled)throw new Error('Uploads are not enabled on this server. Start the updated local listening-room app.');
      input.accept=policy.extensions.join(',');
      $('upload-types').textContent=`${policy.extensions.map(s=>s.slice(1).toUpperCase()).join(' · ')} · up to ${Math.round(policy.max_bytes/1048576)} MB${policy.max_audio_seconds?' · '+Math.round(policy.max_audio_seconds/60)+' minutes':''}`;
      validate();
    }catch(problem){policy=null;submit.disabled=true;error(problem.message||'Could not reach the local upload service.');}
  });
  $('upload-close').addEventListener('click',()=>dialog.close());
  input.addEventListener('change',validate);
  $('upload-form').addEventListener('submit',async event=>{
    event.preventDefault();if(busy||submit.disabled||!policy)return;
    const file=input.files[0];clearError();busy=true;submit.disabled=true;input.disabled=true;
    $('upload-progress-box').hidden=false;$('upload-status').textContent='Uploading to the app on this computer…';
    try{
      const response=await fetch('/api/uploads',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Filename':encodeURIComponent(file.name)},body:file});
      const result=await response.json();if(!response.ok)throw new Error(result.error||'The upload could not be accepted.');
      job=result.id;$('upload-status').textContent=result.message;await poll();
    }catch(problem){busy=false;submit.disabled=false;input.disabled=false;$('upload-progress-box').hidden=true;error(problem.message||'Upload failed. Check that the local app is running.');}
  });
  window.addEventListener('pagehide',()=>{if(pollTimer)clearTimeout(pollTimer);});
})();
