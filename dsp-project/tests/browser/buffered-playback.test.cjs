const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname,'../../web/recording-viewer/buffered-playback.js'),'utf8');

function fixture(fetchAudio, decodeError=false, timeoutMs=500) {
  const status=[], revoked=[];
  const audio=new EventTarget();
  audio.load=()=>queueMicrotask(()=>audio.dispatchEvent(new Event(decodeError?'error':'loadedmetadata')));
  const context={window:{}, fetch:fetchAudio, AbortController, Blob, setTimeout, clearTimeout,
    URL:{createObjectURL:blob=>{assert.equal(blob.type,'audio/mp4');return 'blob:fixture';},revokeObjectURL:url=>revoked.push(url)}};
  vm.runInNewContext(source,context);
  const adapter=context.window.createBufferedPlayback({audio,source:'fixture.m4a',mimeType:'audio/mp4',timeoutMs,onStatus:text=>status.push(text)});
  return {adapter,audio,status,revoked};
}
const good=()=>Promise.resolve({ok:true,blob:()=>Promise.resolve(new Blob(['mock audio']))});
test('Concurrent requests download once, decode a typed blob, then reuse it',async()=>{
  let calls=0;
  const f=fixture((url,options)=>{calls++;assert.equal(url,'fixture.m4a');assert.equal(options.credentials,'omit');return good();});
  await Promise.all([f.adapter.ensure(),f.adapter.ensure()]);
  await f.adapter.ensure();
  assert.equal(calls,1);assert.equal(f.audio.src,'blob:fixture');assert.match(f.status.at(-1),/Audio ready/);
});
test('Download failures are visible and can be retried',async()=>{
  let calls=0;
  const f=fixture(()=>++calls===1?Promise.resolve({ok:false}):good());
  await assert.rejects(f.adapter.ensure());assert.match(f.status.at(-1),/retry/);
  await f.adapter.ensure();assert.equal(calls,2);
});
test('Decode errors release their temporary object URL',async()=>{
  const f=fixture(good,true);await assert.rejects(f.adapter.ensure());assert.deepEqual(f.revoked,['blob:fixture']);
});
test('A stalled download times out and remains retryable',async()=>{
  const f=fixture((_,options)=>new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new Error('aborted')))),false,10);
  await assert.rejects(f.adapter.ensure());assert.match(f.status.at(-1),/retry/);
});
