const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const context={window:{},document:{addEventListener:()=>{}}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../../web/recording-viewer/pipeline.js'),'utf8'),context);
const math=context.window.PipelineNotebook;
test('Context and core sizes remain separate',()=>{
  const g=math.chunkGeometry(1.46,1,16000);
  assert.equal(g.crop,3.46);assert.equal(g.samples,55360);
  assert.ok(Math.abs(g.leftFraction*2+g.coreFraction-1)<1e-10);
  assert.equal(math.chunkGeometry(6,1,16000).samples,128000);
});
test('No-context illustration is valid; invalid sizes are not',()=>{
  assert.equal(math.chunkGeometry(2,0,16000).coreFraction,1);
  for(const input of [[-1,1,16000],[1,-1,16000],[1,1,0],[NaN,1,16000]])assert.throws(()=>math.chunkGeometry(...input));
});
test('Decision filtering preserves evidence and handles empty results',()=>{
  const data=[{id:'a',status:'rejected',reason:'match'},{id:'b',status:'skipped',reason:'budget'}];
  assert.equal(math.filterReports(data,'all','all').length,2);
  assert.equal(math.filterReports(data,'skipped','budget')[0].id,'b');
  assert.equal(math.filterReports(data,'accepted','all').length,0);
  assert.equal(data.length,2);
});
test('Timestamps round consistently at minute boundaries',()=>{
  assert.equal(math.time(378.389),'6:18.39');assert.equal(math.time(59.999),'1:00.00');
});
