import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(here,'.build/before');
await fs.mkdir(out,{recursive:true});
const base=process.env.RUNTIME_NODE_MODULES;
const api=base?await import(pathToFileURL(path.join(base,'@oai/artifact-tool/dist/artifact_tool.mjs')).href):await import('@oai/artifact-tool');
const {PresentationFile,FileBlob}=api;
const input=process.env.PRESENTATION_SOURCE || path.resolve(here,'../../docs/Endterm_Defense.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(input));
const r=await p.inspect({kind:'slide,textbox,table,chart,notes',maxChars:100000});
await fs.writeFile(path.join(out,'inspect.json'),JSON.stringify(r));
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(out,`slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));
}
console.log('Inspected and rendered',p.slides.items.length,'source slides');
