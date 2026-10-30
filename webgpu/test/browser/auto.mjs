// What the page picks by itself ("auto") on each GPU setup of this machine, read from the chip and Engine -> Details.
// Hardware-specific on purpose: it reports, and only fails if the software-renderer setups do not end on the CPU.
// Usage: node test/browser/auto.mjs [variant ...]
import { launchChrome } from './chrome.mjs';
import { VARIANTS, startPreview } from './site.mjs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const wanted = process.argv.slice(2);
const variants = wanted.length ? wanted : ['webgl-swiftshader', 'webgpu-swiftshader', 'webgl-igpu', 'webgl-nvidia', 'webgpu-nvidia'];
const preview = await startPreview();
let failed = false;
try {
  for (const name of variants) {
    const v = VARIANTS[name];
    let seen;
    for (let attempt = 1; attempt <= 3; attempt++) {
      const chrome = await launchChrome(v.args, v.env);
      try {
        await chrome.goto(preview.url + '/');
        for (let i = 0; i < 400; i++) {
          if (await chrome.run(() => document.getElementById('loader').classList.contains('done') || document.getElementById('loader').classList.contains('failed'))) break;
          await sleep(150);
        }
        seen = await chrome.run(() => ({
          chip: document.getElementById('engine-chip-text').textContent,
          gpu: document.getElementById('stat-gpu').textContent,
          startup: document.getElementById('stat-calibration').textContent,
          advice: document.getElementById('webgpu-reason').textContent,
          why: !document.getElementById('why-btn').hidden,
          steps: [...document.querySelectorAll('#webgpu-steps li')].map((li) => li.textContent).filter((t) => /WebGL/.test(t)),
        }));
      } finally {
        await chrome.close();
      }
      // headless Chrome sometimes fails to hand out its GPU adapter; try again before believing a fallback
      if (!(v.backend === 'webgpu' && seen.chip !== 'WebGPU')) break;
    }
    console.log(name.padEnd(20), JSON.stringify(seen));
    if (/swiftshader/.test(name) && !seen.chip.startsWith('CPU')) failed = true;
  }
} finally {
  preview.stop();
}
process.exit(failed ? 1 : 0);
