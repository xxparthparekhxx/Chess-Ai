// What to tell the player when WebGPU is not in use: why, and how to turn it on in their browser.
// Flag names checked against Chrome 156 (chrome://flags on Linux lists "Unsafe WebGPU Support" = #enable-unsafe-webgpu
// and "Vulkan" = #enable-vulkan) and the Firefox about:config preference dom.webgpu.enabled.

import type { ReadyInfo } from '../messages';

export interface Advice {
  /** one or two sentences: what is running instead and why */
  reason: string;
  /** steps for this browser and system, plain text; chrome:// and about: addresses are shown as code, not as links */
  steps: string[];
}

export type Browser = 'chrome' | 'edge' | 'firefox' | 'safari' | 'other';
export type System = 'linux' | 'windows' | 'mac' | 'android' | 'ios' | 'other';

export function detectBrowser(ua: string): Browser {
  if (/Firefox\//.test(ua)) return 'firefox';
  if (/Edg\//.test(ua)) return 'edge';
  if (/Chrome\/|Chromium\/|CriOS\//.test(ua)) return 'chrome';
  if (/Safari\//.test(ua)) return 'safari';
  return 'other';
}

export function detectSystem(ua: string): System {
  if (/Android/.test(ua)) return 'android';
  if (/iPhone|iPad|iPod/.test(ua)) return 'ios';
  if (/Windows/.test(ua)) return 'windows';
  if (/Mac OS X|Macintosh/.test(ua)) return 'mac';
  if (/Linux|X11|CrOS/.test(ua)) return 'linux';
  return 'other';
}

type Info = Pick<ReadyInfo, 'backend' | 'webgpuProblem' | 'webgpuNote' | 'webglNote' | 'gpu'>;

function dualGpuTip(system: System): string {
  return system === 'linux'
    ? 'On a laptop with two GPUs the browser may be using the weaker one. Start the browser on the dedicated GPU, for example with DRI_PRIME=1 (AMD / Intel) or __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia (NVIDIA) in front of the command.'
    : 'On a laptop with two GPUs, set the browser to use the high-performance GPU in the system graphics settings.';
}

/** null when WebGPU is the backend in use */
export function webgpuAdvice(info: Info, ua: string): Advice | null {
  if (info.backend === 'webgpu') return null;
  const browser = detectBrowser(ua);
  const system = detectSystem(ua);
  const using = info.backend === 'webgl' ? `WebGL${info.gpu ? ` (${info.gpu})` : ''}` : 'the CPU';

  let why: string;
  switch (info.webgpuProblem) {
    case 'no-api':
      why = 'this browser does not offer WebGPU (navigator.gpu is missing).';
      break;
    case 'no-adapter':
      why = 'the browser offers WebGPU but found no usable GPU for it (the GPU or driver is not supported, or Vulkan is not available).';
      break;
    case 'forced-off':
      why = 'the page address has ?backend= in it, which picks the backend by hand. Remove it to use WebGPU when it is available.';
      break;
    case 'no-jspi':
      why = 'WebGPU is there, but this browser lacks the WebAssembly feature (JSPI) the bundled WebGPU runtime needs, and the fallback runtime is switched off.';
      break;
    case 'slow-gpu':
    case 'adapter-error':
    case 'runtime-failed':
      why = (info.webgpuNote ?? 'WebGPU could not be started.').replace(/^./, (c) => c.toLowerCase());
      break;
    default:
      why = info.webgpuNote ? info.webgpuNote.replace(/^./, (c) => c.toLowerCase()) : 'it was not available.';
  }
  const reason = `Running on ${using} because ${why}`;

  const steps: string[] = [];
  if (info.webgpuProblem === 'forced-off') return { reason, steps };

  if (info.webgpuProblem === 'slow-gpu') {
    steps.push(dualGpuTip(system));
    return { reason, steps };
  }

  if (browser === 'chrome' || browser === 'edge') {
    const scheme = browser === 'edge' ? 'edge' : 'chrome';
    if (system === 'linux') {
      steps.push(`Open ${scheme}://flags/#enable-unsafe-webgpu and set "Unsafe WebGPU Support" to Enabled.`);
      steps.push(`Open ${scheme}://flags/#enable-vulkan and set "Vulkan" to Enabled.`);
      steps.push('Relaunch the browser. WebGPU on Linux needs a working Vulkan driver (vulkaninfo --summary should list your GPU).');
      steps.push(`Check ${scheme}://gpu: the WebGPU line should say it is hardware accelerated.`);
    } else if (system === 'android' || system === 'ios') {
      steps.push('Update the browser and the system. WebGPU needs a recent Android (Chrome 121 or later) and a supported GPU.');
    } else {
      steps.push(`Update the browser (WebGPU is on by default in version 113 and later) and the graphics driver, then restart it.`);
      steps.push(`Open ${scheme}://settings/system and turn on "Use graphics acceleration when available", then relaunch.`);
      steps.push(`Check ${scheme}://gpu: the WebGPU line should say it is hardware accelerated.`);
    }
  } else if (browser === 'firefox') {
    steps.push('Open about:config, accept the warning, set dom.webgpu.enabled to true and restart Firefox.');
    steps.push(
      system === 'linux'
        ? 'On Linux, Firefox only supports WebGPU in Nightly builds so far, so the setting may do nothing in the regular release. Chrome or Edge with the flags above is the reliable way.'
        : system === 'windows'
          ? 'Firefox 141 and later turn WebGPU on by default on Windows, so updating may be enough.'
          : 'On other systems WebGPU in Firefox is still experimental; Chrome or Edge is the reliable way.',
    );
  } else if (browser === 'safari') {
    steps.push('Safari 26 and later support WebGPU. Update macOS / iOS and Safari.');
  } else {
    steps.push('Use a recent Chrome or Edge. Other browsers either lack WebGPU or have it switched off.');
  }
  if (info.backend === 'wasm' && info.webglNote) {
    steps.push(`WebGL was not used either: ${info.webglNote}`);
    if (/positions\/s/.test(info.webglNote)) steps.push(dualGpuTip(system));
  }
  return { reason, steps };
}
