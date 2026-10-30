// Cheap checks on the static pieces that cannot be exercised without a browser.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { WEB_ROOT } from './helpers/fixtures';

const read = (p: string) => readFileSync(resolve(WEB_ROOT, p), 'utf8');

describe('index.html and the UI code agree', () => {
  const html = read('index.html');
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]!));

  it('has every element id the UI looks up', () => {
    const used = [...read('src/ui/app.ts').matchAll(/\$(?:<[^>]*>)?\('([^']+)'\)/g)].map((m) => m[1]!);
    expect(used.length).toBeGreaterThan(20);
    const missing = [...new Set(used)].filter((id) => !ids.has(id));
    expect(missing).toEqual([]);
  });

  it('has no duplicate ids', () => {
    const all = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]!);
    expect(all.length).toBe(new Set(all).size);
  });

  it('loads the entry module', () => {
    expect(html).toContain('src="/src/main.ts"');
  });
});

describe('deploy config', () => {
  it('wrangler.jsonc is assets-only and serves dist/ on the custom domain', () => {
    const text = read('wrangler.jsonc')
      .split('\n')
      .map((l) => l.replace(/^\s*\/\/.*$/, ''))
      .join('\n');
    const cfg = JSON.parse(text) as Record<string, any>;
    expect(cfg.assets.directory).toBe('./dist');
    expect(cfg.main).toBeUndefined(); // no Worker script: static assets only
    expect(cfg.routes).toEqual([{ pattern: 'chess.parthparekhdev.com', custom_domain: true }]);
  });

  it('_headers sets cross-origin isolation and caching', () => {
    const h = read('public/_headers');
    expect(h).toMatch(/Cross-Origin-Opener-Policy: same-origin/);
    expect(h).toMatch(/Cross-Origin-Embedder-Policy: require-corp/);
    expect(h).toMatch(/\/assets\/\*\n\s+Cache-Control: public, max-age=31536000, immutable/);
    expect(h).toMatch(/\/ort\/\*\n\s+Cache-Control: public, max-age=31536000, immutable/);
  });

  it('production build points at Hugging Face and the dev default is local', () => {
    expect(read('.env.production')).toContain('https://huggingface.co/xxparthparekhxx/sigma-go/resolve/main/chess_net.onnx');
    expect(read('src/config.ts')).toContain("'/model/chess_net.onnx'");
  });

  it('git ignores the exported network', () => {
    expect(read('.gitignore')).toMatch(/^model\/$/m);
    expect(readFileSync(resolve(WEB_ROOT, '..', '.gitignore'), 'utf8')).toMatch(/^\*\.onnx$/m);
  });
});
