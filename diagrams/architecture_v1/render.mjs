// Render the Mermaid blocks in docs/architecture; no network or model execution.
// Usage: node render.mjs /path/to/mermaid-11.12.0.min.js /path/to/playwright [chromium-binary]
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const out = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(out, '../..');
const [mermaidPath, playwrightPath, browserPath] = process.argv.slice(2);
if (!mermaidPath || !playwrightPath) {
  throw new Error('Supply the pinned Mermaid bundle and local Playwright package path.');
}
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const rendererSha = '07e37dfa97b337ccc85365d57eddf99b9706f09db3b59b260d0333b23b343c4b';
if (digest(await fs.readFile(mermaidPath)) !== rendererSha) {
  throw new Error('Mermaid bundle differs from the reviewed 11.12.0 artifact.');
}
const require = createRequire(import.meta.url);
const { chromium } = require(path.resolve(playwrightPath));
const docDir = path.join(root, 'docs/architecture');
const diagrams = [];
for (const name of (await fs.readdir(docDir)).filter(n => /^\d.*\.md$/.test(n)).sort()) {
  const content = await fs.readFile(path.join(docDir, name), 'utf8');
  let count = 0;
  for (const match of content.matchAll(/^```mermaid\n([\s\S]*?)^```/gm)) {
    const headings = [...content.slice(0, match.index).matchAll(/^#{1,4} (.+)$/gm)];
    const id = `${name.slice(0, 2)}_${String(++count).padStart(2, '0')}`;
    const source = match[1].trimEnd() + '\n';
    await fs.writeFile(path.join(out, `${id}.mmd`), source);
    diagrams.push({
      id, title: headings.at(-1)[1], document: `docs/architecture/${name}`,
      source_line: content.slice(0, match.index).split('\n').length,
      source_sha256: digest(source), mermaid: `diagrams/architecture_v1/${id}.mmd`,
      svg: `diagrams/architecture_v1/${id}.svg`,
    });
  }
}
const browser = await chromium.launch({ headless: true, executablePath: browserPath || chromium.executablePath() });
const browserVersion = browser.version();
const selectedPreviews = new Set(['01_01', '01_02', '02_01', '02_05', '03_01', '03_09', '04_03', '04_08']);
try {
  const page = await browser.newPage({ viewport: { width: 1800, height: 1200 }, deviceScaleFactor: 1 });
  // Every input is local. Block incidental font or stylesheet requests.
  await page.route('**/*', route => route.abort());
  await page.setContent('<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:16px;background:white"><div id="diagram"></div></body></html>');
  await page.addScriptTag({ path: path.resolve(mermaidPath) });
  await page.evaluate(() => mermaid.initialize({
    startOnLoad: false, securityLevel: 'strict', theme: 'base',
    themeVariables: {
      fontFamily: 'Arial, sans-serif', fontSize: '17px', primaryColor: '#eef4ff',
      primaryBorderColor: '#356390', primaryTextColor: '#152e47',
      lineColor: '#597184', secondaryColor: '#eef8f4', tertiaryColor: '#fff8e7',
    },
    flowchart: { htmlLabels: false, useMaxWidth: false, nodeSpacing: 28, rankSpacing: 44, padding: 14 },
    sequence: { useMaxWidth: false, wrap: true, actorMargin: 36, messageMargin: 30 },
    state: { useMaxWidth: false },
  }));
  for (const row of diagrams) {
    const source = await fs.readFile(path.join(root, row.mermaid), 'utf8');
    const rendered = await page.evaluate(async ({ id, source }) => {
      const host = document.getElementById('diagram');
      host.innerHTML = '';
      const { svg } = await mermaid.render(`replica_${id}`, source);
      host.innerHTML = svg;
      const node = host.querySelector('svg');
      const box = node.viewBox.baseVal;
      node.setAttribute('width', String(Math.ceil(box.width)));
      node.setAttribute('height', String(Math.ceil(box.height)));
      return { svg: node.outerHTML, width: Math.ceil(box.width), height: Math.ceil(box.height), text_nodes: node.querySelectorAll('text').length };
    }, { id: row.id, source });
    await fs.writeFile(path.join(root, row.svg), rendered.svg + '\n');
    row.svg_sha256 = digest(rendered.svg + '\n');
    row.width = rendered.width;
    row.height = rendered.height;
    row.text_nodes = rendered.text_nodes;
    if (selectedPreviews.has(row.id)) {
      row.png = `diagrams/architecture_v1/${row.id}.png`;
      await page.locator('#diagram > svg').screenshot({ path: path.join(root, row.png) });
      row.png_sha256 = digest(await fs.readFile(path.join(root, row.png)));
    }
    process.stdout.write(`${row.id} ${row.width}x${row.height} rendered\n`);
  }
} finally {
  await browser.close();
}
const manifest = {
  schema_version: 'replica.architecture_diagrams.v1',
  kind: 'proposed_design_not_measured_behavior',
  renderer: { name: 'mermaid', version: '11.12.0', sha256: rendererSha,
    source: 'https://cdn.jsdelivr.net/npm/mermaid@11.12.0/dist/mermaid.min.js',
    browser: 'Playwright Chromium', browser_version: browserVersion, external_requests_blocked: true },
  diagrams,
};
await fs.writeFile(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
process.stdout.write(`Validated and rendered ${diagrams.length} component diagrams.\n`);
