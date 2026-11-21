// Bundles src/app.js and copies the static files and the Basic Pitch model into dist/.
import { build } from 'esbuild';
import { cpSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';

rmSync('dist', { recursive: true, force: true });
mkdirSync('dist/model', { recursive: true });

await build({
  entryPoints: ['src/app.js'],
  bundle: true,
  minify: true,
  format: 'esm',
  target: 'es2020',
  platform: 'browser',
  outfile: 'dist/app.js',
  logLevel: 'info',
});

cpSync('index.html', 'dist/index.html');
cpSync('style.css', 'dist/style.css');
cpSync('favicon.svg', 'dist/favicon.svg');
cpSync('node_modules/@spotify/basic-pitch/model', 'dist/model', { recursive: true });
// GitHub Pages: serve files as-is (no Jekyll processing).
writeFileSync('dist/.nojekyll', '');
