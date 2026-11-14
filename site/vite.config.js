import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site's numbers are generated at build time by src/gen_data.mjs from the
// repo's study record (results/run_full.json, RESULTS.md, PREREGISTRATION.md,
// SPEC.md), so every figure on the page is parsed from the record, never typed.
export default defineConfig({
  base: '/cpp-ann-index-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
