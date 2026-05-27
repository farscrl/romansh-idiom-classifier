import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  root: "demo",
  plugins: [viteSingleFile()],
  build: {
    outDir: "../demo-dist",
    emptyOutDir: true,
  },
});