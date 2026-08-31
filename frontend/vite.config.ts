import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the built dist/ opens straight from the filesystem — the
// offline-presentation requirement: no server, no network, double-click and go.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "dist", assetsInlineLimit: 4096 },
  server: { port: 5173 },
});
