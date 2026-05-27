import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  publicDir: false,
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    rollupOptions: {
      input: "index.html",
    },
  },
  server: {
    proxy: {
      "/search": "http://localhost:5000",
      "/api": "http://localhost:5000",
      "/videos": "http://localhost:5000",
      "/keyframes": "http://localhost:5000",
      "/hls": "http://localhost:5000",
    },
  },
});
