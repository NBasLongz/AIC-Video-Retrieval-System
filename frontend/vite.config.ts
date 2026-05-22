import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": "/src",
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

