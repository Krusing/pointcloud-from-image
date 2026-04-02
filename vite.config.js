import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ["@xenova/transformers"],
  },
  resolve: {
    alias: { "onnxruntime-web": "onnxruntime-web/dist/ort.min.js" },
  },
});
