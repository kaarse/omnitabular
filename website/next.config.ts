import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/omnitabular",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
