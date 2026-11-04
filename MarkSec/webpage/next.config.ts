import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  basePath: "/marksec",
  experimental: { cpus: 2 },
};

export default nextConfig;
