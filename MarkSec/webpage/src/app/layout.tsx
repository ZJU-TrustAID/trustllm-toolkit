import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import PatchMarkClient from "@/components/PatchMarkClient";
import "./globals.css";
import { paper } from "@/content/paper";

export const metadata: Metadata = {
  title: paper.title,
  authors: paper.authors.map(({ name }) => ({ name })),
  description:
    "A research project and Python toolkit for evaluating attacks against LLM watermarking systems.",
  icons: {
    icon: "/marksec/marksec-icon-transparent.png",
    shortcut: "/marksec/marksec-icon-transparent.png",
    apple: "/marksec/marksec-icon-transparent.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="blueprint">
      <body>
        {children}
        {process.env.NEXT_PUBLIC_MARKSEC_ENABLE_ANNOTATIONS === "true" ? <PatchMarkClient /> : null}
      </body>
    </html>
  );
}
