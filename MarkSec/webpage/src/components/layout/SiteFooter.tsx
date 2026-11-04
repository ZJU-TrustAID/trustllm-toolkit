"use client";

import type { MouseEvent } from "react";
import { ArrowUp } from "lucide-react";
import SiteVisitCounter from "./SiteVisitCounter";
import VisitorMapDialog from "./VisitorMapDialog";

type SiteFooterProps = {
  note?: string;
};

export default function SiteFooter({
  note = "Research interface for reading, testing, and running watermark attack workflows.",
}: SiteFooterProps) {
  function handleTopClick(event: MouseEvent<HTMLAnchorElement>) {
    event.preventDefault();
    window.scrollTo({ top: 0, behavior: "smooth" });
    window.history.replaceState(null, "", "#top");
  }

  return (
    <footer className="site-footer">
      <div className="paper-container site-footer-inner">
        <div className="site-footer-note">
          <p>{note}</p>
          <div className="site-footer-tools">
            <SiteVisitCounter />
            <VisitorMapDialog />
          </div>
        </div>
        <a href="#top" onClick={handleTopClick} className="site-footer-top">
          Top
          <ArrowUp className="h-4 w-4" />
        </a>
      </div>
    </footer>
  );
}
