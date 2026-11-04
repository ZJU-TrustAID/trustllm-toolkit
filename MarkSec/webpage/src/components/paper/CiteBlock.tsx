import CopyBlock from "@/components/paper/CopyBlock";
import { paper } from "@/content/paper";

export default function CiteBlock() {
  return <CopyBlock>{paper.bibtex}</CopyBlock>;
}
