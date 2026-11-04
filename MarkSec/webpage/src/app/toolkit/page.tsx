import Container from '@/components/foundation/Container';
import Section from '@/components/foundation/Section';
import SiteHeader from '@/components/layout/SiteHeader';
import SiteFooter from '@/components/layout/SiteFooter';
import PaperHero from '@/components/paper/PaperHero';
import CopyBlock from '@/components/paper/CopyBlock';
import WatermarkWorkbench from '@/components/demo/WatermarkWorkbench';

const backendSetup = `git clone https://github.com/ZJU-TrustAID/MarkSec.git
cd MarkSec
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[api]"
python -m src.api --host 127.0.0.1 --port 8000`;

const frontendSetup = `# In a second terminal, from the repository root:
cd webpage
cp .env.example .env.local
npm ci
npm run dev
# Open http://localhost:3000/marksec/toolkit`;

export default function ToolkitPage() {
  return <main className="min-h-screen text-[var(--foreground)]">
    <SiteHeader navItems={[{ href: '#live-toolkit', label: 'Live toolkit' }, { href: '#get-started', label: 'Get started' }]} />
    <PaperHero id="toolkit-hero" title="MarkSec Toolkit"
      subtitle="Run real watermark generation, rewriting, and detection for your own prompt."
      bordered={false} compact />
    <Section id="live-toolkit"><Container><WatermarkWorkbench mode="live" /></Container></Section>
    <Section id="get-started" tone="blue"><Container>
      <h2 className="text-2xl font-semibold">Run the toolkit locally</h2>
      <div className="prose-block my-6 max-w-3xl">
        <p>Use Python 3.11 or 3.12 and Node.js 20.9 or newer. The default 135M model supports a small CPU trial;
          the first request downloads its weights. Generation and LLMP rewriting share one model.
          This interactive flow supports KGW and Unigram and does not replace the full benchmark protocol.</p>
        <p>Start the backend in the first terminal and the webpage in the second. On Windows, activate
          the environment with <code>.venv\Scripts\Activate.ps1</code>. See the{' '}
          <a href="https://github.com/ZJU-TrustAID/MarkSec">repository</a> for model configuration and API integration,
          and the <a href="https://github.com/ZJU-TrustAID/MarkSec/blob/main/docs/benchmark.md">benchmark guide</a>{' '}
          for the original GPU experiment runner.</p>
      </div>
      <div className="space-y-8">
        <section className="run-local-panel"><h3>1. Start the backend</h3><CopyBlock>{backendSetup}</CopyBlock></section>
        <section className="run-local-panel"><h3>2. Start the frontend</h3><CopyBlock>{frontendSetup}</CopyBlock></section>
      </div>
    </Container></Section>
    <SiteFooter note="Live toolkit. Open Demo to explore preset answers with the same interface." />
  </main>;
}
