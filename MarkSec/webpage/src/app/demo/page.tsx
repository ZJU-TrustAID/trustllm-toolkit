import Container from '@/components/foundation/Container';
import Section from '@/components/foundation/Section';
import SiteHeader from '@/components/layout/SiteHeader';
import SiteFooter from '@/components/layout/SiteFooter';
import PaperHero from '@/components/paper/PaperHero';
import CopyBlock from '@/components/paper/CopyBlock';
import WatermarkWorkbench from '@/components/demo/WatermarkWorkbench';

export default function DemoPage() {
  return <main className="min-h-screen text-[var(--foreground)]">
    <SiteHeader navItems={[{ href: '#interactive-demo', label: 'Demo' }, { href: '#run-locally', label: 'Run locally' }]} />
    <PaperHero id="demo-hero" title="MarkSec Demo"
      subtitle="Explore preset watermark and rewriting examples without a model server."
      bordered={false} compact />
    <Section id="interactive-demo"><Container><WatermarkWorkbench mode="preset" /></Container></Section>
    <Section id="run-locally" tone="blue"><Container>
      <h2 className="text-2xl font-semibold">Run the preset demo</h2>
      <p className="my-4">Node.js 20.9 or newer is sufficient. Open /marksec/demo after starting the app.</p>
      <CopyBlock>{`git clone https://github.com/ZJU-TrustAID/MarkSec.git
cd MarkSec/webpage
npm ci
npm run dev`}</CopyBlock>
    </Container></Section>
    <SiteFooter note="Preset demo. Open Toolkit for live model execution with the same interface." />
  </main>;
}
