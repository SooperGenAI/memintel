import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

const interFont = (
  <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');`}</style>
);

function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroInner}>
        <div className={styles.badge}>Deterministic · Reproducible · Auditable</div>
        <h1 className={styles.heroTitle}>
          The decision layer<br />for agentic AI
        </h1>
        <p className={styles.heroSubtitle}>
          Memintel compiles natural language intent into deterministic
          execution graphs. Same input. Same decision. Every time.
        </p>
        <div className={styles.heroCta}>
          <Link className={styles.ctaPrimary} to="/docs/intro/overview">
            Read the Docs
          </Link>
          <Link className={styles.ctaSecondary} to="/docs/api-reference/overview">
            API Reference
          </Link>
        </div>
        <div className={styles.pipeline}>
          <div className={styles.pipelineStep}>
            <span className={styles.stepLabel}>Intent</span>
            <span className={styles.stepDesc}>You describe what to monitor</span>
          </div>
          <div className={styles.pipelineArrow}>→</div>
          <div className={styles.pipelineStep}>
            <span className={styles.stepLabel}>Concept ψ</span>
            <span className={styles.stepDesc}>System computes the signal</span>
          </div>
          <div className={styles.pipelineArrow}>→</div>
          <div className={styles.pipelineStep}>
            <span className={styles.stepLabel}>Condition φ</span>
            <span className={styles.stepDesc}>Memintel decides if it matters</span>
          </div>
          <div className={styles.pipelineArrow}>→</div>
          <div className={styles.pipelineStep}>
            <span className={styles.stepLabel}>Action α</span>
            <span className={styles.stepDesc}>Your system executes</span>
          </div>
        </div>
      </div>
    </section>
  );
}

interface FeatureProps { icon: string; title: string; description: string; }

function Feature({icon, title, description}: FeatureProps) {
  return (
    <div className={styles.feature}>
      <div className={styles.featureIcon}>{icon}</div>
      <h3 className={styles.featureTitle}>{title}</h3>
      <p className={styles.featureDesc}>{description}</p>
    </div>
  );
}

const features: FeatureProps[] = [
  { icon: '⚙️', title: 'Deterministic by design',    description: 'Same input, same guardrails, same decision — every execution. No probabilistic drift, no LLM on the hot path.' },
  { icon: '🔍', title: 'Fully auditable',             description: 'Every decision is traceable: which primitives were fetched, which concept was computed, which strategy fired and why.' },
  { icon: '📐', title: 'Strategy-driven conditions',  description: 'Conditions evaluate meaning through structured strategies — threshold, percentile, z-score, change, composite — not prompt heuristics.' },
  { icon: '🔄', title: 'Calibration without mutation',description: 'Feedback drives parameter recommendations. Applying calibration creates a new immutable version. Historical decisions stay reproducible.' },
  { icon: '🏗️', title: 'Guardrails system',           description: 'Admin-defined policy layer constrains LLM output at task creation time. Strategy registry, type-compatibility, parameter priors, bias rules.' },
  { icon: '🧩', title: 'Composable primitives',       description: 'Concepts compose from versioned primitives. Features derive intermediate signals. The entire graph is typed, validated, and version-pinned.' },
];

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      {interFont}
      <main>
        <Hero />
        <div className={styles.divider} />
        <section className={styles.features}>
          {/* FIX 5: proper heading size for "Why Memintel" */}
          <h2 className={styles.featuresHeading}>Why Memintel</h2>
          <div className={styles.featuresGrid}>
            {features.map((f) => <Feature key={f.title} {...f} />)}
          </div>
        </section>
        <section className={styles.quicklinks}>
          {/* FIX 5: proper heading size for "Jump in" */}
          <h2 className={styles.quicklinksTitle}>Jump in</h2>
          <div className={styles.quicklinksGrid}>
            <Link className={styles.quicklink} to="/docs/intro/overview">
              <strong>Introduction</strong>
              <span>Understand the architecture and core concepts</span>
            </Link>
            <Link className={styles.quicklink} to="/docs/intro/quickstart">
              <strong>Quickstart</strong>
              <span>Build your first deterministic decision loop in 5 minutes</span>
            </Link>
            {/* Links use the slug: field from each doc's frontmatter */}
            <Link className={styles.quicklink} to="/docs/api-reference/overview">
              <strong>App Developer API</strong>
              <span>Tasks, execution, conditions, feedback — full reference</span>
            </Link>
            <Link className={styles.quicklink} to="/docs/python-sdk/python-overview">
              <strong>Python SDK</strong>
              <span>FastAPI backend — execution engine, registry, agents</span>
            </Link>
          </div>
        </section>
      </main>
    </Layout>
  );
}
