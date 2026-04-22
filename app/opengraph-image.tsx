import { ImageResponse } from 'next/og'

export const runtime = 'edge'
export const alt = 'ML from Scratch — Derive it yourself'
export const size = { width: 1200, height: 630 }
export const contentType = 'image/png'

export default function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          backgroundColor: '#080808',
          padding: '64px 72px',
          fontFamily: 'monospace',
          position: 'relative',
        }}
      >
        {/* Top accent line */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: 3,
          background: 'linear-gradient(90deg, transparent, #a78bfa, transparent)',
        }} />

        {/* Grid dots background — subtle */}
        <div style={{
          position: 'absolute', inset: 0,
          backgroundImage: 'radial-gradient(circle, #1e1e1e 1px, transparent 1px)',
          backgroundSize: '40px 40px',
          opacity: 0.4,
        }} />

        {/* Top: badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, position: 'relative' }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '8px 16px',
            border: '1px solid #2a2a2a',
            borderRadius: 6,
            backgroundColor: '#111',
          }}>
            <span style={{ color: '#a78bfa', fontSize: 20 }}>ml ›</span>
            <span style={{ color: '#555', fontSize: 18 }}>harshithvarma.in</span>
          </div>
          <div style={{ color: '#333', fontSize: 18 }}>75 lessons · 14 sections · 168+ widgets</div>
        </div>

        {/* Middle: headline */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, position: 'relative' }}>
          <div style={{ color: '#e8e8e8', fontSize: 96, fontWeight: 700, lineHeight: 1.0, letterSpacing: -2 }}>
            Derive it
          </div>
          <div style={{ color: '#a78bfa', fontSize: 96, fontWeight: 700, lineHeight: 1.0, letterSpacing: -2 }}>
            yourself.
          </div>
          <div style={{ color: '#666', fontSize: 28, lineHeight: 1.4, marginTop: 8, maxWidth: 700 }}>
            From gradient descent to GPT — every equation proven, every line of code built from zero.
          </div>
        </div>

        {/* Bottom: proof points */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 40, position: 'relative' }}>
          <ProofPoint symbol="∇" text="Math first" />
          <Divider />
          <ProofPoint symbol="⌥" text="NumPy → PyTorch" />
          <Divider />
          <ProofPoint symbol="◈" text="Live widgets" />
          <div style={{ marginLeft: 'auto', color: '#444', fontSize: 20 }}>
            ml.harshithvarma.in
          </div>
        </div>
      </div>
    ),
    { ...size },
  )
}

function ProofPoint({ symbol, text }: { symbol: string; text: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{ color: '#a78bfa', fontSize: 22 }}>{symbol}</span>
      <span style={{ color: '#888', fontSize: 20 }}>{text}</span>
    </div>
  )
}

function Divider() {
  return <span style={{ color: '#222', fontSize: 20 }}>·</span>
}
