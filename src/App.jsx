import { useState } from 'react';
import DepthHeuristic from './DepthHeuristic.jsx';
import DepthMidas from './DepthMidas.jsx';

export default function App() {
  const [mode, setMode] = useState('midas');

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(160deg, #0a0a0f 0%, #0f1018 50%, #12101e 100%)',
      color: '#e0dfe6',
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0a0a0f; }
      `}</style>

      {/* Top nav */}
      <div style={{
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        padding: '12px 24px',
        display: 'flex', gap: 8, alignItems: 'center',
        background: 'rgba(0,0,0,0.3)',
        backdropFilter: 'blur(12px)',
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        <span style={{
          fontFamily: "'Space Grotesk', sans-serif",
          fontWeight: 700, fontSize: 16,
          color: '#ffffff',
          marginRight: 16,
        }}>
          Djupuppskattning
        </span>
        <NavBtn active={mode === 'midas'} onClick={() => setMode('midas')}>
          Neuralt nätverk
        </NavBtn>
        <NavBtn active={mode === 'heuristic'} onClick={() => setMode('heuristic')}>
          Heuristik
        </NavBtn>
      </div>

      {mode === 'midas' ? <DepthMidas /> : <DepthHeuristic />}
    </div>
  );
}

function NavBtn({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      padding: '6px 14px',
      border: `1px solid ${active ? 'rgba(0,212,255,0.4)' : 'rgba(255,255,255,0.1)'}`,
      borderRadius: 8,
      background: active
        ? 'linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,97,255,0.15))'
        : 'rgba(255,255,255,0.04)',
      color: active ? '#00d4ff' : '#9995a8',
      cursor: 'pointer',
      fontFamily: 'inherit',
      fontSize: 12,
      transition: 'all 0.2s',
    }}>
      {children}
    </button>
  );
}
