/**
 * Project Nexus — PitchSense Crowd Safety Dashboard
 * ===================================================
 * Real-time crowd density + acoustic sentiment monitoring.
 * Connects to the PitchSense WebSocket backend and displays:
 *   - Live stadium heatmap (density per sector)
 *   - Dual-signal alert score gauge
 *   - Acoustic sentiment waveform
 *   - Alert history log
 *   - Exit gate status
 *
 * Author  : Sanchayan (Unwilling-mcu)
 * GitHub  : github.com/Unwilling-mcu/ProjectNexus
 */

import { useState, useEffect, useRef, useCallback } from "react";

// ── Constants ─────────────────────────────────────────────────────
const SECTORS = [
  { id: "NW", label: "North West",  x: 0,   y: 0,   w: 33, h: 33 },
  { id: "N",  label: "North",       x: 33,  y: 0,   w: 34, h: 33 },
  { id: "NE", label: "North East",  x: 67,  y: 0,   w: 33, h: 33 },
  { id: "W",  label: "West",        x: 0,   y: 33,  w: 33, h: 34 },
  { id: "C",  label: "Central",     x: 33,  y: 33,  w: 34, h: 34 },
  { id: "E",  label: "East",        x: 67,  y: 33,  w: 33, h: 34 },
  { id: "SW", label: "South West",  x: 0,   y: 67,  w: 33, h: 33 },
  { id: "S",  label: "South",       x: 33,  y: 67,  w: 34, h: 33 },
  { id: "SE", label: "South East",  x: 67,  y: 67,  w: 33, h: 33 },
];

const GATES = [
  { id: "G1", label: "Gate 1 — North Entrance", sector: "N" },
  { id: "G2", label: "Gate 2 — East Stand",     sector: "E" },
  { id: "G3", label: "Gate 3 — South Exit",     sector: "S" },
  { id: "G4", label: "Gate 4 — West Stand",     sector: "W" },
  { id: "G5", label: "Gate 5 — NE Corner",      sector: "NE" },
  { id: "G6", label: "Gate 6 — SW Corner",      sector: "SW" },
];

const ALERT_LEVELS = {
  green: { label: "Normal",    color: "#0F6E56", bg: "#E8F4F1", border: "#0F6E56" },
  amber: { label: "Caution",   color: "#BA7517", bg: "#FDF3E3", border: "#BA7517" },
  red:   { label: "CRITICAL",  color: "#C0392B", bg: "#FDECEA", border: "#C0392B" },
};

const AUDIO_CLASSES = ["Chant", "Crowd Roar", "Aggression"];
const AUDIO_COLORS  = ["#0F6E56", "#BA7517", "#C0392B"];
const WAVEFORM_POINTS = 60;

// ── Demo data generator ────────────────────────────────────────────
function generateDemoState(tick) {
  // Slowly rising tension arc over time
  const tension = 0.3 + 0.5 * Math.sin(tick * 0.04) + 0.1 * Math.sin(tick * 0.13);

  const densities = {};
  SECTORS.forEach(s => {
    const base = 0.4 + 0.3 * Math.sin(tick * 0.02 + s.x * 0.1);
    const spike = s.id === "N" && tick % 80 > 60 ? 0.4 : 0;
    densities[s.id] = Math.min(0.99, Math.max(0.05, base + spike + (Math.random() - 0.5) * 0.08));
  });

  const audioScores = [
    0.5 + 0.4 * Math.cos(tick * 0.07),
    0.3 + 0.3 * Math.sin(tick * 0.05),
    Math.max(0, tension - 0.5 + Math.random() * 0.2),
  ].map(v => Math.min(1, Math.max(0, v)));

  const visualScore = Object.values(densities).reduce((a, b) => a + b, 0) / SECTORS.length;
  const audioScore  = audioScores[2] * 0.6 + audioScores[1] * 0.4;
  const fusedScore  = 0.6 * visualScore + 0.4 * audioScore;

  const gateStatus = {};
  GATES.forEach(g => {
    const sec = densities[g.sector] || 0;
    gateStatus[g.id] = sec > 0.75 ? "open" : sec > 0.55 ? "monitor" : "normal";
  });

  const alert =
    fusedScore > 0.70 ? "red" :
    fusedScore > 0.40 ? "amber" : "green";

  return {
    timestamp: Date.now(),
    densities,
    audioScores,
    visualScore,
    audioScore,
    fusedScore,
    gateStatus,
    alert,
    capacity: 67000,
    present: Math.round(67000 * (0.88 + 0.08 * visualScore)),
  };
}

// ── Density colour ────────────────────────────────────────────────
function densityToColor(d) {
  if (d < 0.35) return "rgba(15,110,86,0.25)";
  if (d < 0.55) return "rgba(15,110,86,0.50)";
  if (d < 0.70) return "rgba(186,117,23,0.55)";
  if (d < 0.85) return "rgba(186,117,23,0.80)";
  return "rgba(192,57,43,0.85)";
}

// ── Components ────────────────────────────────────────────────────

function AlertBanner({ level }) {
  const cfg = ALERT_LEVELS[level] || ALERT_LEVELS.green;
  return (
    <div style={{
      background: cfg.bg,
      border: `2px solid ${cfg.border}`,
      borderRadius: 10,
      padding: "14px 20px",
      display: "flex",
      alignItems: "center",
      gap: 14,
      marginBottom: 20,
    }}>
      <div style={{
        width: 16, height: 16,
        borderRadius: "50%",
        background: cfg.color,
        boxShadow: level === "red" ? `0 0 12px ${cfg.color}` : "none",
        animation: level === "red" ? "pulse 1s infinite" : "none",
      }} />
      <div>
        <span style={{ fontWeight: 700, color: cfg.color, fontSize: 15 }}>
          {cfg.label}
        </span>
        <span style={{ color: "#555", fontSize: 13, marginLeft: 10 }}>
          {level === "green" && "All sectors within safe parameters."}
          {level === "amber" && "Elevated density detected. Additional stewards deployed."}
          {level === "red"   && "CRITICAL — Crowd pressure approaching threshold. Action required."}
        </span>
      </div>
    </div>
  );
}

function ScoreGauge({ score, label, color }) {
  const pct  = Math.round(score * 100);
  const dash = 2 * Math.PI * 45;
  const fill = dash * score;
  return (
    <div style={{ textAlign: "center", padding: "0 12px" }}>
      <svg width={110} height={110} viewBox="0 0 110 110">
        <circle cx={55} cy={55} r={45} fill="none"
                stroke="#eee" strokeWidth={10} />
        <circle cx={55} cy={55} r={45} fill="none"
                stroke={color} strokeWidth={10}
                strokeDasharray={`${fill} ${dash - fill}`}
                strokeDashoffset={dash / 4}
                strokeLinecap="round" />
        <text x={55} y={60} textAnchor="middle"
              fontSize={20} fontWeight={700} fill={color}>
          {pct}%
        </text>
      </svg>
      <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>{label}</div>
    </div>
  );
}

function StadiumHeatmap({ densities }) {
  return (
    <div style={{ position: "relative", width: "100%", paddingBottom: "100%",
                  background: "#1a3a2a", borderRadius: 10, overflow: "hidden",
                  border: "1px solid #2d5a3d" }}>
      <div style={{ position: "absolute", inset: 0, padding: 10 }}>
        {/* Pitch outline */}
        <div style={{
          position: "absolute", inset: "15%",
          border: "2px solid rgba(255,255,255,0.15)",
          borderRadius: 4,
          boxSizing: "border-box",
        }} />
        {/* Centre circle */}
        <div style={{
          position: "absolute", top: "40%", left: "40%",
          width: "20%", height: "20%",
          border: "2px solid rgba(255,255,255,0.1)",
          borderRadius: "50%",
        }} />
        {/* Sector overlays */}
        {SECTORS.map(s => (
          <div key={s.id} style={{
            position: "absolute",
            left: `${s.x}%`, top: `${s.y}%`,
            width: `${s.w}%`, height: `${s.h}%`,
            background: densityToColor(densities[s.id] || 0),
            transition: "background 0.5s ease",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}>
            <span style={{
              color: "rgba(255,255,255,0.9)",
              fontSize: 11,
              fontWeight: 600,
              textShadow: "0 1px 3px rgba(0,0,0,0.8)",
            }}>
              {s.id}
              <br />
              <span style={{ fontWeight: 400, fontSize: 10 }}>
                {Math.round((densities[s.id] || 0) * 100)}%
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AudioWaveform({ history, audioScores }) {
  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
        {AUDIO_CLASSES.map((cls, i) => (
          <div key={cls} style={{
            display: "flex", alignItems: "center", gap: 5,
            fontSize: 12, color: "#555",
          }}>
            <div style={{ width: 10, height: 3, borderRadius: 2,
                          background: AUDIO_COLORS[i] }} />
            {cls}: <strong style={{ color: AUDIO_COLORS[i] }}>
              {Math.round(audioScores[i] * 100)}%
            </strong>
          </div>
        ))}
      </div>
      <svg width="100%" height={80} viewBox={`0 0 ${WAVEFORM_POINTS} 80`}
           preserveAspectRatio="none">
        {AUDIO_CLASSES.map((cls, ci) => {
          const pts = history.map((h, xi) => {
            const y = 80 - (h.audioScores?.[ci] || 0) * 70;
            return `${xi},${y}`;
          }).join(" ");
          return (
            <polyline key={cls} points={pts}
                      fill="none"
                      stroke={AUDIO_COLORS[ci]}
                      strokeWidth={1.5}
                      opacity={0.85} />
          );
        })}
      </svg>
    </div>
  );
}

function GatePanel({ gateStatus }) {
  const colors = { normal: "#0F6E56", monitor: "#BA7517", open: "#C0392B" };
  const labels = { normal: "Normal", monitor: "Monitor", open: "OPEN" };
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
      {GATES.map(g => {
        const status = gateStatus[g.id] || "normal";
        return (
          <div key={g.id} style={{
            background: "#f9f9f9",
            border: `1.5px solid ${colors[status]}`,
            borderRadius: 8,
            padding: "8px 12px",
          }}>
            <div style={{ fontSize: 11, color: "#888", marginBottom: 2 }}>{g.id}</div>
            <div style={{ fontSize: 12, color: "#333", marginBottom: 4, lineHeight: 1.3 }}>
              {g.label}
            </div>
            <div style={{
              display: "inline-block",
              fontSize: 11,
              fontWeight: 700,
              color: colors[status],
              background: colors[status] + "18",
              padding: "2px 8px",
              borderRadius: 4,
            }}>
              {labels[status]}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AlertLog({ log: alertLog }) {
  return (
    <div style={{ maxHeight: 180, overflowY: "auto" }}>
      {alertLog.length === 0 && (
        <div style={{ color: "#aaa", fontSize: 13, padding: "12px 0" }}>
          No alerts this session.
        </div>
      )}
      {alertLog.slice().reverse().map((entry, i) => {
        const cfg = ALERT_LEVELS[entry.level] || ALERT_LEVELS.green;
        return (
          <div key={i} style={{
            display: "flex", alignItems: "flex-start", gap: 10,
            padding: "8px 0",
            borderBottom: "1px solid #f0f0f0",
          }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: cfg.color, flexShrink: 0, marginTop: 4,
            }} />
            <div>
              <span style={{ fontSize: 12, fontWeight: 600, color: cfg.color }}>
                {cfg.label}
              </span>
              <span style={{ fontSize: 12, color: "#666", marginLeft: 8 }}>
                {entry.time}
              </span>
              <div style={{ fontSize: 12, color: "#555", marginTop: 2 }}>
                {entry.message}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────
export default function PitchSense() {
  const [state,    setState]    = useState(() => generateDemoState(0));
  const [history,  setHistory]  = useState([]);
  const [alertLog, setAlertLog] = useState([]);
  const [running,  setRunning]  = useState(true);
  const [tick,     setTick]     = useState(0);
  const prevAlert = useRef("green");
  const tickRef   = useRef(0);

  // Simulate real-time updates
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      tickRef.current += 1;
      const s = generateDemoState(tickRef.current);
      setState(s);
      setTick(tickRef.current);
      setHistory(h => {
        const next = [...h, s].slice(-WAVEFORM_POINTS);
        return next;
      });

      // Log state transitions
      if (s.alert !== prevAlert.current) {
        const now = new Date().toLocaleTimeString();
        const msg =
          s.alert === "amber" ? `Elevated crowd pressure detected in sector ${
            Object.entries(s.densities).sort((a,b) => b[1]-a[1])[0][0]
          }. Score: ${Math.round(s.fusedScore*100)}%.` :
          s.alert === "red"   ? `CRITICAL threshold reached. Score: ${Math.round(s.fusedScore*100)}%. Opening emergency exits.` :
                                `Conditions normalised. Score: ${Math.round(s.fusedScore*100)}%.`;
        setAlertLog(l => [...l, { level: s.alert, time: now, message: msg }]);
        prevAlert.current = s.alert;
      }
    }, 800);
    return () => clearInterval(id);
  }, [running]);

  const card = (children, style = {}) => (
    <div style={{
      background: "#fff",
      borderRadius: 12,
      border: "1px solid #e8e8e8",
      padding: "16px 18px",
      boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
      ...style,
    }}>
      {children}
    </div>
  );

  const cardTitle = (text) => (
    <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em",
                  textTransform: "uppercase", color: "#999",
                  marginBottom: 12 }}>
      {text}
    </div>
  );

  return (
    <div style={{
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      background: "#f4f5f7",
      minHeight: "100vh",
      padding: "20px 20px 40px",
      color: "#1a1a2e",
    }}>
      <style>{`
        @keyframes pulse {
          0%,100% { opacity:1; transform:scale(1); }
          50% { opacity:0.5; transform:scale(1.3); }
        }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: 20, display: "flex",
                    justifyContent: "space-between", alignItems: "flex-start",
                    flexWrap: "wrap", gap: 12 }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: "0.1em",
                        textTransform: "uppercase", color: "#0F6E56",
                        marginBottom: 4, fontWeight: 700 }}>
            PROJECT NEXUS · MODULE 04
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0, color: "#0A1628" }}>
            PitchSense — Crowd Intelligence
          </h1>
          <div style={{ fontSize: 13, color: "#888", marginTop: 4 }}>
            Stadium: Wembley · Capacity: {state.capacity.toLocaleString()} ·
            Present: <strong>{state.present.toLocaleString()}</strong> ·
            Occupancy: <strong>{Math.round(state.present / state.capacity * 100)}%</strong>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <div style={{ fontSize: 12, color: "#888" }}>
            Tick #{tick}
          </div>
          <button
            onClick={() => setRunning(r => !r)}
            style={{
              padding: "7px 16px",
              background: running ? "#0F6E56" : "#888",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
            }}>
            {running ? "⏸ Pause" : "▶ Resume"}
          </button>
        </div>
      </div>

      {/* Alert Banner */}
      <AlertBanner level={state.alert} />

      {/* Main grid */}
      <div style={{ display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                    gap: 16 }}>

        {/* Heatmap */}
        {card(
          <>
            {cardTitle("Stadium Density Heatmap")}
            <StadiumHeatmap densities={state.densities} />
            <div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
              {[["rgba(15,110,86,0.5)","< 55%"],
                ["rgba(186,117,23,0.7)","55–85%"],
                ["rgba(192,57,43,0.85)","> 85%"]].map(([col, lbl]) => (
                <div key={lbl} style={{ display: "flex", alignItems: "center",
                                        gap: 5, fontSize: 11, color: "#666" }}>
                  <div style={{ width: 12, height: 12, borderRadius: 2,
                                background: col }} />
                  {lbl}
                </div>
              ))}
            </div>
          </>
        )}

        {/* Score gauges */}
        {card(
          <>
            {cardTitle("Dual-Signal Alert Score")}
            <div style={{ display: "flex", justifyContent: "space-around",
                          flexWrap: "wrap", gap: 8 }}>
              <ScoreGauge score={state.visualScore}
                          label="Visual Density"
                          color="#0F6E56" />
              <ScoreGauge score={state.audioScore}
                          label="Acoustic Signal"
                          color="#BA7517" />
              <ScoreGauge score={state.fusedScore}
                          label="Fused Score"
                          color={state.alert === "red" ? "#C0392B" :
                                 state.alert === "amber" ? "#BA7517" : "#0F6E56"} />
            </div>
            <div style={{ marginTop: 14, fontSize: 12, color: "#666",
                          lineHeight: 1.6, background: "#f9f9f9",
                          borderRadius: 8, padding: "10px 12px" }}>
              <strong>Fusion formula:</strong> Score = 0.6 × Visual + 0.4 × Acoustic
              <br />
              Thresholds: Amber ≥ 40% · Red ≥ 70%
            </div>
          </>
        )}

        {/* Audio waveform */}
        {card(
          <>
            {cardTitle("Acoustic Sentiment — Live (60s)")}
            <AudioWaveform history={history} audioScores={state.audioScores} />
          </>
        )}

        {/* Gate status */}
        {card(
          <>
            {cardTitle("Exit Gate Status")}
            <GatePanel gateStatus={state.gateStatus} />
          </>
        )}

        {/* Alert log */}
        {card(
          <>
            {cardTitle(`Alert Log (${alertLog.length} events)`)}
            <AlertLog log={alertLog} />
          </>,
          { gridColumn: "1 / -1" }
        )}

      </div>

      <div style={{ textAlign: "center", marginTop: 24, fontSize: 11, color: "#bbb" }}>
        Project Nexus · PitchSense v1.0 · github.com/Unwilling-mcu/ProjectNexus
      </div>
    </div>
  );
}
