"use client";

import { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

// Suppress known upstream warning from react-three-fiber about THREE.Clock
if (typeof window !== "undefined") {
  const origWarn = console.warn;
  console.warn = (...args: any[]) => {
    if (args[0] && typeof args[0] === "string" && args[0].includes("THREE.Clock")) return;
    origWarn(...args);
  };
}

interface EmbeddingPointsProps {
  embeddings: number[][];
}

function EmbeddingPoints({ embeddings }: EmbeddingPointsProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Normalize positions so the ring fills the viewport nicely
  const targets = useMemo(() => {
    if (!embeddings || embeddings.length === 0) return [];

    let maxAbs = 0;
    for (const row of embeddings) {
      for (const val of row) {
        const a = Math.abs(val);
        if (a > maxAbs) maxAbs = a;
      }
    }
    const scale = maxAbs > 0.001 ? 4.0 / maxAbs : 1.0;
    return embeddings.map(
      (c) => new THREE.Vector3(c[0] * scale, c[1] * scale, c[2] * scale)
    );
  }, [embeddings]);

  // Smooth interpolation state
  const current = useRef<THREE.Vector3[]>(
    Array.from({ length: 97 }, () => new THREE.Vector3(0, 0, 0))
  );

  // Rainbow colors per token
  const colors = useMemo(() => {
    const arr = new Float32Array(97 * 3);
    for (let i = 0; i < 97; i++) {
      const c = new THREE.Color().setHSL(i / 97, 0.75, 0.55);
      arr[i * 3] = c.r;
      arr[i * 3 + 1] = c.g;
      arr[i * 3 + 2] = c.b;
    }
    return arr;
  }, []);

  const lineGeometryRef = useRef<THREE.BufferGeometry>(null);

  useFrame(() => {
    if (!meshRef.current || targets.length === 0) return;
    for (let i = 0; i < Math.min(97, targets.length); i++) {
      current.current[i].lerp(targets[i], 0.12);
      dummy.position.copy(current.current[i]);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;

    // Update lines connecting the sequence by geometric proximity (perimeter)
    if (lineGeometryRef.current) {
      const positions = lineGeometryRef.current.attributes.position.array as Float32Array;
      
      // Get indices sorted by polar angle to draw a clean perimeter
      const sortedByAngle = current.current
        .map((pos, idx) => ({ idx, angle: Math.atan2(pos.y, pos.x) }))
        .sort((a, b) => a.angle - b.angle);

      for (let i = 0; i < 97; i++) {
        const point = current.current[sortedByAngle[i].idx];
        positions[i * 3] = point.x;
        positions[i * 3 + 1] = point.y;
        positions[i * 3 + 2] = point.z;
      }
      // Close the loop
      const firstPoint = current.current[sortedByAngle[0].idx];
      positions[97 * 3] = firstPoint.x;
      positions[97 * 3 + 1] = firstPoint.y;
      positions[97 * 3 + 2] = firstPoint.z;
      
      lineGeometryRef.current.attributes.position.needsUpdate = true;
    }
  });

  // Pre-allocate line vertices (97 points + 1 to close the loop)
  const linePositions = useMemo(() => new Float32Array(98 * 3), []);

  // Show permanent labels for every 10th token + hovered
  const labelIndices = useMemo(() => {
    const set = new Set<number>();
    for (let i = 0; i < 97; i += 10) set.add(i);
    set.add(96); // last token
    return set;
  }, []);

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, 97]}
        onPointerMove={(e) => setHoveredIndex(e.instanceId ?? null)}
        onPointerOut={() => setHoveredIndex(null)}
      >
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshStandardMaterial vertexColors />
        <instancedBufferAttribute
          attach="geometry-attributes-color"
          args={[colors, 3]}
        />
      </instancedMesh>

      {/* Logic Line connecting 0 -> 1 -> ... -> 96 -> 0 */}
      <line>
        <bufferGeometry ref={lineGeometryRef}>
          <bufferAttribute
            attach="attributes-position"
            args={[linePositions, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color="var(--accent-secondary)"
          transparent
          opacity={0.35}
          linewidth={1.5}
        />
      </line>

      {/* Permanent labels for every 10th token */}
      {targets.length > 0 &&
        Array.from(labelIndices).map((i) =>
          targets[i] ? (
            <Html key={i} position={targets[i]} center>
              <span
                className="pointer-events-none select-none"
                style={{
                  fontSize: 9,
                  fontWeight: 600,
                  color: "rgba(232,198,122,0.7)",
                  textShadow: "0 0 4px rgba(0,0,0,0.8)",
                }}
              >
                {i}
              </span>
            </Html>
          ) : null
        )}

      {/* Hover label */}
      {hoveredIndex !== null &&
        !labelIndices.has(hoveredIndex) &&
        targets[hoveredIndex] && (
          <Html position={targets[hoveredIndex]} center>
            <div
              className="px-1.5 py-0.5 rounded text-[10px] font-bold pointer-events-none whitespace-nowrap"
              style={{
                background: "var(--surface-700)",
                border: "1px solid var(--accent-primary)",
                color: "var(--accent-secondary)",
                transform: "translateY(-14px)",
              }}
            >
              {hoveredIndex}
            </div>
          </Html>
        )}
    </>
  );
}

function SceneContent({ embeddings }: { embeddings: number[][] }) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[8, 8, 8]} intensity={0.8} />
      <pointLight position={[-5, -5, 5]} intensity={0.3} />

      {embeddings.length > 0 && <EmbeddingPoints embeddings={embeddings} />}

      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        autoRotate
        autoRotateSpeed={0.3}
        minDistance={3}
        maxDistance={25}
      />
    </>
  );
}

interface EmbeddingSpaceProps {
  embeddings: number[][];
  grokked: boolean;
}

export default function EmbeddingSpace({
  embeddings,
  grokked,
}: EmbeddingSpaceProps) {
  return (
    <div className="glass-panel p-4 flex flex-col h-full overflow-hidden relative">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3
            className="text-sm font-semibold tracking-wider uppercase"
            style={{ color: "var(--accent-primary)" }}
          >
            Token Embedding Space
          </h3>
          <p
            className="text-xs mt-0.5"
            style={{ color: "var(--text-muted)" }}
          >
            97 tokens projected via PCA — drag to rotate
          </p>
        </div>
        {grokked && (
          <div
            className="text-xs font-mono px-2 py-0.5 rounded"
            style={{
              color: "var(--signal-grok)",
              background: "rgba(110, 224, 94, 0.1)",
            }}
          >
            Ring Formed
          </div>
        )}
      </div>

      <div
        className="flex-1 three-canvas-container rounded-lg overflow-hidden relative"
        style={{ background: "var(--surface-900)", minHeight: 0 }}
      >
        {embeddings.length === 0 ? (
          <div
            className="h-full flex items-center justify-center text-sm"
            style={{ color: "var(--text-muted)" }}
          >
            Start a simulation to see the embedding geometry.
          </div>
        ) : (
          <Canvas
            camera={{ position: [8, 5, 8], fov: 40 }}
            gl={{ antialias: true }}
          >
            <SceneContent embeddings={embeddings} />
          </Canvas>
        )}
      </div>
    </div>
  );
}
