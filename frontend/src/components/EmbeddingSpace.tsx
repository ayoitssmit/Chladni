"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

interface EmbeddingPointsProps {
  embeddings: number[][];
}

function EmbeddingPoints({ embeddings }: EmbeddingPointsProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Target positions from PCA
  const targets = useMemo(() => {
    if (!embeddings || embeddings.length === 0) return [];
    return embeddings.map((coords) => new THREE.Vector3(...coords));
  }, [embeddings]);

  // Current positions for smooth interpolation
  const currentPositions = useRef<THREE.Vector3[]>(
    Array.from({ length: 97 }, () => new THREE.Vector3(0, 0, 0))
  );

  // Color each token by its index, creating a rainbow around the 97 tokens
  const colors = useMemo(() => {
    const arr = new Float32Array(97 * 3);
    for (let i = 0; i < 97; i++) {
      const hue = i / 97;
      const color = new THREE.Color().setHSL(hue, 0.7, 0.6);
      arr[i * 3] = color.r;
      arr[i * 3 + 1] = color.g;
      arr[i * 3 + 2] = color.b;
    }
    return arr;
  }, []);

  useFrame(() => {
    if (!meshRef.current || targets.length === 0) return;

    for (let i = 0; i < 97; i++) {
      if (!targets[i]) continue;

      // Smooth lerp toward target position
      currentPositions.current[i].lerp(targets[i], 0.08);

      dummy.position.copy(currentPositions.current[i]);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, 97]}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshStandardMaterial vertexColors />
      <instancedBufferAttribute
        attach="geometry-attributes-color"
        args={[colors, 3]}
      />
    </instancedMesh>
  );
}

function SceneContent({ embeddings }: { embeddings: number[][] }) {
  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />

      {embeddings.length > 0 && <EmbeddingPoints embeddings={embeddings} />}

      {/* Subtle axis lines */}
      <group>
        {[
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ].map((dir, i) => (
          <line key={i}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                args={[
                  new Float32Array([
                    -dir[0] * 3,
                    -dir[1] * 3,
                    -dir[2] * 3,
                    dir[0] * 3,
                    dir[1] * 3,
                    dir[2] * 3,
                  ]),
                  3,
                ]}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#333" transparent opacity={0.3} />
          </line>
        ))}
      </group>

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        autoRotate
        autoRotateSpeed={0.5}
        minDistance={2}
        maxDistance={15}
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
    <div className="glass-panel p-4 flex flex-col h-full">
      {/* Header */}
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
            97 tokens projected via PCA into 3D — drag to rotate
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

      {/* Canvas */}
      <div className="flex-1 min-h-0 three-canvas-container rounded-lg overflow-hidden"
        style={{ background: "var(--surface-900)" }}>
        {embeddings.length === 0 ? (
          <div
            className="h-full flex items-center justify-center text-sm"
            style={{ color: "var(--text-muted)" }}
          >
            Start a simulation to see the embedding geometry.
          </div>
        ) : (
          <Canvas
            camera={{ position: [4, 3, 4], fov: 50 }}
            gl={{ antialias: true }}
          >
            <SceneContent embeddings={embeddings} />
          </Canvas>
        )}
      </div>
    </div>
  );
}
