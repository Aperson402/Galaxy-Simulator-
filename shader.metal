#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float3 color;
};

struct Triangle {
    float2 p0, p1, p2;
    float3 color;
    float size;
};

struct Varying {
    float4 position [[position]];
    float3 color;
    float twinkle;
};

vertex Varying vertex_main(const device Vertex* vertices [[buffer(0)]],
                            uint id [[vertex_id]]) {
    Varying out;
    float zoom = 0.08 ;
    float2 pos = vertices[id].position;
    out.position = float4(pos * zoom, 0, 1);
    out.color = vertices[id].color;
    out.twinkle = 1.0;
    return out;
}

fragment float4 fragment_main(Varying in [[stage_in]]) {
    float2 screenPos = in.position.xy / float2(800.0, 800.0);
    float dist = length(screenPos - float2(0.5, 0.5));
    
    if (dist < 0.005) {
        return float4(0, 0, 0, 1);
    }
    
    return float4(in.color * in.twinkle, 1.0);
    
    
}


kernel void gravity_kernel(device Triangle* triangles [[buffer(0)]],
                            device float2* velocities [[buffer(1)]],
                            device float* masses      [[buffer(2)]],
                            device Vertex* outputVertices [[buffer(3)]],
                            constant uint& count      [[buffer(4)]],
                            constant float& dt        [[buffer(5)]],
                            uint id [[thread_position_in_grid]])
{
    if (id >= count) return;

    // ---------------------------------------------------------------
    // 1. JET PARTICLES — handle and exit immediately, before any
    //    gravity or translation touches them.
    // ---------------------------------------------------------------
    if (triangles[id].color.z > 1.2) {
        float2 vel = velocities[id];
        float2 translation = vel * dt;

        triangles[id].p0 += translation;
        triangles[id].p1 += translation;
        triangles[id].p2 += translation;

        // Reset to center when the jet particle leaves the galaxy
        if (length(triangles[id].p0) > 1.5) {
            // Respawn near origin with a small random-ish offset
            // (deterministic per-id so it looks natural)
            float seed = float(id) * 0.618033;  // golden-ratio hash
            float ox = (fract(seed) - 0.5) * 0.02;
            float oy = (fract(seed * 2.3) - 0.5) * 0.02;
            triangles[id].p0 = float2(ox, oy);
            triangles[id].p1 = float2(ox, oy);
            triangles[id].p2 = float2(ox, oy);
        }

        uint vIdx = id * 3;
        outputVertices[vIdx]   = { triangles[id].p0, triangles[id].color };
        outputVertices[vIdx+1] = { triangles[id].p1, triangles[id].color };
        outputVertices[vIdx+2] = { triangles[id].p2, triangles[id].color };
        return;
    }

    // ---------------------------------------------------------------
    // 2. Compute current position & velocity
    // ---------------------------------------------------------------
    float2 p0  = triangles[id].p0;
    float2 p1  = triangles[id].p1;
    float2 p2  = triangles[id].p2;
    float2 pos = (p0 + p1 + p2) / 3.0;
    float2 vel = velocities[id];

    float2 toCenter = -pos;                        // vector from star to origin
    float distSq    = dot(toCenter, toCenter);
    float dist      = sqrt(distSq);

    // ---------------------------------------------------------------
    // 3. Central gravity  (compact mass at origin)
    //    F = M / (r² + softening)^1.5   (softened point mass)
    // ---------------------------------------------------------------
    float centralMass = 3.5;
    float2 centralAcc = toCenter * (centralMass / pow(distSq + 0.05, 1.5));

    // ---------------------------------------------------------------
    // 4. Dark-matter halo  (produces a flat rotation curve)
    //    A singular isothermal sphere gives v_circ = const, which
    //    means a(r) = v²/r  →  force ∝ 1/r.  I use:
    //      a_halo = strength * toCenter / (dist + coreRadius)
    //    The coreRadius softens the singularity at r = 0.
    // ---------------------------------------------------------------
    float haloStrength = 0.75;
    float haloCore     = 1.7;
    float2 haloAcc     = toCenter * (haloStrength / (dist + haloCore));

    // ---------------------------------------------------------------
    // 5. Flat-rotation-curve enforcement  ← NEW, key stabiliser
    //
    //    In a flat rotation curve the tangential speed is constant:
    //      v_target = v_flat  (independent of r)
    //    We decompose the velocity into radial and tangential parts,
    //    then steer the tangential component toward v_target.
    //
    //    v_flat is tuned so that the target tangential speed matches
    //    the speed the Swift generator gives stars at initialisation
    //    (≈ sqrt(3.5 / r) * 1.05  at small r, flattening outward).
    //    A value around 1.9 works well across the whole disk.
    // ---------------------------------------------------------------
    // Radius-aware target orbital speed with gentle falloff and clamp
    float v_flat_inner = 2.0;    // baseline near center
    float v_flat_outer = v_flat_inner;
    float t = clamp((dist - 0.2) / 1.2, 0.0, 1.0);
    float v_flat = mix(v_flat_inner, v_flat_outer, t);

    // Unit radial vector (pointing outward from center)
    float2 rHat = (dist > 0.001) ? (pos / dist) : float2(1.0, 0.0);
    // Unit tangential vector (90° CCW from radial — the "prograde" direction)
    float2 tHat = float2(-rHat.y, rHat.x);

    float v_radial  = dot(vel, rHat);              // radial component of velocity
    float v_tang    = dot(vel, tHat);              // tangential component

    float v_tang_target = (v_tang >= 0.0) ? v_flat : -v_flat;
    float tang_error    = v_tang_target - v_tang;

    float deadzone = 0.32;
    float absErr = fabs(tang_error);
    float effectiveErr = (absErr > deadzone) ? (tang_error - copysign(deadzone, tang_error)) : 0.0;

    // Radius-dependent gain: weaker in the outer disk
    float baseGain = 0.38;
    float gain = mix(baseGain, baseGain * 0.35, t); // uses t from v_flat calc

    float2 tangCorrAcc = tHat * (effectiveErr * gain);
    // Clamp the corrective acceleration to avoid over-steering
    float tangCorrMax = 0.8;
    float tangCorrMag = length(tangCorrAcc);
    if (tangCorrMag > tangCorrMax) tangCorrAcc *= tangCorrMax / max(tangCorrMag, 1e-5);

    // Softer radial damping
    float radialDampRate = mix(0.10, 0.06, t);
    float2 radialDampAcc = -rHat * (v_radial * radialDampRate);

    // Weak epicyclic restoring force toward a star-specific preferred radius
    // Derive a pseudo-random preferred radius from id to avoid synchrony
    float seed = (float(id) * 0.754877666); // low-discrepancy multiplier
    float preferredR = clamp(0.4 + (fract(seed) - 0.5) * 0.5, 0.15, 1.2);
    float k_epicycle = mix(0.06, 0.02, t); // spring constant reduced with radius
    float2 epicycleAcc = -rHat * ((dist - preferredR) * k_epicycle);

    // ---------------------------------------------------------------
    // Local clumping (lightweight neighbor sampling)
    // Sample a few pseudo-neighbors using a deterministic hash of id.
    // Apply a very small, softened attraction, fading with distance.
    // ---------------------------------------------------------------
    float2 clumpAcc = float2(0.0);
    const uint samples = 6u;             // small fixed sample count
    const float clumpRadius = 0.14;      // only care about local neighborhood
    const float clumpStrength = 0.035;   // very subtle
    const float soften = 0.02;           // softening for stability

    auto hashU = [&](uint x) -> uint {
        x ^= x >> 17; x *= 0xed5ad4bbU; x ^= x >> 11; x *= 0xac4c1b51U; x ^= x >> 15; x *= 0x31848babU; x ^= x >> 14; return x;
    };

    for (uint s = 0u; s < samples; ++s) {
        uint h = hashU(id * 73856093u + s * 19349663u + 83492791u);
        uint j = (h % count);
        if (j == id) continue;
        Triangle ot = triangles[j];
        float2 opos = (ot.p0 + ot.p1 + ot.p2) / 3.0;
        float2 d = opos - pos;
        float r2 = dot(d, d);
        if (r2 < clumpRadius * clumpRadius) {
            float r2s = r2 + soften * soften;
            float invR = rsqrt(r2s);
            float invR3 = invR * invR * invR;
            // Fade with distance so near neighbors matter more
            float w = 1.0 - saturate(sqrt(r2) / clumpRadius);
            clumpAcc += (clumpStrength * w) * d * invR3;
        }
    }

    // ---------------------------------------------------------------
    // 7. Combine all accelerations
    // ---------------------------------------------------------------
    float2 acc = centralAcc + haloAcc + tangCorrAcc + radialDampAcc + epicycleAcc + clumpAcc;

    // ---------------------------------------------------------------
    // 8. Integrate (symplectic Euler — vel first, then pos)
    // ---------------------------------------------------------------
    vel += acc * dt;

    // Tiny stochastic energy injection to avoid phase-locking
    float jitterSeed = fract(sin((float(id) + dist) * 12.9898) * 43758.5453);
    float ang = jitterSeed * 6.2831853;
    float jmag = 0.0035 * dt; // slightly increased to avoid phase-locking
    float2 jitter = float2(cos(ang), sin(ang)) * jmag;
    vel += jitter;

    float2 translation = vel * dt;

    triangles[id].p0 += translation;
    triangles[id].p1 += translation;
    triangles[id].p2 += translation;
    velocities[id]   = vel;

    // ---------------------------------------------------------------
    // 9. Boundary: if a star escapes, pull it back gently
    //    (prevents the galaxy from haemorrhaging mass over time)
    // ---------------------------------------------------------------
    float maxR = 1000;
    if (dist > maxR) {
        // Partial reflection with damping and a small inward nudge
        vel -= 1.4 * dot(vel, rHat) * rHat;
        vel += (-rHat) * 0.15;
        velocities[id] = vel;
    }

    // ---------------------------------------------------------------
    // 10. Write output vertices (single pass, correct colors)
    //     Proximity boost is applied once here.
    // ---------------------------------------------------------------
    float d = length((triangles[id].p0 + triangles[id].p1 + triangles[id].p2) / 3.0);
    float proximityBoost = 1.0 / (d + 0.2);
    float3 finalColor    = triangles[id].color * proximityBoost;

    uint vIdx = id * 3;
    outputVertices[vIdx]   = { triangles[id].p0, finalColor };
    outputVertices[vIdx+1] = { triangles[id].p1, finalColor };
    outputVertices[vIdx+2] = { triangles[id].p2, finalColor };
}

