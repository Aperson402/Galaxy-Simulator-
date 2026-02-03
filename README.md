# Metal Galaxy Simulator

A real-time galaxy simulation that uses Metal compute and render pipelines to generate and animate a million-star galaxy with multiple morphologies. Stars are rendered as tiny additive-blended triangles and updated on the GPU via a single compute kernel that handles gravity, dark-matter-like halo acceleration, flat rotation curve stabilization, light clumping, and gentle stochastic jitter to keep structures lively.

## Features
- Multiple galaxy morphologies generated procedurally in Swift:
  - Grand spiral, flocculent, elliptical, ring (starburst), barred spiral, lenticular, irregular dwarf, merging pair, fractal spiral, polar ring, vortex lens, butterfly, lopsided arc.
- GPU compute kernel (`shader.metal`) integrates velocities and positions, applies stabilizing forces, and writes final vertices in one pass.
- Additive blending for star glow.
- Deterministic, data-parallel update that scales to large star counts.

## Screenshots
<img width="425" height="224" alt="Screenshot 2026-02-03 at 9 36 50 AM" src="https://github.com/user-attachments/assets/64c4f47a-68a4-459d-833b-a570b3c4d3f6" />
<img width="425" height="224" alt="Screenshot 2026-02-03 at 9 37 38 AM" src="https://github.com/user-attachments/assets/68fab41c-46d5-4fcb-a378-b68fc25e7fd0" />
<img width="425" height="224" alt="Screenshot 2026-02-03 at 9 37 59 AM" src="https://github.com/user-attachments/assets/d58c6c68-4a88-454d-9dc5-aa05ea237a30" />
<img width="572" height="610" alt="Screenshot 2026-02-03 at 9 38 39 AM" src="https://github.com/user-attachments/assets/f925418f-6c5f-4806-a709-51f1a5658887" />

## Requirements
- macOS with a Metal-capable GPU
- Xcode 15+ (tested with Xcode 26.1.1)
- Swift 5.9+

## Build & Run
1. Open the project in Xcode.
2. Select the macOS target.
3. Run (⌘R).
4. On launch, a random galaxy type is generated. Re-run to see other morphologies.

If you encounter performance issues on your machine, lower the star count (see Performance section below).

## Controls (optional)
This project currently regenerates a random galaxy on launch. If you want quick iteration:
- Add a key handler to call `resetSimulation(view:)` for in-app regeneration.
- Expose a runtime toggle for the star count (e.g., 100k / 300k / 1M) to accommodate a range of GPUs.

## How it works
- Engine setup (`Engine.swift`):
  - Initializes Metal device, command queue, render pipeline (`vertex_main`, `fragment_main`) and compute pipeline (`gravity_kernel`).
  - Creates buffers for triangles (star geometry), velocities, masses, and a large vertex buffer.
  - Procedurally generates initial positions/velocities/masses per selected galaxy type.
- Compute pass (`shader.metal`):
  - For each star (triangle instance), computes accelerations from a compact central mass and a softened halo term to approximate flat rotation curves.
  - Applies a tangential speed correction with a deadzone to stabilize disk rotation, mild radial damping, and an epicyclic spring toward a star-specific preferred radius.
  - Samples a few pseudo-neighbors to add subtle clumping.
  - Integrates velocity/position (symplectic Euler), adds a tiny jitter to avoid phase locking, and writes three output vertices per star directly into the render vertex buffer.
- Render pass:
  - Draws all triangles with additive blending for a soft glow effect.

## Performance
- Default star count in `Engine.swift` is set high (e.g., 1,000,000). This pushes the GPU; performance will vary by machine.
- To make the project accessible:
  - Reduce `starCount` in `Engine.swift` (e.g., 100_000 or 300_000) if your GPU struggles.
  - Consider choosing `threadsPerThreadgroup` aligned to the device `threadExecutionWidth` for better occupancy.
  - Keep an eye on buffer sizes: the render vertex buffer stores 3 vertices per star.

## Customization ideas
- Deterministic seeds per galaxy type for reproducible visuals.
- Keyboard shortcuts: reset/regenerate, cycle galaxy types, toggle star count.
- Simple FPS overlay for performance comparisons.
- Visual polish: bloom-like pass, star size falloff by mass/radius, color grading for bulge/disk.

## File overview
- `Engine.swift`: Metal setup, buffer creation, galaxy generation, and per-frame dispatch of compute+render passes.
- `shader.metal`: Vertex/fragment shaders and the `gravity_kernel` compute shader that updates star positions and writes vertices.
- `AppDelegate.swift`: Creates a window, sets up a `GalaxyView` (custom `MTKView`), wires up keyboard/mouse input (Space or click to reset), and instantiates `Engine`.
- `main.swift`: Minimal Cocoa entry point. Creates the `NSApplication`, assigns `AppDelegate`, sets activation policy, and starts the run loop.

## Contributing
Contributions are welcome! Good first issues:
- Add runtime star count toggles.
- Implement a reset hotkey to regenerate galaxies without relaunching.
- Add deterministic seeds for reproducibility.
- Add an FPS overlay or simple performance HUD.
- Help make simulation more accurate
