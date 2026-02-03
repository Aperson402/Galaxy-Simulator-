import MetalKit
import simd

// --- Data Structures ---
struct Vertex {
    var position: SIMD2<Float>
    var color: SIMD3<Float>
}

struct Triangle {
    var p0, p1, p2: SIMD2<Float>
    var color: SIMD3<Float>
    var size: Float
}

enum GalaxyType: CaseIterable {
    case grandSpiral    // Classic 2-arm spiral
    case flocculent     // Messy, many-armed spiral
    case elliptical     // Egg-shaped swarm of old stars
    case starburstRing  // Rare ring of intense star formation
    case barredSpiral   // Central bar with trailing arms
    case lenticular     // S0: bright disk, weak arms
    case irregularDwarf // Clumpy, chaotic structure
    case mergingPair    // Two cores with tidal tails
    case fractalSpiral  // Multi-scale spiral noise arms
    case polarRing      // Orthogonal inner disk with tilted ring
    case vortexLens     // Log-spiral lensing toward a focal point
    case butterfly      // Mirrored twin-lobe with waist
    case lopsidedArc    // Single sweeping arc with debris
}


final class Engine: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let renderPSO: MTLRenderPipelineState
    private let computePSO: MTLComputePipelineState
    
    private var triangleBuffer: MTLBuffer?
    private var velocityBuffer: MTLBuffer?
    private var massBuffer: MTLBuffer?
    private var vertexBuffer: MTLBuffer?
    
    private let starCount: Int = 1000000
    private var totalCount: Int = 2
    private var lastTime: CFTimeInterval = CACurrentMediaTime()
    
    init(view: MTKView) {
        self.device = view.device!
        self.commandQueue = device.makeCommandQueue()!
        let library = device.makeDefaultLibrary()!
        
        let pipelineDesc = MTLRenderPipelineDescriptor()
        pipelineDesc.vertexFunction = library.makeFunction(name: "vertex_main")
        pipelineDesc.fragmentFunction = library.makeFunction(name: "fragment_main")
        pipelineDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        
        if let attachment = pipelineDesc.colorAttachments[0] {
            attachment.isBlendingEnabled = true
            attachment.rgbBlendOperation = .add
            attachment.alphaBlendOperation = .add
            attachment.sourceRGBBlendFactor = .one
            attachment.destinationRGBBlendFactor = .one
        }
        
        self.renderPSO = try! device.makeRenderPipelineState(descriptor: pipelineDesc)
        self.computePSO = try! device.makeComputePipelineState(function: library.makeFunction(name: "gravity_kernel")!)
        
        super.init()
        
        self.resetSimulation(view: view)
        
        view.delegate = self
        view.clearColor = MTLClearColor(red: 0.005, green: 0.005, blue: 0.01, alpha: 1.0)
    }
    
    func resetSimulation(view: MTKView) {
        let type = GalaxyType.allCases.randomElement()!
        var tris: [Triangle] = []
        var vels: [SIMD2<Float>] = []
        var masses: [Float] = []
        
        for _ in 0..<300 {
            let r = Float.random(in: 0...0.04)
            let angle = Float.random(in: 0...(.pi * 2))
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            tris.append(createTriangle(at: pos, color: SIMD3(1.5, 1.3, 0.9), size: 0.01))
            vels.append(.zero)
            masses.append(0.0)
        }
        
        for _ in 0..<20000 {
            let isUp = Bool.random()
            let pos = SIMD2<Float>(Float.random(in: -0.01...0.01), Float.random(in: -0.01...0.01))
            tris.append(createTriangle(at: pos, color: SIMD3(0.4, 0.6, 2.0), size: 0.008))
            vels.append(SIMD2<Float>(0, isUp ? 2.5 : -2.5))
            masses.append(0.0)
        }
        

        switch type {
        case .grandSpiral:
            generateSpiral(arms: 2, tightness: 4.5, tris: &tris, vels: &vels, masses: &masses)
        case .flocculent:
            generateSpiral(arms: 7, tightness: 2.0, tris: &tris, vels: &vels, masses: &masses)
        case .elliptical:
            generateElliptical(tris: &tris, vels: &vels, masses: &masses)
        case .starburstRing:
            generateRing(tris: &tris, vels: &vels, masses: &masses)
        case .barredSpiral:
            generateBarredSpiral(tris: &tris, vels: &vels, masses: &masses)
        case .lenticular:
            generateLenticular(tris: &tris, vels: &vels, masses: &masses)
        case .irregularDwarf:
            generateIrregularDwarf(tris: &tris, vels: &vels, masses: &masses)
        case .mergingPair:
            generateMergingPair(tris: &tris, vels: &vels, masses: &masses)
        case .fractalSpiral:
            generateFractalSpiral(tris: &tris, vels: &vels, masses: &masses)
        case .polarRing:
            generatePolarRing(tris: &tris, vels: &vels, masses: &masses)
        case .vortexLens:
            generateVortexLens(tris: &tris, vels: &vels, masses: &masses)
        case .butterfly:
            generateButterfly(tris: &tris, vels: &vels, masses: &masses)
        case .lopsidedArc:
            generateLopsidedArc(tris: &tris, vels: &vels, masses: &masses)
        }
        
        self.totalCount = tris.count
        

        triangleBuffer = device.makeBuffer(bytes: tris, length: MemoryLayout<Triangle>.stride * totalCount, options: .storageModeShared)
        velocityBuffer = device.makeBuffer(bytes: vels, length: MemoryLayout<SIMD2<Float>>.stride * totalCount, options: .storageModeShared)
        massBuffer = device.makeBuffer(bytes: masses, length: MemoryLayout<Float>.stride * totalCount, options: .storageModeShared)
        vertexBuffer = device.makeBuffer(length: MemoryLayout<Vertex>.stride * totalCount * 3, options: .storageModeShared)
    }
    

    private func createTriangle(at pos: SIMD2<Float>, color: SIMD3<Float>, size: Float) -> Triangle {
        return Triangle(p0: pos + SIMD2(0, size),
                        p1: pos + SIMD2(-size * 0.86, -size * 0.5),
                        p2: pos + SIMD2(size * 0.86, -size * 0.5),
                        color: color, size: size)
    }
    
    private func getRandomStarColor() -> SIMD3<Float> {
        let rand = Float.random(in: 0...1)
        
        if rand < 0.45 {
            return SIMD3<Float>(0.5, 0.7, 1.0)
        } else if rand < 0.90 {
            return SIMD3<Float>(1.0, 0.9, 0.5)
        } else {
            return SIMD3<Float>(1.0, 0.3, 0.2)
        }
    }
    
    private func generateSpiral(arms: Int, tightness: Float, tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        for i in 0..<starCount {
            let arm = Float(i % arms) * ((2 * .pi) / Float(arms))
            let r = sqrt(Float.random(in: 0...1)) * 1.3 + 0.05
            let angle = arm + (r * tightness) + Float.random(in: -0.15...0.15)
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            
            let speed = sqrt(3.5 / (r + 0.05)) * 1.05
            let vel = SIMD2<Float>(-sin(angle), cos(angle)) * speed
            
            let starColor = getRandomStarColor()
            
            tris.append(createTriangle(at: pos, color: starColor, size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.5...2.0))
        }
    }
    
    private func generateElliptical(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        for _ in 0..<starCount {
            let r = pow(Float.random(in: 0...1), 0.7) * 0.9
            let angle = Float.random(in: 0...(.pi * 2))
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            
            // Random orbit directions for that "beehive" look
            let speed = sqrt(3.5 / (r + 0.1)) * Float.random(in: 0.7...1.1)
            let orbitAngle = Float.random(in: 0...(.pi * 2))
            let vel = SIMD2<Float>(cos(orbitAngle), sin(orbitAngle)) * speed
            
            tris.append(createTriangle(at: pos, color: SIMD3(1.0, 0.6, 0.3), size: 0.007))
            vels.append(vel)
            masses.append(1.0)
        }
    }
    
    private func generateRing(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        for _ in 0..<starCount {
            let r = Float.random(in: 0.7...0.9) // Tight band
            let angle = Float.random(in: 0...(.pi * 2))
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            let speed = sqrt(3.5 / r) * 1.1
            
            tris.append(createTriangle(at: pos, color: SIMD3(0.2, 0.8, 1.0), size: 0.006))
            vels.append(SIMD2<Float>(-sin(angle), cos(angle)) * speed)
            masses.append(1.2)
        }
    }

    private func generateBarredSpiral(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        // Central bar
        let barLen: Float = 0.5
        let barStars = max(600, starCount / 20)
        for i in 0..<barStars {
            let t = (Float(i) / Float(barStars - 1) - 0.5) * 2.0
            let pos = SIMD2<Float>(t * barLen, Float.random(in: -0.04...0.04))
            let angle: Float = 0
            let r = max(0.12, abs(pos.x))
            let speed = sqrt(3.5 / (r + 0.05)) * 0.9
            let vel = SIMD2<Float>(-sin(angle), cos(angle)) * speed
            tris.append(createTriangle(at: pos, color: SIMD3(1.0, 0.85, 0.6), size: 0.007))
            vels.append(vel)
            masses.append(1.0)
        }
        // Trailing arms from bar ends
        let arms = 2
        let tightness: Float = 3.8
        let armStars = max(0, starCount - barStars)
        for i in 0..<armStars {
            let which = i % arms
            let baseAngle: Float = which == 0 ? 0.0 : .pi
            let r = sqrt(Float.random(in: 0...1)) * 1.4 + 0.2
            let angle = baseAngle + (r * tightness) + Float.random(in: -0.12...0.12)
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            let speed = sqrt(3.5 / (r + 0.05)) * 1.0
            let vel = SIMD2<Float>(-sin(angle), cos(angle)) * speed
            let color = Bool.random() ? SIMD3<Float>(0.9, 0.8, 0.5) : SIMD3<Float>(0.55, 0.75, 1.0)
            tris.append(createTriangle(at: pos, color: color, size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.7...1.4))
        }
    }

    private func generateLenticular(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        // Bright disk with weak arms: concentrate stars in a smooth disk
        for _ in 0..<starCount {
            let r = pow(Float.random(in: 0...1), 0.4) * 1.2 + 0.05
            let angle = Float.random(in: 0...(.pi * 2))
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            let speed = sqrt(3.5 / (r + 0.05)) * Float.random(in: 0.95...1.05)
            let vel = SIMD2<Float>(-sin(angle), cos(angle)) * speed
            let color = SIMD3<Float>(1.0, 0.85, 0.6)
            tris.append(createTriangle(at: pos, color: color, size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.8...1.2))
        }
        for _ in 0..<(starCount / 8) {
            let r = Float.random(in: 0.9...1.1)
            let angle = Float.random(in: 0...(.pi * 2))
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            let speed = sqrt(3.5 / (r + 0.05))
            let vel = SIMD2<Float>(-sin(angle), cos(angle)) * speed
            tris.append(createTriangle(at: pos, color: SIMD3(0.9, 0.9, 0.9), size: 0.005))
            vels.append(vel)
            masses.append(1.0)
        }
    }

    private func generateIrregularDwarf(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {

        let clumps = 6
        var centers: [SIMD2<Float>] = []
        for _ in 0..<clumps {
            let r = Float.random(in: 0.0...0.7)
            let a = Float.random(in: 0...(.pi * 2))
            centers.append(SIMD2<Float>(cos(a) * r, sin(a) * r))
        }
        for _ in 0..<starCount {
            let c = centers.randomElement()!
            let offset = SIMD2<Float>(Float.random(in: -0.12...0.12), Float.random(in: -0.12...0.12))
            let pos = c + offset
            let r = max(0.08, length(pos))
            let baseSpeed = sqrt(3.5 / (r + 0.08))
            let velDir = normalize(SIMD2<Float>(-pos.y, pos.x) + SIMD2<Float>(Float.random(in: -0.3...0.3), Float.random(in: -0.3...0.3)))
            let vel = velDir * baseSpeed * Float.random(in: 0.6...1.2)
            let color = Bool.random() ? SIMD3<Float>(0.6, 0.8, 1.0) : SIMD3<Float>(1.0, 0.7, 0.5)
            tris.append(createTriangle(at: pos, color: color, size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.5...1.5))
        }
    }

    private func generateMergingPair(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        // Two cores on a collision course with tidal tails
        let sep: Float = 0.6
        let centers = [SIMD2<Float>(-sep, 0), SIMD2<Float>(sep, 0)]
        let coreStars = max(600, starCount / 10)
        let tailStars = max(0, starCount - coreStars * 2)

        // Cores
        for (idx, c) in centers.enumerated() {
            for _ in 0..<coreStars {
                let r = Float.random(in: 0.0...0.15)
                let a = Float.random(in: 0...(.pi * 2))
                let pos = c + SIMD2<Float>(cos(a) * r, sin(a) * r)
                let speed = sqrt(3.5 / (max(0.08, r) + 0.05)) * 0.8
                let dir = normalize(SIMD2<Float>(-pos.y, pos.x))
                let vel = dir * speed + SIMD2<Float>(idx == 0 ? 0.25 : -0.25, 0) // approach
                tris.append(createTriangle(at: pos, color: SIMD3(1.2, 1.0, 0.8), size: 0.007))
                vels.append(vel)
                masses.append(1.2)
            }
        }
        // Tidal tails
        for i in 0..<tailStars {
            let which = i % 2
            let base = centers[which]
            let r = Float.random(in: 0.2...1.6)
            let a = Float.random(in: 0...(.pi * 2))
            let pos = base + SIMD2<Float>(cos(a) * r, sin(a) * r)
            let speed = sqrt(3.5 / (r + 0.1)) * Float.random(in: 0.7...1.1)
            let vel = SIMD2<Float>(-sin(a), cos(a)) * speed + SIMD2<Float>(which == 0 ? 0.25 : -0.25, 0)
            let color = SIMD3<Float>(0.8, 0.9, 1.0)
            tris.append(createTriangle(at: pos, color: color, size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.6...1.3))
        }
    }

    private func generateFractalSpiral(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        // Multi-scale arms: outer log-spiral perturbed by nested noise harmonics
        let arms = 3
        let baseTight: Float = 3.2
        for i in 0..<starCount {
            let arm = Float(i % arms) * ((2 * .pi) / Float(arms))
            let r = sqrt(Float.random(in: 0...1)) * 1.5 + 0.08
            var angle = arm + r * baseTight
            // Fractal-ish perturbations
            angle += sin(r * 6.0 + Float(i % 7)) * 0.05
            angle += sin(r * 12.0 + Float(i % 11)) * 0.03
            angle += sin(r * 24.0 + Float(i % 13)) * 0.015
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            let speed = sqrt(3.5 / (r + 0.05)) * Float.random(in: 0.95...1.1)
            let vel = SIMD2<Float>(-sin(angle), cos(angle)) * speed
            let color = (i % 5 == 0) ? SIMD3<Float>(0.55, 0.8, 1.0) : getRandomStarColor()
            tris.append(createTriangle(at: pos, color: color, size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.6...1.4))
        }
    }

    private func generatePolarRing(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        // Inner disk
        let innerCount = starCount / 3
        for _ in 0..<innerCount {
            let r = pow(Float.random(in: 0...1), 0.6) * 0.6 + 0.05
            let angle = Float.random(in: 0...(.pi * 2))
            let pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            let speed = sqrt(3.5 / (r + 0.05))
            let vel = SIMD2<Float>(-sin(angle), cos(angle)) * speed
            tris.append(createTriangle(at: pos, color: SIMD3(1.0, 0.9, 0.7), size: 0.006))
            vels.append(vel)
            masses.append(1.0)
        }
        // Tilted polar ring
        let ringCount = starCount - innerCount
        let tilt: Float = .pi / 2 * 0.85 // ~77 deg
        let rot = float2x2(SIMD2<Float>(cos(tilt), -sin(tilt)), SIMD2<Float>(sin(tilt), cos(tilt)))
        for _ in 0..<ringCount {
            let r = Float.random(in: 0.9...1.3)
            let a = Float.random(in: 0...(.pi * 2))
            var p = SIMD2<Float>(cos(a) * r, sin(a) * r)
            p = rot * p
            let speed = sqrt(3.5 / (r + 0.05)) * 0.95
            var v = SIMD2<Float>(-sin(a), cos(a)) * speed
            v = rot * v
            tris.append(createTriangle(at: p, color: SIMD3(0.7, 0.85, 1.0), size: 0.006))
            vels.append(v)
            masses.append(1.0)
        }
    }

    private func generateVortexLens(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        let focus = SIMD2<Float>(0.35, 0.2)
        for i in 0..<starCount {
            let r = sqrt(Float.random(in: 0...1)) * 1.4 + 0.1
            var angle = r * 4.5 + Float.random(in: -0.2...0.2)
            var pos = SIMD2<Float>(cos(angle) * r, sin(angle) * r)
            let toward = normalize(focus - pos)
            pos += toward * min(0.25, 0.12 * r)
            let dir = normalize(SIMD2<Float>(-pos.y, pos.x))
            let speed = sqrt(3.5 / (length(pos) + 0.05))
            let vel = dir * speed
            let color = (i % 9 == 0) ? SIMD3<Float>(1.0, 0.6, 0.9) : getRandomStarColor()
            tris.append(createTriangle(at: pos, color: color, size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.7...1.3))
        }
    }

    private func generateButterfly(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        let lobeCount = starCount / 2
        let waist: Float = 0.12
        for side in [-1, 1] {
            for _ in 0..<lobeCount {
                let r = pow(Float.random(in: 0...1), 0.6) * 0.9 + 0.1
                let a = Float.random(in: -0.9...0.9)
                let pos = SIMD2<Float>(Float(side) * (waist + r * 0.6), a * r)
                let speed = sqrt(3.5 / (length(pos) + 0.08)) * 0.9
                let dir = normalize(SIMD2<Float>(-pos.y, pos.x))
                let vel = dir * speed
                let color = side < 0 ? SIMD3<Float>(0.8, 0.9, 1.0) : SIMD3<Float>(1.0, 0.7, 0.6)
                tris.append(createTriangle(at: pos, color: color, size: 0.006))
                vels.append(vel)
                masses.append(1.0)
            }
        }
    }

    private func generateLopsidedArc(tris: inout [Triangle], vels: inout [SIMD2<Float>], masses: inout [Float]) {
        let mainCount = Int(Float(starCount) * 0.7)
        for i in 0..<mainCount {
            let r = Float.random(in: 0.6...1.4)
            let a = Float.random(in: 0.1...2.7) 
            let pos = SIMD2<Float>(cos(a) * r, sin(a) * r)
            let speed = sqrt(3.5 / (r + 0.05)) * 1.0
            let vel = SIMD2<Float>(-sin(a), cos(a)) * speed
            let color = (i % 7 == 0) ? SIMD3<Float>(1.0, 0.8, 0.5) : getRandomStarColor()
            tris.append(createTriangle(at: pos, color: color, size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.7...1.3))
        }
        // Debris and counter-tail
        for _ in 0..<(starCount - mainCount) {
            let r = Float.random(in: 0.2...1.8)
            let a = Float.random(in: -0.3...(.pi + 0.3))
            let pos = SIMD2<Float>(cos(a) * r, sin(a) * r) + SIMD2<Float>(Float.random(in: -0.15...0.15), Float.random(in: -0.15...0.15))
            let speed = sqrt(3.5 / (max(0.12, r))) * Float.random(in: 0.7...1.1)
            let vel = normalize(SIMD2<Float>(-pos.y, pos.x)) * speed
            tris.append(createTriangle(at: pos, color: getRandomStarColor(), size: 0.006))
            vels.append(vel)
            masses.append(Float.random(in: 0.6...1.4))
        }
    }

    func draw(in view: MTKView) {
        guard let cb = commandQueue.makeCommandBuffer(),
              let rpd = view.currentRenderPassDescriptor,
              let tBuf = triangleBuffer, let vBuf = velocityBuffer,
              let mBuf = massBuffer, let verBuf = vertexBuffer else { return }
        
        var dt = Float(CACurrentMediaTime() - lastTime)
        lastTime = CACurrentMediaTime()
        dt = min(dt, 0.02)

        let ce = cb.makeComputeCommandEncoder()!
        ce.setComputePipelineState(computePSO)
        ce.setBuffers([tBuf, vBuf, mBuf, verBuf], offsets: [0,0,0,0], range: 0..<4)
        var c = UInt32(totalCount)
        ce.setBytes(&c, length: 4, index: 4)
        ce.setBytes(&dt, length: 4, index: 5)
        ce.dispatchThreads(MTLSize(width: totalCount, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: computePSO.maxTotalThreadsPerThreadgroup, height: 1, depth: 1))
        ce.endEncoding()

        let re = cb.makeRenderCommandEncoder(descriptor: rpd)!
        re.setRenderPipelineState(renderPSO)
        re.setVertexBuffer(verBuf, offset: 0, index: 0)
        re.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: totalCount * 3)
        re.endEncoding()

        cb.present(view.currentDrawable!)
        cb.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}

