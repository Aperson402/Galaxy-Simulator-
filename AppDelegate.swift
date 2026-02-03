import AppKit
import MetalKit

// 1. Custom View to handle Mouse and Keyboard input
class GalaxyView: MTKView {
    var engine: Engine?

    // Required to allow the view to hear keyboard presses
    override var acceptsFirstResponder: Bool { true }

    // Click to Reset
    override func mouseDown(with event: NSEvent) {
        engine?.resetSimulation(view: self)
    }

    // Spacebar to Reset
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 49 { // 49 is the Virtual Key Code for Space
            engine?.resetSimulation(view: self)
        } else {
            super.keyDown(with: event)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {

    var window: NSWindow!
    var engine: Engine!

    func applicationDidFinishLaunching(_ notification: Notification) {

        let device = MTLCreateSystemDefaultDevice()!

        // 2. Initialize our custom GalaxyView
        let metalView = GalaxyView(
            frame: NSRect(x: 0, y: 0, width: 100, height: 50),
            device: device
        )

        // Set background to near-black so additive stars "pop"
        metalView.clearColor = MTLClearColor(
            red: 0.005, green: 0.005, blue: 0.01, alpha: 1
        )

        window = NSWindow(
            contentRect: metalView.frame,
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )

        window.center()
        window.title = "Cosmos: Procedural Galaxy Generator (Space/Click to Reset)"
        window.contentView = metalView
        window.makeKeyAndOrderFront(nil)

        // 3. Initialize the Engine and link it back to the view
        engine = Engine(view: metalView)
        metalView.engine = engine
        
        // 4. Set focus to the view so it catches the first keyboard press
        window.makeFirstResponder(metalView)
    }
}
