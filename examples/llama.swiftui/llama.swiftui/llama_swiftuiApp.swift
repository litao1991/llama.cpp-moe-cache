import SwiftUI

@main
struct llama_swiftuiApp: App {
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .onChange(of: scenePhase) { phase in
            if phase == .background {
                NotificationCenter.default.post(name: .llamaEnteredBackground, object: nil)
            }
        }
    }
}

extension Notification.Name {
    static let llamaEnteredBackground = Notification.Name("llamaEnteredBackground")
}
