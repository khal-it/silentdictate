# Webcam VSR plugin (stub)

Disabled by default and not implemented. `VsrPlugin` defines the interface; `WebcamVsrStub` raises `NotImplementedError`, so it never opens the camera.

The planned implementation is a later milestone: MediaPipe mouth region, then Auto-AVSR visual-only recognition, then LLM cleanup. It must show a camera-active indicator and save no video outside enrollment. It is installed with the `silentdictate[vsr]` extra, which is empty for now.
