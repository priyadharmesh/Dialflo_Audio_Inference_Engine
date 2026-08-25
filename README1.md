# Dialflo Audio Inference Service

A real-time backend service that infers caller **gender** and **age bracket** from voice audio, specifically engineered for noisy logistics environments (e.g., truck cabs, warehouses, and road conditions).

## Highlights

* **Fast:** ~150-200ms end-to-end latency (SLA target: <500ms).
* **Noise-Tolerant:** 80Hz high-pass filter removes mechanical truck rumble while preserving human voice fundamentals.
* **Privacy-First:** Zero disk writes—all audio is ingested and processed entirely in-memory for strict PII compliance.
* **Quality-Aware:** Acts as an acoustic gatekeeper, automatically flagging degraded audio and scaling ML confidence scores accordingly.
* **Production-Ready:** FastAPI async architecture with ThreadPool offloading, Docker containerization, and graceful error handling.