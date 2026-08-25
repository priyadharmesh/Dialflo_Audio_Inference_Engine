"""
WebSocket Streaming Test Script

Tests the /stream WebSocket endpoint with real audio files.

Usage:
    python test_websocket.py <audio_file>

Example:
    python test_websocket.py test_audio_for_quality_check/01_good_quality.wav
    python test_websocket.py tts_voice_samples/04_female_young_american.mp3

Requirements:
    pip install websockets
"""

import asyncio
import json
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


async def stream_audio_file(file_path: str, chunk_size: int = 16000):
    """
    Stream an audio file to the WebSocket endpoint.

    Args:
        file_path: Path to audio file (WAV, MP3, etc.)
        chunk_size: Bytes per chunk (16000 = ~1 second of WAV)
    """
    uri = "ws://localhost:8000/stream"

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print(f"\n{'='*60}")
    print(f"Streaming: {file_path.name}")
    print(f"File size: {file_path.stat().st_size:,} bytes")
    print(f"Chunk size: {chunk_size:,} bytes")
    print(f"{'='*60}\n")

    try:
        async with websockets.connect(uri) as ws:
            # Receive connection confirmation
            response = await ws.recv()
            data = json.loads(response)
            print(f"[Connected] Session: {data.get('session_id', 'unknown')[:8]}...")
            print()

            # Read and send audio in chunks
            chunk_num = 0
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    chunk_num += 1
                    await ws.send(chunk)
                    print(f"[Sent] Chunk {chunk_num}: {len(chunk):,} bytes")

                    # Receive prediction
                    response = await ws.recv()
                    data = json.loads(response)

                    msg_type = data.get("type", "unknown")

                    if msg_type == "waiting":
                        print(f"  └─ Waiting: {data.get('message')}")

                    elif msg_type == "buffering":
                        bytes_recv = data.get("bytes_received", 0)
                        bytes_need = data.get("bytes_needed", 0)
                        pct = (bytes_recv / bytes_need * 100) if bytes_need > 0 else 0
                        print(f"  └─ Buffering: {bytes_recv:,}/{bytes_need:,} bytes ({pct:.0f}%)")

                    elif msg_type == "partial":
                        gender = data.get("gender", {})
                        age = data.get("age_bracket", {})
                        quality = data.get("audio_quality", "?")
                        duration = data.get("duration_seconds", 0)

                        print(f"  └─ Prediction ({duration:.1f}s, quality={quality}):")
                        print(f"      Gender: {gender.get('prediction')} ({gender.get('confidence', 0):.0%})")
                        print(f"      Age: {age.get('prediction')} ({age.get('confidence', 0):.0%})")

                    elif msg_type == "error":
                        print(f"  └─ Error: {data.get('error')}")

                    else:
                        print(f"  └─ {msg_type}: {data.get('message', '')}")

                    print()

                    # Small delay between chunks (simulates real-time streaming)
                    await asyncio.sleep(0.1)

            # Signal done and get final result
            print("-" * 40)
            await ws.send("done")
            response = await ws.recv()
            data = json.loads(response)

            print("\n[FINAL RESULT]")
            print(f"  Total chunks: {data.get('total_chunks', chunk_num)}")
            print(f"  Duration: {data.get('duration_seconds', 0):.2f}s")
            print(f"  Quality: {data.get('audio_quality', '?')}")

            gender = data.get("gender", {})
            age = data.get("age_bracket", {})
            print(f"  Gender: {gender.get('prediction')} ({gender.get('confidence', 0):.0%})")
            print(f"  Age: {age.get('prediction')} ({age.get('confidence', 0):.0%})")

    except websockets.exceptions.ConnectionRefused:
        print("Error: Could not connect to server.")
        print("Make sure the server is running: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"Error: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable test files:")

        # List available test files
        test_dirs = [
            "test_audio_for_quality_check",
            "comprehensive_voice_tests",
            "tts_voice_samples"
        ]

        for dir_name in test_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                files = list(dir_path.glob("*.*"))[:3]
                if files:
                    print(f"\n  {dir_name}/")
                    for f in files:
                        print(f"    - {f.name}")

        return

    audio_file = sys.argv[1]

    # Optional: chunk size as second argument
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16000

    asyncio.run(stream_audio_file(audio_file, chunk_size))


if __name__ == "__main__":
    main()
