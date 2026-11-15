"""The real 2 kHz dataset must preview without browser decoder support."""
import base64
import io
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile


def test_browser_pcm_codec_preserves_waveform_and_resamples_playback():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser PCM codec regression")
    samples = np.array([0, 16384, -16384, 32767, -32768] * 400, dtype=np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, 2000, samples)
    program = """
require('./api/web/wav.js');
const assert = require('node:assert/strict');
(async () => {
const raw = Buffer.from(process.argv[1], 'base64');
const decoded = CardioWave.decode(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
assert.equal(decoded.sampleRate, 2000);
assert.equal(decoded.duration, 1);
assert.equal(decoded.signal[1], .5);
assert.equal(decoded.signal[2], -.5);
const playback = CardioWave.decode(await CardioWave.playback(decoded.signal, decoded.sampleRate).arrayBuffer());
assert.equal(playback.sampleRate, 8000);
assert.equal(playback.duration, 1);
assert.ok(Math.abs(playback.signal[4] - .5) < 1/10000);
assert.ok(playback.signal.every(Number.isFinite));
})();
"""
    subprocess.run([node, "-e", program, base64.b64encode(buffer.getvalue()).decode()],
        cwd=Path(__file__).resolve().parent.parent, check=True, capture_output=True, text=True)
