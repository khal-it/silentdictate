# Third-party licenses

SilentDictate's own code is MIT-licensed. Dependencies and models keep their own licenses, and you're responsible for complying with them, especially if you redistribute a packaged build.

## Worth checking

| Component | License | How it's used |
|---|---|---|
| `piper-tts` | GPL-3.0-or-later | Optional `piper` extra; a dev dependency for tests. A build that bundles it must meet the GPL |
| `pynput` | LGPL-3.0 | Required, used as an unmodified library |
| `certifi`, `tqdm`, `pathspec` | MPL-2.0 (`tqdm` also MIT) | Used unmodified |

The other required dependencies use permissive licenses such as MIT, BSD and Apache-2.0. `uv.lock` has the exact versions.

## Models

Models aren't included in this repository. They're downloaded from Hugging Face on request, and each has its own license. Read its model card before commercial use. That matters most for Piper voices, because some come from datasets with their own terms.

| Model | Source |
|---|---|
| Whisper | [OpenAI](https://github.com/openai/whisper) models in CTranslate2 format |
| Silero VAD | Bundled with faster-whisper |
| Parakeet TDT | [NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Piper voices | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) |
