# English Shorts Audio Pipeline

Wan 2.2 clips → character voices → SFX → music with auto-ducking → -14 LUFS → word-by-word captions → final 9:16 Short.

## Install (GPU machine for voices)
```bash
sudo apt install ffmpeg            # needs libass for captions
pip install -r requirements.txt
```

## Folder layout
```
clips/        Wan 2.2 clips, one per scene (any size; auto-cropped to 1080x1920, 30fps)
voices/       5–15 s clean reference sample per character (your voice or licensed)
music/        royalty-free track (YouTube Audio Library etc. — never copyrighted songs)
sfx_library/  CC0 sounds in folders by tag: sfx_library/footsteps/*.wav, rain/, door/ ...
project.json  one file per video (see demo.json)
```

## Run
```bash
python pipeline.py demo.json                 # full run
python pipeline.py demo.json --voices-only   # voices first -> feed into Wan 2.2 S2V for lip-sync
python pipeline.py demo.json --no-captions
TTS_DEVICE=cpu python pipeline.py demo.json  # no GPU (slow)
```
Every step skips files that already exist in `build/<project>/`. Don't like one line?
Delete `build/<project>/voice/s02_01.wav` and re-run — only that line is regenerated.

## Natural-sounding voices
| field | where | what it does |
|---|---|---|
| `takes` | project / line | generate N takes (default 3); Whisper picks the one with every word clear and a natural pace |
| `voice_refs` + `mood` | character / line | different reference clip per emotion (record yourself scared, happy, angry...) |
| `emotion` | line | 0.3 calm … 0.9 dramatic |
| `pace` | line | 0.3 slow & weighty … 0.6 quick |
| `space` | project / scene / line | `clean`, `room`, `outdoor`, `phone`, `none` — matches the voice to the scene so it doesn't sound pasted on |
| `gap` / `gap_after` | scene / line | leave out `start` and lines follow each other with a natural 0.35 s gap |

Writing tips: contractions ("I'm", "don't"), short sentences, commas and "..." for pauses,
numbers as words ("twenty-five"), one emotion per line.

## Lip-sync workflow
1. `--voices-only` → wavs in `build/<project>/voice/`
2. Feed keyframe image + wav to **Wan 2.2 S2V** (or InfiniteTalk for long dialogue) → save as the scene clip
3. Full run. Voice is laid over the lip-synced clip at the same `start` time.

## Scene JSON fields
| field | meaning |
|---|---|
| `lines[].speaker` | key in `characters` |
| `lines[].start` | seconds into the scene |
| `lines[].emotion` | 0.3 calm … 0.9 dramatic (overrides character default) |
| `sfx.tag` | pick from `sfx_library/<tag>/` (commercial-safe if CC0) |
| `sfx.file` | exact sound file |
| `sfx.prompt` | sent to `sfx_model_cmd` (video-to-audio model) |
| `sfx.gain` | default 0.35 |
| `keep_clip_audio` | keep audio already in the clip (e.g. S2V output) |

## SFX model (optional)
Set `sfx_model_cmd` with placeholders `{video} {prompt} {duration} {out}`, e.g. for MMAudio:
```
"sfx_model_cmd": "cd ~/MMAudio && python demo.py --video {video} --prompt '{prompt}' --duration {duration} --output /tmp/mma && ffmpeg -y -i /tmp/mma/*.flac {out}"
```
⚠️ Check model weights licenses before monetizing. MMAudio weights are released for
non-commercial use; Chatterbox TTS is MIT. When unsure, use the CC0 `sfx_library`.

## YouTube checklist
- Turn on **"Altered or synthetic content"** when uploading realistic AI content.
- Vary stories, shots and voices between uploads — templated mass uploads get flagged as inauthentic content.
