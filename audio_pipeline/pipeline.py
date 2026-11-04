#!/usr/bin/env python3
"""
English Shorts audio pipeline.

  scene JSON -> character voices (Chatterbox TTS) -> SFX (model or CC0 library)
             -> per-scene mix -> concat -> music with ducking -> loudness -14 LUFS
             -> word-by-word captions (faster-whisper) -> final 9:16 Short

Every step skips work whose output already exists, so you can re-run safely,
replace any wav by hand, or run steps on different machines (GPU vs CPU).

Usage:
  python pipeline.py project.json                 # full run
  python pipeline.py project.json --voices-only   # make voice wavs first (for Wan S2V lip-sync)
  python pipeline.py project.json --no-captions
"""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path

# ---------------------------------------------------------------- helpers
def run(cmd):
    print("  $", " ".join(shlex.quote(str(c)) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)

def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)

def has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    return bool(out)

# ---------------------------------------------------------------- 1. voices
_tts = None
def tts_model(device):
    global _tts
    if _tts is None:
        from chatterbox.tts import ChatterboxTTS   # pip install chatterbox-tts (MIT)
        _tts = ChatterboxTTS.from_pretrained(device=device)
    return _tts

def make_voices(proj, work, device):
    chars = proj.get("characters", {})
    for sc in proj["scenes"]:
        for i, line in enumerate(sc.get("lines", [])):
            out = work / "voice" / f"{sc['id']}_{i:02d}.wav"
            line["_wav"] = out
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            c = chars.get(line["speaker"], {})
            print(f"[voice] {sc['id']} #{i} {line['speaker']}: {line['text'][:50]}")
            import torchaudio as ta
            m = tts_model(device)
            # reference clip can be per-emotion: "voice_refs": {"angry": "...", "sad": "..."}
            ref = c.get("voice_refs", {}).get(line.get("mood"), c.get("voice_ref"))
            n_takes = int(line.get("takes", proj.get("takes", 3)))
            takes = []
            for t in range(n_takes):
                wav = m.generate(
                    line["text"],
                    audio_prompt_path=ref,                      # 5-15s clean sample; YOUR voice or licensed
                    exaggeration=line.get("emotion", c.get("exaggeration", 0.5)),  # 0.3 calm .. 0.9 dramatic
                    cfg_weight=line.get("pace", c.get("cfg_weight", 0.5)),  # lower = slower, more deliberate
                    temperature=0.7 + 0.1 * t,                  # small variation between takes
                )
                p = out.with_name(f"{out.stem}_take{t}.wav")
                ta.save(str(p), wav, m.sr)
                takes.append(p)
            best = pick_best_take(takes, line["text"]) if len(takes) > 1 else takes[0]
            best.rename(out)
            for p in takes:
                if p.exists():
                    p.unlink()

_asr = None
def _words(s):
    import re
    return re.findall(r"[a-z0-9']+", s.lower())

def pick_best_take(takes, text):
    """Score each take: words actually spoken correctly (no slurs/skips/garbage),
    then a natural pace (~2.3-3.0 words/sec). Returns best path."""
    import difflib
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        _asr = WhisperModel("small", compute_type="int8")
    target = _words(text)
    best, best_score = takes[0], -1e9
    for p in takes:
        segs, _ = _asr.transcribe(str(p), language="en")
        heard = _words(" ".join(s.text for s in segs))
        acc = difflib.SequenceMatcher(None, target, heard).ratio()     # 1.0 = perfect
        wps = len(target) / max(duration(p), 0.1)
        pace_pen = 0 if 2.3 <= wps <= 3.0 else min(abs(wps - 2.3), abs(wps - 3.0)) * 0.1
        score = acc - pace_pen
        print(f"    {p.name}: accuracy {acc:.2f}, {wps:.1f} words/s -> {score:.2f}")
        if score > best_score:
            best, best_score = p, score
    return best

# light "recorded in a room" polish so TTS doesn't sound dry / robotic
VOICE_FX = {
    "clean": "highpass=f=80,lowpass=f=14000,acompressor=threshold=0.1:ratio=3:attack=5:release=80,"
             "equalizer=f=3000:t=q:w=1:g=2",
    "room":  "highpass=f=80,lowpass=f=13000,acompressor=threshold=0.1:ratio=3:attack=5:release=80,"
             "equalizer=f=3000:t=q:w=1:g=2,aecho=0.8:0.5:25|40:0.12|0.08",
    "outdoor": "highpass=f=100,lowpass=f=12000,acompressor=threshold=0.1:ratio=3:attack=5:release=80",
    "phone": "highpass=f=400,lowpass=f=3200,acompressor=threshold=0.2:ratio=4",
    "none": "anull",
}

# ---------------------------------------------------------------- 2. sfx
def make_sfx(proj, sc, work, root):
    """Returns a wav path or None. Order: manual file > SFX library tag > model command."""
    spec = sc.get("sfx")
    if not spec:
        return None
    out = work / "sfx" / f"{sc['id']}.wav"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    clip = root / sc["clip"]
    dur = duration(clip)

    if spec.get("file"):                                   # hand-picked sound
        src = root / spec["file"]
    elif spec.get("tag"):                                  # CC0 library: sfx_library/<tag>/*.wav
        lib = root / proj.get("sfx_library", "sfx_library") / spec["tag"]
        files = sorted(p for p in lib.glob("*") if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"))
        if not files:
            print(f"[sfx] no files for tag '{spec['tag']}' in {lib}, skipping")
            return None
        src = files[hash(sc["id"]) % len(files)]           # vary sounds between scenes
    elif spec.get("prompt") and proj.get("sfx_model_cmd"):
        # Video-to-audio model, e.g. MMAudio / HunyuanVideo-Foley. Template placeholders:
        # {video} {prompt} {duration} {out}. CHECK THE WEIGHTS LICENSE before monetizing.
        cmd = proj["sfx_model_cmd"].format(video=clip, prompt=spec["prompt"],
                                           duration=round(dur, 2), out=out)
        print("[sfx] model:", cmd)
        subprocess.run(cmd, shell=True, check=True)
        return out if out.exists() else None
    else:
        return None

    # trim/loop library sound to clip length with short fades
    run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", src, "-t", dur,
         "-af", f"afade=t=in:d=0.15,afade=t=out:st={max(dur-0.3,0)}:d=0.3",
         "-ar", "48000", "-ac", "2", out])
    return out

# ---------------------------------------------------------------- 3. scene mix
def mix_scene(proj, sc, work, root, size):
    out = work / "scenes" / f"{sc['id']}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    clip = root / sc["clip"]
    dur = duration(clip)
    w, h = size

    inputs = ["-i", clip]
    parts, labels = [], []
    n = 1
    # voices placed at their start times
    cursor = 0.2
    for i, line in enumerate(sc.get("lines", [])):
        inputs += ["-i", line["_wav"]]
        # no "start"? follow previous line with a natural conversational gap
        st = float(line["start"]) if "start" in line else cursor
        cursor = st + duration(line["_wav"]) + float(line.get("gap_after", sc.get("gap", 0.35)))
        ms = int(st * 1000)
        fx = VOICE_FX.get(line.get("space", sc.get("space", proj.get("space", "clean"))), VOICE_FX["clean"])
        parts.append(f"[{n}:a]aresample=48000,{fx},aformat=channel_layouts=stereo,"
                     f"adelay={ms}|{ms},volume={sc.get('voice_gain', 1.0)}[v{i}]")
        labels.append(f"[v{i}]"); n += 1
    sfx = make_sfx(proj, sc, work, root)
    if sfx:
        inputs += ["-i", sfx]
        parts.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,"
                     f"volume={sc.get('sfx', {}).get('gain', 0.35)}[sfx]")
        labels.append("[sfx]"); n += 1
    if sc.get("keep_clip_audio") and has_audio(clip):
        parts.append("[0:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.5[ca]")
        labels.append("[ca]")
    # silent bed so every scene has exactly clip-length audio
    parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{dur}[bed]")
    labels.append("[bed]")
    parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,"
                 f"atrim=0:{dur}[aout]")
    # normalise video: 9:16 crop/scale, 30fps, so concat is seamless
    parts.append(f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
                 f"fps=30,setsar=1,format=yuv420p[vout]")

    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(parts),
         "-map", "[vout]", "-map", "[aout]", "-t", dur,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-c:a", "pcm_s16le", out.with_suffix(".mkv")])
    out.with_suffix(".mkv").rename(out)   # keep lossless audio until final encode
    return out

# ---------------------------------------------------------------- 4. concat + music + loudness
def master(proj, scenes, work, root):
    cat = work / "concat.mp4"
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in scenes))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
         "-c", "copy", cat.with_suffix(".mkv")])
    cat = cat.with_suffix(".mkv")
    total = duration(cat)

    out = work / "mastered.mp4"
    music = proj.get("music")
    if music:
        mg = proj.get("music_gain", 0.25)
        fc = (f"[0:a]asplit=2[dia][key];"
              f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={mg},"
              f"afade=t=in:d=1,afade=t=out:st={max(total-2,0)}:d=2[mus];"
              # music ducks automatically whenever someone speaks
              f"[mus][key]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[duck];"
              f"[dia][duck]amix=inputs=2:normalize=0:duration=first,"
              f"loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
        run(["ffmpeg", "-y", "-v", "error", "-i", cat, "-stream_loop", "-1", "-i", root / music,
             "-filter_complex", fc, "-map", "0:v", "-map", "[aout]", "-t", total,
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out])
    else:
        run(["ffmpeg", "-y", "-v", "error", "-i", cat, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out])
    return out

# ---------------------------------------------------------------- 5. captions
def ass_time(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def make_captions(video, work, size, words_per_card=3, model_size="small"):
    from faster_whisper import WhisperModel          # pip install faster-whisper
    print("[captions] transcribing")
    model = WhisperModel(model_size, compute_type="int8")
    segs, _ = model.transcribe(str(video), language="en", word_timestamps=True)
    words = [w for s in segs for w in (s.words or [])]
    w, h = size
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,{int(h*0.045)},&H00FFFFFF,&H0000E5FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{int(h*0.004)},2,2,60,60,{int(h*0.28)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for i in range(0, len(words), words_per_card):
        card = words[i:i + words_per_card]
        for j, cur in enumerate(card):       # highlight the word being spoken
            end = card[j + 1].start if j + 1 < len(card) else cur.end
            txt = " ".join(("{\\c&H00E5FF&}" + x.word.strip().upper() + "{\\c&HFFFFFF&}")
                           if x is cur else x.word.strip().upper() for x in card)
            lines.append(f"Dialogue: 0,{ass_time(cur.start)},{ass_time(end)},Cap,,0,0,0,,{txt}")
    ass = work / "captions.ass"
    ass.write_text(head + "\n".join(lines) + "\n")
    return ass

def burn(video, ass, out):
    sub = str(ass).replace("\\", "/").replace(":", "\\:")
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vf", f"subtitles='{sub}'",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "copy", out])

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--voices-only", action="store_true")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--device", default=os.environ.get("TTS_DEVICE", "cuda"))
    a = ap.parse_args()

    pfile = Path(a.project).resolve()
    root = pfile.parent
    proj = json.loads(pfile.read_text())
    work = root / proj.get("work_dir", "build") / pfile.stem
    work.mkdir(parents=True, exist_ok=True)
    size = tuple(proj.get("size", [1080, 1920]))

    make_voices(proj, work, a.device)
    if a.voices_only:
        print("voices ready in", work / "voice"); return

    scenes = [mix_scene(proj, sc, work, root, size) for sc in proj["scenes"]]
    final = master(proj, scenes, work, root)
    if not a.no_captions:
        ass = make_captions(final, work, size, proj.get("words_per_caption", 3))
        out = root / proj.get("output", f"{pfile.stem}_final.mp4")
        burn(final, ass, out)
    else:
        out = root / proj.get("output", f"{pfile.stem}_final.mp4")
        os.replace(final, out)
    print("\nDONE ->", out, f"({duration(out):.1f}s)")

if __name__ == "__main__":
    main()
