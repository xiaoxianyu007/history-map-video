#!/usr/bin/env python3
"""
历史地图视频一条龙生成管线
用法: python generate.py
流程:
  1. 读取 narration.txt → 分段
  2. 每段生成 TTS → 获取时长
  3. 逐帧渲染地图动画 (matplotlib) → FFmpeg 合成视频
  4. 拼接音频 → 生成字幕 → FFmpeg 最终合成
"""

import asyncio
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
NARRATION_FILE = BASE_DIR / "narration.txt"
OUTPUT_DIR = BASE_DIR / "output"
WORK_DIR = OUTPUT_DIR / "work"
FRAMES_DIR = WORK_DIR / "frames"
FPS = 24
VIDEO_RES = "1920:1080"
TTS_VOICE = "zh-CN-YunyangNeural"     # 深沉男声
TTS_RATE = "+0%"

SEGMENT_SEP = re.compile(r"^\s*---\s*$", re.MULTILINE)

# ============================================================
# 工具函数
# ============================================================
def _find_exe(name: str) -> str:
    # 优先使用 pixelle_video 环境中的完整版 ffmpeg
    candidates = [
        "/home/shuju46/miniconda3/envs/pixelle_video/bin",
        "/home/shuju46/miniconda3/bin",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "bin"))
    for bindir in candidates:
        full = os.path.join(bindir, name)
        if os.path.isfile(full):
            return full
    return name


def get_env() -> dict:
    return os.environ.copy()


# ============================================================
# 1. 解析旁白
# ============================================================
def parse_narration(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").strip()
    segs = [s.strip() for s in SEGMENT_SEP.split(raw) if s.strip()]
    if not segs:
        raise ValueError("旁白文件为空")
    print(f"  共 {len(segs)} 段旁白")
    for i, s in enumerate(segs):
        print(f"    段{i+1}: {s[:40]}...")
    return segs


# ============================================================
# 2. TTS 音频生成
# ============================================================
async def generate_tts(segments: list[str], work_dir: Path) -> tuple[list[Path], list[float]]:
    print(f"\n[2/5] 生成 TTS 音频...")
    try:
        import edge_tts
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts"], check=True)
        import edge_tts

    audio_paths: list[Path] = []
    durations: list[float] = []

    for i, text in enumerate(segments):
        ap = work_dir / f"seg_{i:03d}.mp3"
        comm = edge_tts.Communicate(text=text, voice=TTS_VOICE, rate=TTS_RATE)
        await comm.save(str(ap))

        # 获取时长
        from mutagen.mp3 import MP3 as MutagenMP3
        dur = MutagenMP3(str(ap)).info.length
        audio_paths.append(ap)
        durations.append(dur)
        print(f"  段{i+1}: {dur:.1f}s  \"{text[:30]}...\"")

    total = sum(durations)
    print(f"  总时长: {total:.1f}s")
    return audio_paths, durations


# ============================================================
# 3. 渲染地图动画帧
# ============================================================
def render_map_frames(segments: list[str], durations: list[float]):
    print(f"\n[3/5] 渲染地图动画帧 ({FPS}fps)...")
    sys.path.insert(0, str(BASE_DIR))
    from warring_states import render_frame, CONQUEST_ORDER, TERRITORIES

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    CONQUEST_ORDER_LIST = CONQUEST_ORDER  # ["han","zhao","wei","chu","yan","qi"]

    total_frames = 0
    conquered_before = set()

    for seg_idx, (seg_text, dur) in enumerate(zip(segments, durations)):
        n_frames = max(1, int(dur * FPS))

        # 判断本段目标
        target = None
        if 1 <= seg_idx <= 6:  # 段1-6是灭六国
            target = CONQUEST_ORDER_LIST[seg_idx - 1]

        if seg_idx == 0:
            # 开场：无目标，显示所有国家
            target = None
        elif seg_idx >= 7:
            # 结尾：统一
            target = None
            conquered_before = set(TERRITORIES.keys()) - {"qin"}

        seg_frames_dir = FRAMES_DIR / f"seg_{seg_idx:03d}"
        seg_frames_dir.mkdir(exist_ok=True)

        for frame_i in range(n_frames):
            progress = frame_i / n_frames if n_frames > 1 else 0.0

            # 本段内，当 progress > 0.6 时，目标已征服
            conquered = set(conquered_before)
            if target and progress >= 0.6:
                conquered.add(target)

            fname = f"frame_{total_frames:06d}.png"
            fpath = str(seg_frames_dir / fname)

            render_frame(
                output_path=fpath,
                conquered=conquered,
                target=target,
                progress=progress,
                seg_index=seg_idx,
                dpi=72,
                figsize=(16, 9),
            )
            total_frames += 1

        # 更新已征服集合（进入下一段时）
        if target:
            conquered_before.add(target)

        print(f"  段{seg_idx+1}: {n_frames} 帧 (约{dur:.1f}s) [{seg_text[:25]}...]")

    print(f"  共 {total_frames} 帧")
    return total_frames


# ============================================================
# 4. FFmpeg 合成视频（帧→MP4）
# ============================================================
def frames_to_video(total_frames: int, output_path: Path):
    print(f"\n[4/5] 合成视频（OpenCV 逐帧写入）...")
    import cv2
    import numpy as np

    w, h = (int(x) for x in VIDEO_RES.split(":"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (w, h))

    seg_dirs = sorted(FRAMES_DIR.glob("seg_*"))
    frame_idx = 0
    total_processed = 0

    for sd in seg_dirs:
        for f in sorted(sd.glob("frame_*.png")):
            img = cv2.imread(str(f))
            if img is None:
                continue
            # 缩放填充到目标分辨率
            h_orig, w_orig = img.shape[:2]
            scale = min(w / w_orig, h / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # 填充到目标尺寸
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            writer.write(canvas)
            total_processed += 1
            frame_idx += 1

    writer.release()
    print(f"  写入 {total_processed} 帧 → {output_path}")
    return output_path


# ============================================================
# 5. 音频拼接
# ============================================================
def concat_audio(audio_paths: list[Path], output_path: Path):
    print(f"\n[5/5] 拼接音频（pydub）...")
    from pydub import AudioSegment

    # 设置正确的 ffmpeg 路径
    ffmpeg_path = _find_exe("ffmpeg")
    AudioSegment.converter = ffmpeg_path

    combined = None
    for ap in audio_paths:
        seg = AudioSegment.from_file(str(ap), format="mp3")
        if combined is None:
            combined = seg
        else:
            combined = combined + seg

    out = output_path.with_suffix(".wav")
    combined.export(str(out), format="wav")
    print(f"  拼接音频: {out} ({len(audio_paths)} 段, {combined.duration_seconds:.1f}s)")
    return out


# ============================================================
# 6. 生成 ASS 字幕
# ============================================================
def generate_subtitles(segments: list[str], durations: list[float],
                       starts: list[float], output_path: Path):
    print(f"\n[6/6] 生成字幕...")
    w, h = [int(x) for x in VIDEO_RES.split(":")]

    font_size = 32
    margin_v = 120

    ass_header = f"""[Script Info]
Title: History Demo Subtitles
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    LEAD_IN = 0.15
    LEAD_OUT = 0.10

    for i, seg in enumerate(segments):
        if i >= len(durations):
            break
        text = seg.replace("\n", " ")
        # 每行最多 25 字换行
        wrapped = "\\N".join(text[j:j+25] for j in range(0, len(text), 25))
        start_t = _fmt_ass_time(starts[i] + LEAD_IN)
        end_t = _fmt_ass_time(starts[i] + durations[i] - LEAD_OUT)
        events.append(f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{wrapped}")

    output_path.write_text(ass_header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"  字幕: {output_path}")
    return output_path


def _fmt_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ============================================================
# 7. 最终合成
# ============================================================
def compose_final(video_path: Path, audio_path: Path, ass_path: Path) -> Path:
    print(f"\n[7/7] 合成最终视频...")
    ffmpeg = _find_exe("ffmpeg")
    env = get_env()

    out = OUTPUT_DIR / "qin_conquest_demo.mp4"
    ass_escaped = str(ass_path).replace(":", r"\:")

    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-vf", f"subtitles='{ass_escaped}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-map", "0:v:0", "-map", "1:a:0",
        str(out),
    ]
    print(f"  {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print("  FFmpeg 最终合成失败:")
        print(r.stderr[-2000:])
        raise RuntimeError("FFmpeg 最终合成失败")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n{'='*55}")
    print(f"  ✅ 完成! 最终视频: {out}")
    print(f"  📏 大小: {size_mb:.1f} MB")
    print(f"{'='*55}")
    return out


# ============================================================
# 主流程
# ============================================================
async def main():
    print("=" * 55)
    print("  历史地图视频一条龙生成管线")
    print("  选题: 秦灭六国之战")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 解析旁白
    print(f"\n[1/7] 解析旁白...")
    segments = parse_narration(NARRATION_FILE)

    # 2) TTS（如果已有则跳过，否则生成）
    audio_paths = sorted(WORK_DIR.glob("seg_*.mp3"))
    if len(audio_paths) == len(segments):
        from mutagen.mp3 import MP3 as MutagenMP3
        durations = [MutagenMP3(str(ap)).info.length for ap in audio_paths]
        print(f"  直接使用已有音频，共 {len(audio_paths)} 段，总时长 {sum(durations):.1f}s")
    else:
        audio_paths, durations = await generate_tts(segments, WORK_DIR)

    # 3) 渲染地图帧（如果已有则跳过）
    frames_exist = len(list(FRAMES_DIR.rglob("frame_*.png"))) >= sum(max(1, int(d * FPS)) for d in durations) * 0.9
    if frames_exist:
        total_frames = len(list(FRAMES_DIR.rglob("frame_*.png")))
        print(f"  跳过帧渲染，已有 {total_frames} 帧")
    else:
        total_frames = render_map_frames(segments, durations)

    # 4) 帧→视频
    raw_video = WORK_DIR / "raw_video.mp4"
    frames_to_video(total_frames, raw_video)

    # 5) 拼接音频
    full_audio = WORK_DIR / "full_audio.wav"
    concat_audio(audio_paths, full_audio)

    # 6) 字幕
    starts = []
    t = 0.0
    for d in durations:
        starts.append(t)
        t += d
    ass_path = WORK_DIR / "subtitles.ass"
    generate_subtitles(segments, durations, starts, ass_path)

    # 7) 最终合成
    compose_final(raw_video, full_audio, ass_path)

    print("\n  清理临时文件...")
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(WORK_DIR / "frames_flat", ignore_errors=True)
    print("  完成!")


if __name__ == "__main__":
    asyncio.run(main())
