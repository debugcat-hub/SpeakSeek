import os
import pandas as pd
from moviepy.editor import VideoFileClip
from config import Config

def assign_speakers(matched_intervals_file, diarization_file):
    """Assign speakers to matched frame intervals"""
    if not os.path.exists(matched_intervals_file):
        raise FileNotFoundError(f"Matched intervals file not found: {matched_intervals_file}")
    if not os.path.exists(diarization_file):
        raise FileNotFoundError(f"Diarization file not found: {diarization_file}")

    matched_df = pd.read_csv(matched_intervals_file)
    diar_df = pd.read_csv(diarization_file)

    matched_df['speaker'] = None

    for i, row in matched_df.iterrows():
        match_start = row['start']
        match_end = row['end']

        overlap = diar_df[
            (diar_df['start'] < match_end) &
            (diar_df['end'] > match_start)
        ]

        if not overlap.empty:
            matched_df.at[i, 'speaker'] = ','.join(overlap['speaker'].unique())
        else:
            matched_df.at[i, 'speaker'] = 'UNKNOWN'

    output_file = "matched_with_speakers.csv"
    matched_df.to_csv(output_file, index=False)
    return output_file

def extract_top_speaker_clips(video_path, lip_scores_file, matched_speakers_file, output_dir):
    """Extract video clips for the primary speaker"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(lip_scores_file):
        raise FileNotFoundError(f"Lip scores file not found: {lip_scores_file}")
    if not os.path.exists(matched_speakers_file):
        raise FileNotFoundError(f"Matched speakers file not found: {matched_speakers_file}")

    lip_scores = pd.read_csv(lip_scores_file)
    matched_speakers = pd.read_csv(matched_speakers_file)

    top_frames = lip_scores.nlargest(5, "composite_score")["frame"].tolist()

    speaker_counts = {}
    for frame in top_frames:
        speaker = matched_speakers[matched_speakers["frame_file"] == frame]["speaker"].values
        if len(speaker) > 0:
            speakers = speaker[0].split(',')
            for sp in speakers:
                speaker_counts[sp] = speaker_counts.get(sp, 0) + 1

    if not speaker_counts:
        raise ValueError("No speakers found in top frames")

    primary_speaker = max(speaker_counts, key=speaker_counts.get)
    print(f"[INFO] Primary speaker identified: {primary_speaker}")

    os.makedirs(output_dir, exist_ok=True)

    video = None
    try:
        video = VideoFileClip(video_path)

        speaker_segments = matched_speakers[
            matched_speakers["speaker"].apply(lambda x: primary_speaker in x.split(','))
        ]

        segments = []
        current_start = None
        current_end = None
        config = Config()

        for _, row in speaker_segments.sort_values("start").iterrows():
            if current_start is None:
                current_start = row["start"]
                current_end = row["end"]
            elif row["start"] - current_end <= config.SEGMENT_MERGE_THRESHOLD:
                current_end = row["end"]
            else:
                segments.append((current_start, current_end))
                current_start = row["start"]
                current_end = row["end"]

        if current_start is not None:
            segments.append((current_start, current_end))

        clip_paths = []
        for i, (start, end) in enumerate(segments):
            try:
                clip = video.subclip(start, end)
                output_path = os.path.join(output_dir, f"clip_{i+1}_{start:.2f}-{end:.2f}.mp4")
                clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
                clip.close()
                clip_paths.append(output_path)
                print(f"[INFO] Created clip {i+1}: {start:.2f}s - {end:.2f}s")
            except Exception as e:
                print(f"[WARNING] Failed to create clip {i+1}: {str(e)}")

        return clip_paths, primary_speaker

    except Exception as e:
        raise RuntimeError(f"Clip extraction failed: {str(e)}")
    finally:
        if video is not None:
            video.close()

def cleanup_temp_files(output_base):
    """Clean up temporary files"""
    try:
        import shutil
        if os.path.exists(output_base):
            shutil.rmtree(output_base)
            print(f"[INFO] Cleaned up temporary directory: {output_base}")
    except Exception as e:
        print(f"[WARNING] Failed to cleanup temp files: {str(e)}")
