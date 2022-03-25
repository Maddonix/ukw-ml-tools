import subprocess
from pathlib import Path
import json
import warnings
import cv2

def get_video_meta(path: Path) -> dict:
    command = f"ffprobe -hide_banner -loglevel fatal \
        -show_error -show_format -show_streams -show_chapters \
            -show_private_data -print_format json '{path.as_posix()}'"

    _ = subprocess.run(command, capture_output=True, shell = True)

    meta = json.loads(_.stdout)
    # assert len(meta["streams"]) == 1
    try:
        stream = meta["streams"][0]
        fps_strings = stream["r_frame_rate"].split("/")
        fps = float(fps_strings[0])/float(fps_strings[1])
        duration = float(meta["format"]["duration"])
        frames_total = fps*duration
    
    except:
        warnings.warn(f"Could not Generate Metadata for {path}, using cv2")
        cap = cv2.VideoCapture(path.as_posix())
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames_total = 0
        success, frame = cap.read()

        while success:
            frames_total +=1
            success, frame = cap.read()

        duration = frames_total / fps

    meta = {
        "fps": fps,
        "frames_total": frames_total,
        "duration": duration
    }

    return meta