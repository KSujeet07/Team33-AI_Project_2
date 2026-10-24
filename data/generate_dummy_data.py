
import os
import cv2
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("data_dir", "EchoNet-Dynamic")
VIDEOS_DIR = os.path.join(DATA_DIR, "Videos")
FILE_LIST_PATH = os.path.join(DATA_DIR, "FileList.csv")


def create_synthetic_dataset(num_samples=30):
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    records = []
    splits = ["TRAIN"] * 20 + ["VAL"] * 5 + ["TEST"] * 5

    print("Generating synthetic videos...")

    for i in range(num_samples):
        filename = f"0X1A{i:04d}.avi"
        video_path = os.path.join(VIDEOS_DIR, filename)

        ef = np.random.uniform(25.0, 70.0)
        edv = np.random.uniform(80.0, 160.0)
        esv = edv * (1.0 - ef / 100.0)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            video_path, fourcc, 50.0, (112, 112), True
        )

        if not out.isOpened():
            raise RuntimeError(f"Could not create video: {video_path}")

        for frame_idx in range(50):
            radius = int(
                20 + 10 * np.sin(2 * np.pi * frame_idx / 25)
            )
            frame = np.zeros((112, 112, 3), dtype=np.uint8)
            cv2.circle(frame, (56, 56), radius, (255, 255, 255), -1)
            out.write(frame)

        out.release()

        records.append({
            "FileName": filename,
            "EF": round(ef, 2),
            "EDV": round(edv, 2),
            "ESV": round(esv, 2),
            "Split": splits[i],
            "FPS": 50,
            "NumberOfFrames": 50
        })

    df = pd.DataFrame(records)
    df.to_csv(FILE_LIST_PATH, index=False)

    print(f"Created {len(records)} synthetic videos.")
    print(f"FileList saved at: {FILE_LIST_PATH}")


if __name__ == "__main__":
    create_synthetic_dataset()
