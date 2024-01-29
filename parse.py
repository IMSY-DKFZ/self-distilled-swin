import pandas as pd
import numpy as np
import os
import csv
import hydra


def parse_annotations(CFG, component="triplet"):
    """
    Parse annotations for triplets, instruments, verbs, or targets.

    Args:
        CFG (OmegaConf): Configuration object.
        component (str): Component type to parse, e.g., "triplet", "instrument", "verb", "target".

    Returns:
        pd.DataFrame: DataFrame containing parsed annotations for the specified component.
    """
    # Parse the specified component
    labels = []

    path = os.path.join(CFG.parent_path, component)
    video_list = sorted(os.listdir(path))

    # Loop over the videos
    for video in video_list:
        # Get the video path
        video_path = os.path.join(path, video)

        # Parse the annotations
        with open(video_path, mode="r") as f:
            reader = csv.reader(f)

            for line in reader:
                line = np.array(line, np.int64)
                frame_id, label = line[0], line[1:]

                labels.append(label)

    component_df = pd.DataFrame(labels)
    return component_df


def parse_metadata(CFG):
    """
    Parse metadata information for CholecT45 dataset.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        pd.DataFrame: DataFrame containing parsed metadata information.
    """
    # Lists to store metadata
    nframes, folders, videos, nfolder, nids = [], [], [], [], []

    # Path to videos
    data_path = os.path.join(CFG.parent_path, CFG.train_path)
    video_list = sorted(os.listdir(data_path))

    print("Start parsing the metadata")

    # Loop over the videos
    for n, video in enumerate(video_list):
        vid_path = os.path.join(data_path, video)

        frames_list = sorted(os.listdir(vid_path))

        # Loop over the frames
        for j, image_id in enumerate(frames_list):
            nid = f"{video}/{image_id}"
            nid2 = f"{n}_{j}"

            nframes.append(int(j))
            videos.append(video)
            folders.append(int(j))
            nfolder.append(nid2)
            nids.append(nid)

    # Create a new DataFrame
    metadata = pd.DataFrame.from_dict(
        {
            "folder": folders,
            "frame": nframes,
            "video": videos,
            "image_path": nids,
            "image_id": nfolder,
        }
    )

    # Sort based on video and frame ids
    metadata = metadata.sort_values(by=["video", "frame"], ascending=(True, True))

    print("Start parsing the annotations")

    # Parse annotations for triplets, instruments, verbs, and targets
    triplet = parse_annotations(CFG, "triplet")
    instrument = parse_annotations(CFG, "instrument")
    verb = parse_annotations(CFG, "verb")
    target = parse_annotations(CFG, "target")

    # Concatenate metadata and annotations
    final_df = pd.concat(
        [
            metadata,
            triplet.add_prefix("tri"),
            instrument.add_prefix("inst"),
            verb.add_prefix("v"),
            target.add_prefix("t"),
        ],
        axis=1,
    )

    # Compute combination of triplets per frame
    all_tar = []

    final_df["multi_tri"] = -1
    for i in range(len(triplet)):
        triplets = []
        row = triplet.loc[i]
        for j, k in enumerate(range(15)):
            if row[k] == 1:
                triplets.append(j)
        all_tar.append(triplets)

    # Save the triplet combination in a new column
    final_df["multi_tri"] = all_tar

    # Create a new DataFrame folder to save the final csv
    dataframes_folder = os.path.join(CFG.parent_path, "dataframes")
    if not os.path.exists(dataframes_folder):
        os.mkdir(dataframes_folder)

    # Save final csv
    final_df.to_csv(os.path.join(dataframes_folder, "CholecT45.csv"), index=False)

    return metadata


# Run the code
@hydra.main(config_name="config")
def parse(CFG):
    """
    Main function to parse metadata and annotations for CholecT45 dataset.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None
    """
    parse_metadata(CFG)


if __name__ == "__main__":
    parse()
