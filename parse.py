import pandas as pd 
import numpy as np
import os
import csv
import hydra



def parse_annotations(CFG, component= "triplet"):

    """
    Function to parse the triplet, instrument, verb and targets
    
    CFG: config.yaml containing the path to the CholecT45 dataset
    component: triplet, instrument, verb, target
    """

    # Parse the triplets:
    image_batch, triplet_batch, video_name_batch, frame_id_batch, nfolder = [], [], [], [], []


    path= os.path.join(CFG.parent_path, component)
    video_list = sorted(os.listdir(path))

    # Loop over the videos
    for k, video in enumerate(video_list):

        # Get the video path
        video_path= os.path.join(path, video)
        
        # Parse the annotations
        with open(video_path, mode='r') as f:
            reader = csv.reader(f)
        
            for line in reader:
                line = np.array(line, np.int64)
                frame_id, triplet_label = line[0], line[1:]

                triplet_batch.append(triplet_label)
                                
    component_df = pd.DataFrame(triplet_batch)

    return component_df



def parse_metadata(CFG):

    """
    Function to parse the path to the images
    """

    # Empty list to store the data
    image_ids= []
    nframes= []
    folder= []
    videos = []
    nfolder = []
    nids= []


    # Path to videos
    data_path = os.path.join(CFG.parent_path, CFG.train_path)
    video_list = sorted(os.listdir(data_path))

    print("Start parsing the metadata")
    
    # loop over the videos
    for n , video in enumerate(video_list):
        vid_path= os. path.join(data_path, video)
        
        frames_list = sorted(os.listdir(vid_path))

        # loop over the frames
        for j,image_id in enumerate(frames_list):

            nid= f"{video}/{image_id}"
            nid2= f"{n}_{j}"

            nframes.append(int(j))
            videos.append(video)
            folder.append(int(j))
            nfolder.append(nid2)
            nids.append(nid)

        
    # Create a new df
    metadata = pd.DataFrame.from_dict({'folder':folder, 'nframe':nframes,
                                        "video":videos,
                                        "nid": nids, 'id2':nfolder})
    
    # Sort based on video and frame ids
    metadata= metadata.sort_values(by=['video', 'nframe'], ascending=(True, True))


    print('Start parsing the annotations')

    # Triplet annotations
    triplet = parse_annotations(CFG, "triplet")
    instrument= parse_annotations(CFG, "instrument")
    verb = parse_annotations(CFG, "verb")
    target = parse_annotations(CFG, "target")



    # Concatena the metadata and annotations
    final_df = pd.concat([metadata,
                           triplet.add_prefix('tri'),
                           instrument.add_prefix('inst'),
                           verb.add_prefix('v'),
                           target.add_prefix('t')], axis=1)
    

    # Compute combination of triplets per frame
    all_tar = []
    
    final_df['multi_tri'] = -1
    for i in range(len(triplet)):
        triplets = []
        row = triplet.loc[i]
        for j,k in enumerate(range(15)):
            if row[k] == 1:
                triplets.append(j)
        all_tar.append(triplets)
    
        
    ##Save the triplet combination in a new column
    final_df['multi_tri'] = all_tar

    # Create a new dataframes folder to save the final csv
    dataframes_folder = os.path.join(CFG.parent_path, "dataframes")
    if not os.path.exists(dataframes_folder):
        os.mkdir(dataframes_folder)
    
    # Save final csv
    final_df.to_csv(os.path.join(dataframes_folder, "CholecT45.csv"), index=False)

    return metadata



# Run the code
@hydra.main(config_name="config")
def parse(CFG):
    parse_metadata(CFG)


if __name__ == "__main__":
    parse()




