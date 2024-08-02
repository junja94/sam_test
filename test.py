import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

class ImageClickCoordinates:
    def __init__(self, image, marker_color=(0, 255, 0), id=0):
        self.image = image.copy()
        self.coordinates = []
        self.negative_coordinates = []
        self.marker_color = marker_color
        self.image_name = "Image"+str(id)

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down event
            print(f"Coordinates: ({x}, {y}), Press any key to stop")
            self.coordinates.append((x, y))
            cv2.circle(self.image, (x, y), 5, self.marker_color, -1)
            cv2.imshow(self.image_name, self.image)
        if event == cv2.EVENT_MBUTTONDOWN:  # Right mouse button down event:
            print(f"Coordinates: ({x}, {y}), Press any key to stop")
            self.negative_coordinates.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 0, 0), -1)
            cv2.imshow(self.image_name, self.image)
        
            
    def show_image_and_collect_coordinates(self):
        # Create a window and set a mouse callback
        cv2.imshow(self.image_name, self.image)
        cv2.setMouseCallback(self.image_name, self.click_callback)

        # Wait until a key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.coordinates, self.negative_coordinates

def collect_clicks_from_frame(frame_idx, frames_cv_in, id = 0):
    #sample random color #(0 to 255)
    marker_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    image_to_click = frames_cv_in[frame_idx]
    click_collector = ImageClickCoordinates(image_to_click, marker_color, id)
    pos_coords, neg_coords = click_collector.show_image_and_collect_coordinates()

    # concatenate positive and negative clicks
    all_coords = pos_coords + neg_coords
    points = np.array(all_coords, dtype=np.float32)
    
    # assign labels to the clicks
    pos_labels = np.array([1] * len(pos_coords), np.int32)
    neg_labels = np.array([0] * len(neg_coords), np.int32)
    labels = np.concatenate([pos_labels, neg_labels], axis= 0)

    return points, labels

def get_frames_from_video(video_input):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # cv2.imshow('frame', frame)
                # cv2.waitKey(1)
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    return frames, fps, image_size


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
from sam2.build_sam import build_sam2_video_predictor
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Segment Anything 2')
parser.add_argument('--video_path', type=str, help='Path to the input video', default="/home/joonho/git/sam_test/videos/test1.mp4")
parser.add_argument('--sam_path', type=str, help='Path to the output directory', default="/home/joonho/git/segment-anything-2")
args = parser.parse_args()

# Get the video path and output directory from command line arguments
video_path = args.video_path
sam_path = args.sam_path


sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

ckpt_path = os.path.join(sam_path, sam2_checkpoint)

predictor = build_sam2_video_predictor(model_cfg, ckpt_path)


video_dir = os.path.dirname(video_path)
frames_cv, fps, image_size = get_frames_from_video(video_path)
video_name = video_path.split("/")[-1].split(".")[0]

output_dir_name = "output_" + video_name

output_dir = os.path.join(video_dir, output_dir_name)
frame_dir = os.path.join(output_dir, "frames")

# make a directory to save frames
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

frame_names = [f"{i}.jpg" for i in range(len(frames_cv))]
for i, frame in enumerate(frames_cv):
    Image.fromarray(frame).save(os.path.join(frame_dir, frame_names[i]))


# Initlaize the inference state
inference_state = predictor.init_state(video_path=frame_dir)
predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
object_id = 0

label_names = []

while True:
        # Get label name input from the user
        label_name = input("Enter label name (or type 'exit' to quit): ")
        if label_name.lower() == 'exit':
            break
        
        # Collect clicks from the frame for the current label index
        points, labels = collect_clicks_from_frame(ann_frame_idx, frames_cv, object_id)
        
        # Run the prediction and process the points
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
        
        print(f"Processed label '{label_name}' with label index {object_id}")

        
        # Increment label index for the next label type
        object_id += 1
        label_names.append(label_name)
        


# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
times = []
num_objs = object_id

print(num_objs)
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    start_time = time.time()
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    time_this_frame = time.time() - start_time
    times.append(time_this_frame)
    
print(f"Average propagation time per frame: {np.mean(times):.8f} seconds")


results_dir = os.path.join(output_dir, "segmentation_results")
mask_colors = np.random.random((num_objs, 3))
obj_dirs = []
masked_frames = []

for obj_id in range(num_objs):
    dir_this_obj = os.path.join(results_dir, f"obj_{obj_id}")
    if not os.path.exists(dir_this_obj):
        os.makedirs(dir_this_obj)
    
    frames_dir = os.path.join(dir_this_obj, "frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    

    
    obj_dirs.append((dir_this_obj, frames_dir))
    masked_frames.append([])
        
# save frames for each object
for out_frame_idx in range(0, len(frame_names)):
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        # mask_color =  mask_colors[out_obj_id]
        
        mask_color = np.array([0, 0, 255])
        
        dir_name_this_obj = obj_dirs[out_obj_id][0]
        frame_dir_this_obj = obj_dirs[out_obj_id][1]
            
        # save the mask 
        mask = out_mask.astype(np.uint8) * 255
        np.save(os.path.join(dir_name_this_obj, '{:05d}.npy'.format(i)), mask)
        
        # # # show mask image 
        # if out_frame_idx == 0:
        #     plt.imshow(mask.transpose(1, 2, 0))
        #     plt.show()
            
        # save masked frame
        frame = frames_cv[out_frame_idx].copy()
        mask_image = out_mask.transpose(1, 2, 0) * mask_color.reshape(1, 1, -1)
        mask_image_uint8 = (mask_image).astype(np.uint8)
        
        cv2.imshow('mask_image_uint8', mask_image_uint8)
        cv2.waitKey(1)
        
        # add mask_image to frame
        overlay_image = cv2.addWeighted(frame, 1, mask_image_uint8, 1.0, 0)

        
        # save overlay_image
        Image.fromarray(overlay_image).save(os.path.join(frame_dir_this_obj, frame_names[out_frame_idx]))
        
        masked_frames[obj_id].append(overlay_image)
        

# write video
for obj_id in range(num_objs):
    video_dir_this_obj = obj_dirs[obj_id][0]
    
    # Define the output video path
    output_video_path = os.path.join(video_dir_this_obj, "segemented.mp4")

    # Get the frame size from the first frame
    frame_size = (masked_frames[obj_id][0].shape[1], masked_frames[obj_id][0].shape[0])

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(filename=output_video_path,
                                   apiPreference=cv2.CAP_FFMPEG,
                                   fourcc=fourcc,
                                   fps=fps,
                                   frameSize=frame_size)


    print("fps ", fps)
    # Write each frame to the video
    for frame in masked_frames[obj_id]:
        video_writer.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)
        
    # Release the VideoWriter object
    video_writer.release()

    # Print the path to the output video
    print(f"Video saved at: {output_video_path}")


# Render the segmentation results every few frames using OpenCV
# vis_frame_stride = 15
# for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
#     frame = cv2.imread(os.path.join(frame_dir, frame_names[out_frame_idx]))
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         mask_color = mask_colors[out_obj_id]
#         mask_image = out_mask.transpose(1, 2, 0) * mask_color.reshape(1, 1, -1)
#         mask_image_uint8 = (mask_image * 255).astype(np.uint8)
#         overlay_image = cv2.addWeighted(frame, 1, mask_image_uint8, 0.5, 0)
#         cv2.imshow("Segmentation Results", overlay_image)
#         cv2.waitKey(0)
# cv2.destroyAllWindows()