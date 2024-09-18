## process video to images
import os
from tqdm import tqdm
import argparse
from ixc_utils import load_video,frame2img,Video_transform,get_font

def process_single(input_video_path, output_image_path,num_frm):
    video = load_video(input_video_path,num_frm) #
    img = frame2img(video, get_font())
    img = Video_transform(img)
    img.save(output_image_path)
    return

def process_folder(input_video_folder, output_image_folder,num_frm):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
        print(f"Directory '{output_image_folder}' created successfully")
    else:
        print(f"Directory '{output_image_folder}' already exists")

    videos = os.listdir(input_video_folder)
    for video in tqdm(videos):
        image_name = video.replace('.mp4','.jpg')
        process_single(input_video_folder+'/'+video, output_image_folder+'/'+image_name,num_frm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_folder', type=str, default='../klook_data/first_page_pics')
    parser.add_argument('--output_image_folder', type=str, default='../klook_data/imgs')
    parser.add_argument('--num_frm', type=int, default=8)
    args = parser.parse_args()

    process_folder(args.input_video_folder,args.output_image_folder,args.num_frm)