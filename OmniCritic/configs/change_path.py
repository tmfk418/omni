import json
import os
import re


def modify_paths(jsonl_file, base_path="/new/base/path"):
    file_name = os.path.basename(jsonl_file)
    dir_name = os.path.dirname(jsonl_file)
    modified_file_path = os.path.join(dir_name, file_name)

    with open(jsonl_file, "r") as infile, open(modified_file_path, "w") as outfile:
        for line in infile:
            data = json.loads(line)

            # 处理旧格式中的字段
            # 修改 image 字段
            if "image" in data:
                image_path = data["image"]
                idx = image_path.find("rlaif-v-dataset")
                if idx != -1:
                    relative_path = image_path[idx:]
                    new_image_path = os.path.join(base_path, relative_path)
                    data["image"] = new_image_path

            # 修改 audio 字段
            if "audio" in data:
                audio_path = data["audio"]
                # 处理重复的audio_files路径
                audio_path = audio_path.replace("audio_files/audio_files", "audio_files")
                idx = audio_path.find("Clotho-AQA dataset")
                if idx != -1:
                    relative_path = audio_path[idx:]
                    new_audio_path = os.path.join(base_path, relative_path)
                    data["audio"] = new_audio_path

            # 修改 video 字段
            if "video" in data:
                video_path = data["video"]
                idx = video_path.find("academic_source")
                if idx != -1:
                    relative_path = video_path[idx:]
                    new_video_path = os.path.join(base_path, relative_path)
                    data["video"] = new_video_path

            # 处理新格式中的数组字段
            # 修改 images 数组中的路径
            if "images" in data and isinstance(data["images"], list):
                new_images = []
                for image_path in data["images"]:
                    idx = image_path.find("rlaif-v-dataset")
                    if idx != -1:
                        relative_path = image_path[idx:]
                        new_image_path = os.path.join(base_path, relative_path)
                        new_images.append(new_image_path)
                    else:
                        new_images.append(image_path)
                data["images"] = new_images

            # 修改 videos 数组中的路径
            if "videos" in data and isinstance(data["videos"], list):
                new_videos = []
                for video_path in data["videos"]:
                    idx = video_path.find("academic_source")
                    if idx != -1:
                        relative_path = video_path[idx:]
                        new_video_path = os.path.join(base_path, relative_path)
                        new_videos.append(new_video_path)
                    else:
                        new_videos.append(video_path)
                data["videos"] = new_videos

            # 修改 audios 数组中的路径
            if "audios" in data and isinstance(data["audios"], list):
                new_audios = []
                for audio_path in data["audios"]:
                    # 处理重复的audio_files路径
                    audio_path = audio_path.replace("audio_files/audio_files", "audio_files")
                    idx = audio_path.find("Clotho-AQA dataset")
                    if idx != -1:
                        relative_path = audio_path[idx:]
                        new_audio_path = os.path.join(base_path, relative_path)
                        new_audios.append(new_audio_path)
                    else:
                        new_audios.append(audio_path)
                data["audios"] = new_audios

            # 处理 input 字段或 messages 中的 content 字段
            if "input" in data:
                input_text = data["input"]
                # 修改 Image file
                match_img = re.search(r"Image file:\s*(.*?)(\s{2,}|\n)", input_text)
                if match_img:
                    old_image_file_path = match_img.group(1)
                    idx = old_image_file_path.find("rlaif-v-dataset")
                    if idx != -1:
                        relative_path = old_image_file_path[idx:]
                        new_image_file_path = os.path.join(base_path, relative_path)
                        input_text = re.sub(
                            r"Image file:\s*.*?(\s{2,}|\n)",
                            f"Image file: {new_image_file_path}  ",
                            input_text
                        )

                # 修改 Audio file
                match_audio = re.search(r"Audio file:\s*(.*?)(\s{2,}|\n)", input_text)
                if match_audio:
                    old_audio_file_path = match_audio.group(1)
                    # 处理重复的audio_files路径
                    old_audio_file_path = old_audio_file_path.replace("audio_files/audio_files", "audio_files")
                    idx = old_audio_file_path.find("Clotho-AQA dataset")
                    if idx != -1:
                        relative_path = old_audio_file_path[idx:]
                        new_audio_file_path = os.path.join(base_path, relative_path)
                        input_text = re.sub(
                            r"Audio file:\s*.*?(\s{2,}|\n)",
                            f"Audio file: {new_audio_file_path}  ",
                            input_text
                        )

                # 修改 Video file
                match_video = re.search(r"Video file:\s*(.*?)(\s{2,}|\n)", input_text)
                if match_video:
                    old_video_file_path = match_video.group(1)
                    idx = old_video_file_path.find("academic_source")
                    if idx != -1:
                        relative_path = old_video_file_path[idx:]
                        new_video_file_path = os.path.join(base_path, relative_path)
                        input_text = re.sub(
                            r"Video file:\s*.*?(\s{2,}|\n)",
                            f"Video file: {new_video_file_path}  ",
                            input_text
                        )

                data["input"] = input_text

            elif "messages" in data and isinstance(data["messages"], list):
                for message in data["messages"]:
                    if "content" in message:
                        content = message["content"]

                        # 修改 Image file
                        match_img = re.search(r"Image file:\s*(.*?)(\s{2,}|\n)", content)
                        if match_img:
                            old_image_file_path = match_img.group(1)
                            idx = old_image_file_path.find("rlaif-v-dataset")
                            if idx != -1:
                                relative_path = old_image_file_path[idx:]
                                new_image_file_path = os.path.join(base_path, relative_path)
                                content = re.sub(
                                    r"Image file:\s*.*?(\s{2,}|\n)",
                                    f"Image file: {new_image_file_path}  ",
                                    content
                                )

                        # 修改 Audio file
                        match_audio = re.search(r"Audio file:\s*(.*?)(\s{2,}|\n)", content)
                        if match_audio:
                            old_audio_file_path = match_audio.group(1)
                            # 处理重复的audio_files路径
                            old_audio_file_path = old_audio_file_path.replace("audio_files/audio_files", "audio_files")
                            idx = old_audio_file_path.find("Clotho-AQA dataset")
                            if idx != -1:
                                relative_path = old_audio_file_path[idx:]
                                new_audio_file_path = os.path.join(base_path, relative_path)
                                content = re.sub(
                                    r"Audio file:\s*.*?(\s{2,}|\n)",
                                    f"Audio file: {new_audio_file_path}  ",
                                    content
                                )

                        # 修改 Video file
                        match_video = re.search(r"Video file:\s*(.*?)(\s{2,}|\n)", content)
                        if match_video:
                            old_video_file_path = match_video.group(1)
                            idx = old_video_file_path.find("academic_source")
                            if idx != -1:
                                relative_path = old_video_file_path[idx:]
                                new_video_file_path = os.path.join(base_path, relative_path)
                                content = re.sub(
                                    r"Video file:\s*.*?(\s{2,}|\n)",
                                    f"Video file: {new_video_file_path}  ",
                                    content
                                )

                        message["content"] = content

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


# 示例调用：
modify_paths(
    "/data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/audio/final_rl_data.jsonl",
    base_path="/data/yiwei.ru/audio_files"
)