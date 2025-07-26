import re
from PIL import Image
from io import BytesIO

def process_vision_info(conversation):
    images = []
    videos = []
    for turn in conversation:
        if turn['role'] == 'user':
            for content in turn['content']:
                if content['type'] == 'image':
                    images.append(content['image'])
                elif content['type'] == 'video':
                    videos.append(content['video'])
    return images, videos
