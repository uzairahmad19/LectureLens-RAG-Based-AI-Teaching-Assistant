import os,re
import subprocess

def parse_tutorial_filename(file):
    # remove extension
    name = os.path.splitext(file)[0]

    # extract tutorial number
    number_match = re.search(r'(?:Part|Episode|Lesson|Ep)\s*(\d+)', name, re.IGNORECASE)
    tutorial_number = int(number_match.group(1)) if number_match else -1

    # remove common noise
    name = re.sub(
        r'(?:Part|Episode|Lesson|Ep)\s*\d+(?:\s*of\s*\d+)?', '', name, flags=re.IGNORECASE
    )
    name = re.sub(r'\(\d+p\)', '', name)          # remove resolution
    name = re.sub(r'\[.*?\]', '', name)           # remove [brackets]
    name = re.sub(r'\(.*?\)', '', name)           # remove (extra tags)
    name = re.sub(r'-\s*[A-Za-z0-9_.]+\s*$', '', name)  # remove channel names
    name = re.sub(r'\s+', ' ', name).strip()      # normalize spaces

    return tutorial_number, name


files = os.listdir('videos')
for file in files:
    if file.endswith('.mp4'):
        tutorial_number, tutorial_name = parse_tutorial_filename(file)       
        subprocess.run(['ffmpeg', '-i', f'videos/{file}', f'audios/{tutorial_number}_ {tutorial_name}.mp3']) # Extract audio using ffmpeg and save with a cleaned name format