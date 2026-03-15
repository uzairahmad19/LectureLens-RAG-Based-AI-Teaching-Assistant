''' This code uses the faster-whisper library instead of the original OpenAI Whisper.
The faster-whisper library is optimized for performance and can run on lower-end hardware, making it a great choice for speech-to-text tasks on devices with limited resources '''

import json,os

from faster_whisper import WhisperModel

# Create the Whisper model
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

audios = os.listdir('audios')
for audio in audios:
    if "_" in audio:
        name = audio.split('_')[1][:-4]
        number = audio.split('_')[0]
        segments, info = model.transcribe( # Transcribe the audio file and get the segments and info
            f"audios/{audio}",
            task="translate"
        )

        chunks = []
        full_text = ""
        for segment in segments: # Iterate through the segments and create a chunk for each segment
            chunks.append({
                "number": number,
                "name": name,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })
            full_text += segment.text.strip() + " "

        chunks_with_metadata = { # Create a dictionary with the chunks and the full text
            "chunk": chunks,
            "text": full_text.strip(),
        }

        # Save the chunks with metadata to a JSON file
        with open(f'transcripts/{number}_{name}.json', 'w') as f:
            json.dump(chunks_with_metadata, f)

