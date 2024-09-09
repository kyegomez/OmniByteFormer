import io
import mimetypes
import os
import wave
from typing import Union

# import moviepy.editor as mp
from PIL import Image


def byteify(data: Union[str, Image.Image]) -> bytes:
    """
    Converts various data types (text, image, audio, video) to a byte representation.
    Automatically detects the data type based on the file extension or input type.

    Args:
        data (Union[str, Image.Image]): The data to be byteified. This can be a file path or PIL Image.

    Returns:
        bytes: The byte representation of the data.

    Raises:
        ValueError: If the file type is unsupported or data cannot be processed.
    """
    if isinstance(data, Image.Image):
        return _image_to_bytes(data)

    if isinstance(data, str):
        if not os.path.exists(data):
            raise ValueError(f"File not found: {data}")

        mime_type, _ = mimetypes.guess_type(data)

        if mime_type is None:
            raise ValueError(
                f"Cannot determine file type for: {data}"
            )

        if mime_type.startswith("text"):
            return _text_to_bytes(data)
        elif mime_type.startswith("image"):
            return _imagefile_to_bytes(data)
        elif mime_type.startswith("audio"):
            return _audiofile_to_bytes(data)
        # elif mime_type.startswith("video"):
        #     return _videofile_to_bytes(data)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

    raise ValueError(
        "Unsupported data type. Only file paths or PIL Image objects are supported."
    )


# Helper functions for specific file types
def _text_to_bytes(filepath: str) -> bytes:
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read().encode("utf-8")


def _imagefile_to_bytes(filepath: str) -> bytes:
    with Image.open(filepath) as img:
        return _image_to_bytes(img)


def _image_to_bytes(image: Image.Image) -> bytes:
    with io.BytesIO() as byte_io:
        image.save(byte_io, format="PNG")
        return byte_io.getvalue()


def _audiofile_to_bytes(filepath: str) -> bytes:
    with wave.open(filepath, "rb") as audio_file:
        return audio_file.readframes(audio_file.getnframes())


# def _videofile_to_bytes(filepath: str) -> bytes:
#     video = mp.VideoFileClip(filepath)
#     with io.BytesIO() as byte_io:
#         video.write_videofile(byte_io, codec="libx264", audio=False)
#         return byte_io.getvalue()


def decode_bytes(
    byte_data: bytes, output_type: str, output_path: str = None
) -> Union[str, Image.Image, None]:
    """
    Decodes byte sequences back into their original data forms, such as text, image, audio, or video.

    Args:
        byte_data (bytes): The byte sequence to decode.
        output_type (str): The type of output to generate. Expected values: 'text', 'image', 'audio', 'video'.
        output_path (str): The file path to save the decoded output (for images, audio, and video).

    Returns:
        Union[str, Image.Image, None]: Returns decoded data. For text, returns a string.
                                        For images, returns a PIL Image object.
                                        For audio and video, saves the output to a file and returns None.

    Raises:
        ValueError: If the output type is unsupported or there is an error in decoding.
    """
    if output_type == "text":
        return _bytes_to_text(byte_data)

    elif output_type == "image":
        return _bytes_to_image(byte_data, output_path)

    elif output_type == "audio":
        _bytes_to_audio(byte_data, output_path)
        return None  # Audio will be saved to output_path

    elif output_type == "video":
        _bytes_to_video(byte_data, output_path)
        return None  # Video will be saved to output_path

    else:
        raise ValueError(
            f"Unsupported output type: {output_type}. Supported types: 'text', 'image', 'audio', 'video'."
        )


def _bytes_to_text(byte_data: bytes) -> str:
    """Decodes bytes to a UTF-8 encoded string."""
    return byte_data.decode("utf-8")


def _bytes_to_image(
    byte_data: bytes, output_path: str = None
) -> Image.Image:
    """Decodes bytes to an image and optionally saves it to a file."""
    image = Image.open(io.BytesIO(byte_data))

    if output_path:
        image.save(output_path)

    return image


def _bytes_to_audio(byte_data: bytes, output_path: str):
    """Decodes bytes to a WAV audio file and saves it."""
    if not output_path.endswith(".wav"):
        raise ValueError(
            "Output path for audio must be a '.wav' file."
        )

    with wave.open(output_path, "wb") as audio_file:
        audio_file.setnchannels(2)  # Assuming stereo
        audio_file.setsampwidth(2)  # 16-bit per sample
        audio_file.setframerate(44100)  # Standard sample rate
        audio_file.writeframes(byte_data)


def _bytes_to_video(byte_data: bytes, output_path: str):
    """Decodes bytes to a video file and saves it."""
    if not output_path.endswith((".mp4", ".avi", ".mkv")):
        raise ValueError(
            "Output path for video must have a video file extension (e.g., .mp4, .avi, .mkv)."
        )

    # Save byte stream to video file (this is a placeholder, actual implementation will depend on format)
    with open(output_path, "wb") as video_file:
        video_file.write(byte_data)
