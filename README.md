[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# OmniByteFormer


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


OmniByteFormer is a generalized Transformer model that can process any type of data by converting it into byte sequences, bypassing traditional tokenization or specific data-type encodings. Whether the input is text, images, videos, audio, or other data formats, OmniByteFormer treats all data uniformly as bytes, and generates the output directly in bytes. This makes OmniByteFormer a flexible and universal model for multi-modal tasks.

## Key Features

- **Universal Input**: Accepts various data types (text, image, audio, video, etc.) by converting them into byte sequences.
- **Transformer-Based Architecture**: Uses the power of Transformer models for generative tasks with arbitrary data.
- **Byte-Level Processing**: Instead of tokenizing or using modality-specific encodings, it processes byte sequences directly, offering a uniform representation for all data types.
- **Multi-Modal Compatibility**: Can be trained to generate text, images, videos, or even sound as output from different types of input data.

## Architecture

OmniByteFormer is built on a byte-level Transformer architecture. The core model leverages the following components:

- **Byte Embeddings**: Converts each byte (0-255) into a learnable embedding vector.
- **Transformer Encoder-Decoder**: Applies self-attention and cross-attention mechanisms on byte sequences, enabling the model to learn representations across different modalities.
- **Positional Encoding**: Ensures the model retains sequence information by encoding position in byte sequences.
- **Universal Decoder**: Outputs the byte sequences, which can be converted back into the original data types (text, image, video, etc.).


### Training the Model

```bash
python train_byte_transformer.py --epochs 10 --batch_size 32 --lr 1e-4 --seq_len 128 --save_path ./checkpoints
```


## Example


### Converting Outputs Back to Data

After generating byte sequences with OmniByteFormer, you can convert the byte outputs back into their original format (text, image, etc.). You’ll need to decode these bytes appropriately based on your task (e.g., text decoding or saving image files).

## Examples

- **Text Generation**: Convert text to bytes and generate text from bytes.
- **Image-to-Image Generation**: Convert images to bytes, pass through the model, and generate new images.
- **Audio/Video Processing**: Work with audio or video data by converting them to byte sequences.

## Contributing

Feel free to contribute to this project. Fork the repo, make changes, and submit a pull request.

## License

This project is licensed under the MIT License.
