import torch
import torch.nn as nn
import math
from torch.nn import Transformer

# Define EOS token (value 255 for end-of-sequence)
EOS_TOKEN = 255

# Model definition with PositionalEncoding
class ByteTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(ByteTransformer, self).__init__()

        self.byte_embedding = nn.Embedding(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(
            embed_dim, dropout
        )
        self.transformer = Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor = None
    ) -> torch.Tensor:
        if tgt is None:
            return self.generate(src)

        src_embedded = self.byte_embedding(src) * math.sqrt(
            self.byte_embedding.embedding_dim
        )
        tgt_embedded = self.byte_embedding(tgt) * math.sqrt(
            self.byte_embedding.embedding_dim
        )

        src_embedded = self.positional_encoding(src_embedded)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        output = self.transformer(src_embedded, tgt_embedded)
        output = self.output_layer(output)

        return output

    def generate(self, src: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        src_embedded = self.byte_embedding(src) * math.sqrt(self.byte_embedding.embedding_dim)
        src_embedded = self.positional_encoding(src_embedded)

        generated_seq = src
        for _ in range(max_len):
            tgt_embedded = self.byte_embedding(generated_seq) * math.sqrt(self.byte_embedding.embedding_dim)
            tgt_embedded = self.positional_encoding(tgt_embedded)

            output = self.transformer(src_embedded, tgt_embedded)
            next_token_logits = self.output_layer(output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Stop generating if EOS token is produced
            if next_token.item() == EOS_TOKEN:
                break

            generated_seq = torch.cat([generated_seq, next_token.unsqueeze(1)], dim=1)

        return generated_seq


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# Byteify function to convert text to byte sequence
def byteify(data: str) -> bytes:
    return data.encode("utf-8")


# Convert bytes to tensor
def bytes_to_tensor(byte_data: bytes) -> torch.Tensor:
    byte_list = list(
        byte_data
    )  # Convert bytes to list of integers (0-255)
    return torch.tensor(byte_list, dtype=torch.long).unsqueeze(
        0
    )  # Shape: [1, seq_len]


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Converts a tensor of byte values (0-255) to a bytes object.

    Args:
        tensor (torch.Tensor): A tensor of shape [batch_size, seq_len] where each element is a byte.

    Returns:
        bytes: A bytes object that can be decoded to a string.
    """
    # Convert tensor to a list of byte values
    byte_list = tensor.squeeze(
        0
    ).tolist()  # Remove batch dimension if present
    return bytes(byte_list)


def bytes_to_text(byte_data: bytes) -> str:
    """
    Decodes bytes back into a UTF-8 string.

    Args:
        byte_data (bytes): The byte sequence to decode.

    Returns:
        str: The decoded text.
    """
    return byte_data.decode(
        "utf-8", errors="ignore"
    )  # Use 'ignore' to avoid decoding errors


# # Example usage with model output
# if __name__ == "__main__":
#     # Example text that was processed by the model
#     example_text = "Hello, ByteTransformer! What is your name?"

#     # Byteify the text data
#     byte_data = byteify(example_text)

#     # Convert byte data to tensor
#     byte_tensor = bytes_to_tensor(byte_data)

#     # Add EOS token to the input tensor
#     byte_tensor = torch.cat([byte_tensor, torch.tensor([[EOS_TOKEN]], dtype=torch.long)], dim=1)

#     # Instantiate the ByteTransformer model
#     model = ByteTransformer(
#         input_dim=256,
#         embed_dim=512,
#         num_heads=8,
#         num_layers=6,
#         dim_feedforward=2048,
#         dropout=0.1,
#     )

#     # Perform inference (generation) using the byte tensor
#     generated_output = model(byte_tensor)

#     # Decode model output to bytes
#     generated_bytes = tensor_to_bytes(generated_output)

#     # Truncate the bytes at the EOS token
#     if EOS_TOKEN in generated_bytes:
#         generated_bytes = generated_bytes[:generated_bytes.index(EOS_TOKEN)]

#     # Decode the bytes to text
#     decoded_text = bytes_to_text(generated_bytes)

#     # Print the decoded text
#     print("Decoded text:", decoded_text)
