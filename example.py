from omni_byte_former.model import ByteTransformer
from omni_byte_former.utils import byteify

model = ByteTransformer(
    input_dim=256,  # Byte values are 0-255
    embed_dim=512,  # Embedding dimension
    num_heads=8,  # Number of attention heads
    num_layers=6,  # Number of Transformer layers
    dim_feedforward=2048,  # Feedforward network size
)

# Example usage: byteify and pass through model
input_data = byteify("hello.txt")
output = model(input_data)
print(output)
