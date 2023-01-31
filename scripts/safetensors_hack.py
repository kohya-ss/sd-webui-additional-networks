import io
import os
import mmap
import torch
import json
import hashlib
import safetensors
import safetensors.torch

from modules import sd_models

# PyTorch 1.13 and later have _UntypedStorage renamed to UntypedStorage
UntypedStorage = torch.storage.UntypedStorage if hasattr(torch.storage, 'UntypedStorage') else torch.storage._UntypedStorage

def read_metadata(filename):
    """Reads the JSON metadata from a .safetensors file"""
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    return metadata.get("__metadata__", {})


def load_file(filename, device):
    """"Loads a .safetensors file without memory mapping that locks the model file.
    Works around safetensors issue: https://github.com/huggingface/safetensors/issues/164"""
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    size = os.stat(filename).st_size
    storage = UntypedStorage.from_file(filename, False, size)
    offset = n + 8
    md = metadata.get("__metadata__", {})
    return {name: create_tensor(storage, info, offset) for name, info in metadata.items() if name != "__metadata__"}, md


def hash_file(filename):
    """Hashes a .safetensors file using the new hashing method.
    Only hashes the weights of the model."""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")

    with open(filename, mode="rb") as file_obj:
        offset = n + 8
        file_obj.seek(offset)
        for chunk in iter(lambda: file_obj.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def legacy_hash_file(filename):
    """Hashes a model file using the legacy `sd_models.model_hash()` method."""
    hash_sha256 = hashlib.sha256()

    metadata = read_metadata(filename)

    # For compatibility with legacy models: This replicates the behavior of
    # sd_models.model_hash as if there were no user-specified metadata in the
    # .safetensors file. That leaves the training parameters, which are
    # immutable. It is important the hash does not include the embedded user
    # metadata as that would mean the hash could change every time the user
    # updates the name/description/etc. The new hashing method fixes this
    # problem by only hashing the region of the file containing the tensors.
    if any(not k.startswith("ss_") for k in metadata):
      # Strip the user metadata, re-serialize the file as if it were freshly
      # created from sd-scripts, and hash that with model_hash's behavior.
      tensors, metadata = load_file(filename, "cpu")
      metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}
      model_bytes = safetensors.torch.save(tensors, metadata)

      hash_sha256.update(model_bytes[0x100000:0x110000])
      return hash_sha256.hexdigest()[0:8]
    else:
      # This should work fine with model_hash since when the legacy hashing
      # method was being used the user metadata system hadn't been implemented
      # yet.
      return sd_models.model_hash(filename)


DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    # "U64": torch.uint64,
    "I32": torch.int32,
    # "U32": torch.uint32,
    "I16": torch.int16,
    # "U16": torch.uint16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool
}


def create_tensor(storage, info, offset):
    """Creates a tensor without holding on to an open handle to the parent model
    file."""
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return torch.asarray(storage[start + offset : stop + offset], dtype=torch.uint8).view(dtype=dtype).reshape(shape).clone().detach()
