"""Data-only transport for ASPIRE's existing public contract objects.

No pickle, dynamic imports or simulator objects cross this boundary.
"""
import base64
from collections.abc import Mapping
import dataclasses
from enum import Enum
import json
import zlib

import numpy as np
from cap_harness import contracts, errors, runtime

REPLY_LIMIT = 512 * 1024**2
VIDEO_FRAME_LIMIT = 16 * 1024**2
VIDEO_STREAM = 'observation_sequence_v1'


def _write_record(stream, value, limit):
    data = dumps(value) + b'\n'
    if len(data) > limit:
        raise ValueError('Public tool reply record exceeds transport bound')
    stream.write(data)
    return len(data)


def write_reply(stream, response, *, video=False):
    """Stream video items; preserve the public Observation sequence and values."""
    frames = response.get('value')
    if not video or not isinstance(frames, (list, tuple)):
        size = _write_record(stream, response, REPLY_LIMIT)
        return {'mode': 'single', 'bytes': size}
    size = _write_record(stream, {'__stream__': VIDEO_STREAM, 'count': len(frames)}, VIDEO_FRAME_LIMIT)
    for index, frame in enumerate(frames):
        size += _write_record(stream, {'index': index, 'value': frame}, VIDEO_FRAME_LIMIT)
    size += _write_record(stream, {'end': len(frames)}, VIDEO_FRAME_LIMIT)
    return {'mode': VIDEO_STREAM, 'frames': len(frames), 'bytes': size}


def _read_record(stream, limit):
    data = stream.readline(limit + 1)
    if not data or not data.endswith(b'\n') or len(data) > limit:
        raise ValueError('Incomplete or oversized public tool reply record')
    return loads(data)


def read_reply(stream, *, video=False):
    result = _read_record(stream, REPLY_LIMIT)
    if not isinstance(result, dict):
        raise ValueError('Public tool reply must be an object')
    if '__stream__' not in result:
        return result
    count = result.get('count')
    if (not video or result.get('__stream__') != VIDEO_STREAM
            or type(count) is not int or not 0 <= count <= 100000):
        raise ValueError('Invalid observation sequence header')
    frames = []
    for index in range(count):
        row = _read_record(stream, VIDEO_FRAME_LIMIT)
        if not isinstance(row, dict) or type(row.get('index')) is not int or row['index'] != index or 'value' not in row:
            raise ValueError('Observation sequence is missing or reorders frames')
        if not isinstance(row['value'], contracts.Observation):
            raise ValueError('Observation sequence contains a non-observation')
        frames.append(row['value'])
    if _read_record(stream, VIDEO_FRAME_LIMIT) != {'end': count}:
        raise ValueError('Observation sequence has an invalid end marker')
    # The existing wire conversion returns lists for both list and tuple inputs.
    return {'value': frames}

TYPES = {name: value for module in (contracts, errors, runtime)
         for name, value in vars(module).items()
         if isinstance(value, type) and (dataclasses.is_dataclass(value) or issubclass(value, Enum))}


def encode(value):
    if isinstance(value, Enum):
        return {'__enum__': type(value).__name__, 'value': value.value}
    if dataclasses.is_dataclass(value):
        return {'__type__': type(value).__name__, 'fields':
                {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays are not public sensor data')
        return {'__array__': base64.b64encode(zlib.compress(value.tobytes(), 1)).decode(),
                'dtype': str(value.dtype), 'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported wire value: {type(value).__name__}')


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if '__array__' in value:
        dtype = np.dtype(value['dtype'])
        if dtype.hasobject or dtype.kind not in 'bifu':
            raise ValueError('Unsupported array dtype')
        shape = value['shape']
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid array shape')
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if not 0 <= size <= 256 * 1024**2:
            raise ValueError('Array exceeds transport bound')
        d = zlib.decompressobj()
        raw = d.decompress(base64.b64decode(value['__array__'], validate=True), size + 1)
        if len(raw) != size or not d.eof:
            raise ValueError('Array payload size differs')
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    if '__type__' in value:
        cls = TYPES.get(value['__type__'])
        if cls is None or not dataclasses.is_dataclass(cls):
            raise ValueError('Unrecognized public contract')
        return cls(**{k: decode(v) for k, v in value['fields'].items()})
    if '__enum__' in value:
        cls = TYPES.get(value['__enum__'])
        if cls is None or not issubclass(cls, Enum):
            raise ValueError('Unrecognized enum')
        return cls(value['value'])
    return {k: decode(v) for k, v in value.items()}


def dumps(value):
    return json.dumps(encode(value), allow_nan=False).encode()


def loads(value):
    return decode(json.loads(value))
