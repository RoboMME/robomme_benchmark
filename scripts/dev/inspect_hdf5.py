import h5py
import sys
import numpy as np

def _decode_h5_object(value):
    """Decode HDF5 object dtype (vlen str) to Python str for display."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = np.reshape(value, -1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return value.decode("utf-8")
        except Exception:
            return repr(value)
    if isinstance(value, str):
        return value
    return str(value)


def print_hdf5_structure(name, obj, indent=0):
    """
    Recursively print the structure of an HDF5 group or dataset.
    Only prints one 'episode_' and one 'timestep_' per level to avoid cluttering.
    """
    base_name = name.split('/')[-1]
    
    # Check if we should skip this item to limit to one episode/timestep
    parent_path = '/'.join(name.split('/')[:-1])
    
    # This logic is a bit tricky inside visititems since it's flat traversal normally
    # But we can implement a custom recursive function instead
    pass

def _format_value(obj, max_elems=20, max_str_len=200, max_array_size=10000):
    """Read dataset and format for display; handle scalars and arrays."""
    try:
        shape = obj.shape
        size = int(np.prod(shape)) if shape else 0
        if size > max_array_size:
            # 大数组：只读前 max_elems 个元素（按 C-order 展平）
            take = min(max_elems, size)
            if take == 0:
                return "[]"
            idx = np.unravel_index(take - 1, shape)
            slice_tuple = tuple(slice(0, int(i) + 1) for i in idx)
            raw = obj[slice_tuple]
            flat = np.asarray(raw).reshape(-1)[:take]
            n = len(flat)
            total = size
        else:
            raw = obj[()]
            if raw is None:
                return "None"
            if obj.shape == () or np.isscalar(raw):
                out = _decode_h5_object(raw)
                if out is None:
                    out = str(raw)
                if isinstance(out, str) and len(out) > max_str_len:
                    out = out[:max_str_len] + "..."
                return out
            arr = np.asarray(raw)
            flat = np.reshape(arr, -1)
            n = min(flat.size, max_elems)
            total = flat.size
    except Exception as e:
        return f"(read error: {e})"

    if n == 0:
        return "[]"
    parts = []
    for i in range(n):
        v = flat.flat[i]
        if isinstance(v, (bytes, np.bytes_)):
            try:
                v = v.decode("utf-8")
            except Exception:
                v = repr(v)
        parts.append(str(v))
    s = "[" + ", ".join(parts) + "]"
    if total > max_elems:
        s += f" ... ({total} total)"
    return s


def print_recursive(obj, indent=0):
    tab = "  " * indent
    if isinstance(obj, h5py.Dataset):
        name = (obj.name or "").split("/")[-1]
        print(f"{tab}- [Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
        # 打印值：标量、小数组或数组摘要
        value_str = _format_value(obj)
        if value_str:
            print(f"{tab}    -> {value_str}")
    elif isinstance(obj, h5py.Group):
        print(f"{tab}+ [Group] {(obj.name or '').split('/')[-1]}")
        
        # Sort items: groups first, then datasets? Or just as is.
        # Filter items to only show one episode_* or timestep_*
        
        items = list(obj.items())
        
        shown_episode = False
        shown_timestep = False
        
        for name, item in items:
            is_episode = name.startswith('episode_')
            is_timestep = name.startswith('timestep_')
            
            if is_episode:
                if not shown_episode:
                    print_recursive(item, indent + 1)
                    shown_episode = True
                continue
            
            if is_timestep:
                if not shown_timestep:
                    print_recursive(item, indent + 1)
                    shown_timestep = True
                continue
                
            # Regular items (meta, obs, action, info etc)
            print_recursive(item, indent + 1)

DEFAULT_PATH = "/data/hongzefu/data_0226/record_dataset_SwingXtimes.h5"

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print(f"Inspecting HDF5 file: {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # The root itself if it has a name (usually empty string or '/')
            print("/")
            
            items = list(f.items())
            shown_episode = False
            
            for name, item in items:
                if name.startswith('episode_'):
                    if not shown_episode:
                        print_recursive(item, 1)
                        shown_episode = True
                    continue
                print_recursive(item, 1)
                
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")

if __name__ == "__main__":
    main()
