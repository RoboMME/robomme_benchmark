import h5py
import sys

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

def print_recursive(obj, indent=0):
    tab = "  " * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{tab}- [Dataset] {obj.name.split('/')[-1]}: shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{tab}+ [Group] {obj.name.split('/')[-1]}")
        
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

DEFAULT_PATH = "/data/hongzefu/dataset_generate-b4/record_dataset_PickXtimes.h5"

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
