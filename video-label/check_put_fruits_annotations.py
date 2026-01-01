import os
import json
import re

def check_annotations(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    results = {
        "passed": [],
        "failed": [],
        "skipped": []
    }

    print(f"Checking {len(files)} files in {directory}...\n")

    for filename in files:
        filepath = os.path.join(directory, filename)
        
        # Parse filename for expected fruit count
        # Pattern: Put_N_fruit...
        match = re.search(r'Put_(\d+)_fruit', filename)
        if not match:
            print(f"SKIPPED: {filename} - Could not parse fruit count from filename")
            results["skipped"].append(filename)
            continue
            
        expected_count = int(match.group(1))
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if "keyframes" not in data:
                print(f"FAILED: {filename} - No 'keyframes' key found")
                results["failed"].append((filename, "No 'keyframes' key"))
                continue
                
            keyframes = data["keyframes"]
            
            # Sort frames numerically
            sorted_frames = sorted(keyframes.keys(), key=lambda x: int(x))
            
            actions = []
            for frame in sorted_frames:
                actions.append(keyframes[frame]["option"])
                
            # validation logic
            errors = []
            
            # 1. Check counts
            pickup_count = actions.count("pickup fruit")
            putdown_count = actions.count("putdown fruit")
            press_button_count = actions.count("press button")
            
            if pickup_count != expected_count:
                errors.append(f"Expected {expected_count} 'pickup fruit', found {pickup_count}")
            
            if putdown_count != expected_count:
                errors.append(f"Expected {expected_count} 'putdown fruit', found {putdown_count}")
                
            if press_button_count < 1:
                errors.append("Missing 'press button'")
            
            # 2. Check strict order (pickup -> putdown -> pickup -> putdown ... -> press button)
            # Filter out irrelevant actions if any (though instructions imply strict structure)
            relevant_actions = [a for a in actions if a in ["pickup fruit", "putdown fruit", "press button"]]
            
            expected_sequence = []
            for _ in range(expected_count):
                expected_sequence.append("pickup fruit")
                expected_sequence.append("putdown fruit")
            expected_sequence.append("press button")
            
            # It's possible there are multiple press buttons or noise, but requirement says "Last must be press button"
            # and "Must strictly correspond to count". 
            # Let's check the core sequence.
            
            # The prompt says: "Last must be a press button"
            if relevant_actions and relevant_actions[-1] != "press button":
                 errors.append("Last action is not 'press button'")
            
            # Check strict alternation of pickup/putdown
            current_state = "putdown" # Expect pickup next
            seq_error = False
            
            processed_pickups = 0
            processed_putdowns = 0
            
            for action in relevant_actions:
                if action == "pickup fruit":
                    if current_state != "putdown":
                        if not seq_error:
                            errors.append(f"Sequence error: Found 'pickup fruit' but expected 'putdown fruit' (Duplicate pickup?)")
                            seq_error = True
                    current_state = "pickup"
                    processed_pickups += 1
                elif action == "putdown fruit":
                    if current_state != "pickup":
                        if not seq_error:
                            errors.append(f"Sequence error: Found 'putdown fruit' but expected 'pickup fruit' (Duplicate putdown?)")
                            seq_error = True
                    current_state = "putdown"
                    processed_putdowns += 1
                elif action == "press button":
                    # Press button can happen at the end. 
                    # If we haven't finished the sequence, it might be an error if strictly interpreted, 
                    # but we already checked counts. 
                    pass

            if not errors:
                results["passed"].append(filename)
            else:
                print(f"FAILED: {filename}")
                for err in errors:
                    print(f"  - {err}")
                results["failed"].append((filename, errors))
                
        except json.JSONDecodeError:
            print(f"FAILED: {filename} - Invalid JSON")
            results["failed"].append((filename, "Invalid JSON"))
        except Exception as e:
            print(f"FAILED: {filename} - Error: {str(e)}")
            results["failed"].append((filename, str(e)))

    print("\n" + "="*50)
    print(f"Summary:")
    print(f"Total Files: {len(files)}")
    print(f"Passed: {len(results['passed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print("="*50)

if __name__ == "__main__":
    check_annotations("/data/hongzefu/historybench_real_dataset/annotation_results/PutFruits")

