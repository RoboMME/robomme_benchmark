num2words = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty"
}

num2words_2 = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth"
}

def get_language_goal(self, env):
    language_goal = ""
    language_goal_2 = ""
    if env == "BinFill":
        color_counts = {
            "red": getattr(self.env.unwrapped, "red_cubes_target_number", 0),
            "blue": getattr(self.env.unwrapped, "blue_cubes_target_number", 0),
            "green": getattr(self.env.unwrapped, "green_cubes_target_number", 0),
        }
        phrases = []
        for color, count in color_counts.items():
            if count <= 0:
                continue
            word = num2words.get(count, str(count))
            noun = "cube" if count == 1 else "cubes"
            phrases.append(f"{word} {color} {noun}")

        if not phrases:
            language_goal = "put the cubes into the bin, then press the button to stop"
            language_goal_2 = "put the cubes into the bin and press the button to stop"
        elif len(phrases) == 1:
            language_goal = f"put {phrases[0]} into the bin, then press the button to stop"
            language_goal_2 = f"put {phrases[0]} into the bin and press the button to stop"
        elif len(phrases) == 2:
            language_goal = f"put {phrases[0]} and {phrases[1]} into the bin, then press the button to stop"
            language_goal_2 = f"put {phrases[0]} and {phrases[1]} into the bin and press the button to stop"
        else:
            language_goal = f"put {', '.join(phrases[:-1])} and {phrases[-1]} into the bin, then press the button to stop"
            language_goal_2 = f"put {', '.join(phrases[:-1])} and {phrases[-1]} into the bin and press the button to stop"

    elif env == "PickXtimes":
        repeats = getattr(self.env.unwrapped, "num_repeats", 1)
        target_color = getattr(self.env.unwrapped, "target_color_name", "unknown")
        if repeats > 1:
            word = num2words.get(repeats, str(repeats))
            language_goal = f"pick up the {target_color} cube and place it on the target, repeating this action {word} times, then press the button to stop"
            language_goal_2 = f"pick up the {target_color} cube and place it on the target, repeating this pick-and-place action {word} times, then press the button to stop"
        else:
            language_goal = f"pick up the {target_color} cube and place it on the target, then press the button to stop"
            language_goal_2 = f"pick up the {target_color} cube and place it on the target, then press the button to stop"

    elif env == "SwingXtimes":
        repeats = getattr(self.env.unwrapped, "num_repeats", 1)
        target_color = getattr(self.env.unwrapped, "target_color_name", "unknown")
        if repeats > 1:
            word = num2words.get(repeats, str(repeats))
            language_goal = f"pick up the {target_color} cube, move it to the top of the right-side target, then move it to the top of the left-side target, repeating this back and forth motion {word} times, finally press the button to stop"
            language_goal_2 = f"pick up the {target_color} cube, move it to the right-side target and then to the left-side target, repeating this right-to-left swing motion {word} times, then put down the cube and press the button to stop"
        else:
            language_goal = f"pick up the {target_color} cube, move it to the top of the right-side target, then put it down on the left-side target, finally press the button to stop"
            language_goal_2 = f"pick up the {target_color} cube, move it to the right-side target and then put it down on the left-side target, then press the button to stop"

    elif env == "VideoUnmask":
        color_names = getattr(self.env.unwrapped, "color_names", ["unknown", "unknown", "unknown"])
        cube_0_color = color_names[0]
        cube_1_color = color_names[1]
        if self.env.unwrapped.configs[self.difficulty]['pick'] > 1:
            language_goal = f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube"
            language_goal_2 = f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube"
        else:
            language_goal = f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube"
            language_goal_2 = f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube"

    elif env == "VideoUnmaskSwap":
        color_names = getattr(self.env.unwrapped, "color_names", ["unknown", "unknown", "unknown"])
        cube_0_color = color_names[0]
        cube_1_color = color_names[1]
        if self.pick_times == 2:
            language_goal = f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube"
            language_goal_2 = f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube"
        else:
            language_goal = f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube"
            language_goal_2 = f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube"

    elif env == "ButtonUnmask":
        color_names = getattr(self.env.unwrapped, "color_names", ["unknown", "unknown", "unknown"])
        cube_0_color = color_names[0]
        cube_1_color = color_names[1]
        if self.env.unwrapped.configs[self.difficulty]['pick'] > 1:
            language_goal = f"first press the button, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube"
            language_goal_2 = f"first press the button, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube"
        else:
            language_goal = f"first press the button, then pick up the container hiding the {cube_0_color} cube"
            language_goal_2 = f"first press the button, then pick up the container hiding the {cube_0_color} cube"

    elif env == "ButtonUnmaskSwap":
        color_names = getattr(self.env.unwrapped, "color_names", ["unknown", "unknown", "unknown"])
        cube_0_color = color_names[0]
        cube_1_color = color_names[1]
        if self.pick_times == 2:
            language_goal = f"first press both buttons on the table, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube"
            language_goal_2 = f"first press both buttons on the table, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube"
        else:
            language_goal = f"first press both buttons on the table, then pick up the container hiding the {cube_0_color} cube"
            language_goal_2 = f"first press both buttons on the table, then pick up the container hiding the {cube_0_color} cube"

    elif env == "VideoPlaceButton":
        target_color_name = self.target_color_name
        target_target_language = self.target_target_language
        language_goal = f"watch the video carefully, then place the {target_color_name} cube on the target right {target_target_language} the button was pressed"
        language_goal_2 = f"watch the video carefully and place the {target_color_name} cube on the target where it was placed immediately {target_target_language} the button was pressed"

    elif env == "VideoPlaceOrder":
        target_color_name = self.target_color_name
        which_in_subset = self.which_in_subset
        num = num2words_2.get(which_in_subset, str(which_in_subset))
        language_goal = f"watch the video carefully, then place the {target_color_name} cube on the {num} target it was previously placed on"
        language_goal_2 = f"watch the video carefully and place the {target_color_name} cube on the {num} target where it was placed"

    elif env == 'PickHighlight':
        language_goal = f'first press the button, then pick up all cubes that have been highlighteted with white areas on the table'
        language_goal_2 = "first press the button, then pick up all highlighted cubes, finally press the button again to stop"

    elif env == "VideoRepick":
        num_repeats = self.num_repeats
        if num_repeats > 1:
            word = num2words.get(num_repeats, str(num_repeats))
            language_goal = f'watch the video carefully, then repeatedly pick up and put down the same block that was previously picked up for {word} times, finally put it down and press the button to stop'
            language_goal_2 = f"watch the video carefully, then pick up the same cube that was previously picked up for {word} times, finally press the button to stop"
        else:
            language_goal = f'watch the video carefully, then pick up the same block that was previously picked up again, finally put it down and press the button to stop'
            language_goal_2 = "watch the video carefully, then pick up the same cube that was previously picked up again, finally press the button to stop"

    elif env == "StopCube":
        repeats = getattr(self.env.unwrapped, "stop_time", 1)
        word = num2words_2.get(repeats, str(repeats))
        language_goal = f"press the button to stop the cube just as it reaches the target for the {word} time"
        language_goal_2 = f"press the button to stop the cube exactly at the target on its {word} visit"

    elif env == "InsertPeg":
        language_goal = f"watch the video carefully, then grasp the same end of the same peg you've picked before and insert it into the same side of the box"
        language_goal_2 = "watch the video carefully, then grasp the same peg at the same end and insert it into the same side of the box as in the video"

    elif env == "MoveCube":
        language_goal = "watch the video carefully, then move the cube to the target in the same manner as before"
        language_goal_2 = "watch the video carefully, then move the cube to the target in the same manner shown in the video"

    elif env == "PatternLock":
        language_goal = "Watch the video carefully, then use the stick attached to the robot to retrace the same pattern"
        language_goal_2 = "watch the video carefully, then use the stick attached to the robot to retrace the same pattern shown in the video"

    elif env == "RouteStick":
        language_goal = "Watch the video carefully, then use the stick attached to the robot to navigate around the sticks on the table, following the same path"
        language_goal_2 = "watch the video carefully, then use the stick attached to the robot to navigate around the sticks on the table, following the same path shown in the video"

    language_goal = language_goal or ""
    language_goal_2 = language_goal_2 or ""
    if language_goal == language_goal_2:
        return [language_goal]
    return [language_goal, language_goal_2]
