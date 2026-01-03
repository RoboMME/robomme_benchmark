

GROUNDED_SUBGOAL_INFORMATION = """
Note that the returned subgoal should have grounding information represented by a point to the corresponding object in the image.  The point is the coordinate of the corresponding object in the image, normalized to 0-1000. The expression should be like "do something to the object at <y, x>".

If the input is a video, always use the last frame to predict the point location of the object.
"""


SYSTEM_PROMPT = """You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of language subgoals.

Possible subgoals:{subgoals}

Example:{example}

Output Format:
If the input is the first image frame, the output should be:
```json
{{
    "subgoal_sequence": "you need to decompose the overall task goal into a sequence of subgoals, e.g. 1, 2, 3, ...",
    "subgoal": "predict the first subgoal"
}}
```

If the input is ongoing video clips, the output should be:
```json
{{
    "description": "describe what the robot has done in the video clips, the robot may not do anything or incomplete due to the clip is too short",
    "subgoal": "predict the next subgoal, if you think the robot has not yet completed the last subgoal, you should output the same last subgoal again, e.g. the robot is approaching the cube but has not yet picked it up, you should continue to pick up"
}}
```
"""



SYSTEM_PROMPT_with_DEMO = """You are assisting with predicting the next language subgoal for a robot.

Possible subgoals:{subgoals}

Example:{example}

Output Format:
If the input is a pre-recorded video, the output should be:
```json
{{
    "subgoal_sequence": "according to the robot actions in the pre-recorded video, carefully analyze the robot movements and what it has done, then summarize the task goal into a sequence of subgoals into 1, 2, 3, ...",
    "subgoal": "predict the first subgoal"
}}
```

If the input is ongoing new video clips showing the robot current execution, the output should be:
```json
{{
    "description": "describe what the robot has done in the video clips of the robot execution, the robot may not do anything or incomplete due to the clip is too short",
    "subgoal": "predict the next subgoal, if you think the robot has not yet completed the last subgoal, you should output the same last subgoal again"
}}
```
"""


SYSTEM_PROMPT_DYNAMIC_CHANGE = """You are assisting with predicting the next language subgoal for a robot.

Possible subgoals:{subgoals}

Example:{example}

Output Format:
The output should be:
```json
{{
    "description": "describe what the robot has done in the video clips or image. You need observe closely what the robot has done and once the button pressing is all finished, then decompose the task goal into a sequence of subgoals",
    "subgoal": "predict the next subgoal, if you think the robot has not yet completed the last subgoal, you should output the same last subgoal again, e.g. the robot is approaching the cube but has not yet picked it up, you should continue to pick up, if the button pressing is not finished, you should continue to press the button rather than predict the next subgoal"
}}
```
"""




DEMO_TEXT_QUERY = "The task goal is '{task_goal}'. Given this pre-recorded video, please analyze the video and summarize the robot's actions in the video as a sequential list of subgoals, then predict the first subgoal"

DEMO_TEXT_QUERY_multi_image = "The task goal is '{task_goal}'. Given this pre-recorded video (the video is represented by multiple images instead of a video file), please analyze the video and summarize the robot's actions in the video as a sequential list of subgoals, then predict the first subgoal"

IMAGE_TEXT_QUERY = "The task goal is '{task_goal}'. Given this first input image, please decompose the task goal into a sequence of subgoals and then predict the first subgoal"

VIDEO_TEXT_QUERY = "Given current new video clip of the robot execution, please describe what the robot has done in the video, then predict the next subgoal"

VIDEO_TEXT_QUERY_multi_image = "Given current new video clip of the robot execution (the video clip is represented by multiple images instead of a video file), please describe what the robot has done in the video, then predict the next subgoal"





ORACLE_MORE_EXPLANATION = """
The point is a location of the corresponding object in the image, normalized to 0-1000.
If the input is a video, always use the last frame to predict the point location of the object.
"""


SYSTEM_PROMPT_ORACLE_PLANNER = """You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of language subgoals.

Possible subgoals:{subgoals}

Example:{example}

Output Format:
If the input is the first image frame, the output should be:
```json
{{
    "subgoal_sequence": "you need to decompose the overall task goal into a sequence of subgoals, e.g. 1, 2, 3, ...",
    "subgoal": {{"action": "predict the first subgoal", "point": [y, x] if the subgoal has grounding information, otherwise None}}
}}
```

If the input is ongoing video clips, the output should be:
```json
{{
    "description": "describe what the robot has done in the video clips",
    "subgoal": {{"action":"predict the next subgoal", "point": [y, x] if the subgoal has grounding information, otherwise None}}
}}
```
""" + ORACLE_MORE_EXPLANATION



SYSTEM_PROMPT_with_DEMO_ORACLE_PLANNER = """You are assisting with predicting the next language subgoal for a robot.

Possible subgoals:{subgoals}

Example:{example}

Output Format:
If the input is a pre-recorded video, the output should be:
```json
{{
    "subgoal_sequence": "according to the robot actions in the pre-recorded video, carefully analyze the robot movements and what it has done, then summarize the task goal into a sequence of subgoals into 1, 2, 3, ...",
    "subgoal": {{"action":"predict the first subgoal", "point": [y, x] if the subgoal has grounding information, otherwise None}}
}}
```

If the input is ongoing new video clips showing the robot current execution, the output should be:
```json
{{
    "description": "describe what the robot has done in the video clips of the robot execution",
    "subgoal": {{"action":"predict the next subgoal", "point": [y, x] if the subgoal has grounding information, otherwise None}}
}}
```
""" + ORACLE_MORE_EXPLANATION      


SYSTEM_PROMPT_DYNAMIC_CHANGE_ORACLE_PLANNER = """You are assisting with predicting the next language subgoal for a robot.

Possible subgoals:{subgoals}

Example:{example}

Output Format:
The output should be:
```json
{{
    "description": "describe what the robot has done in the video clips or image. You need observe closely what the robot has done and once the button pressing is all finished, then decompose the task goal into a sequence of subgoals",
    "subgoal": {{"action":"predict the next subgoal", "point": [y, x]}}
}}
```
""" + ORACLE_MORE_EXPLANATION

# PatternLock 专用（无定位信息，point 恒为 null）
SYSTEM_PROMPT_PATTERNLOCK_ORACLE = """You are assisting with predicting the next language subgoal for a robot.

Possible subgoals:{subgoals}

Example:{example}

Output Format:
If the input is a pre-recorded video, the output should be:
```json
{{
    "subgoal_sequence": "According to the pre-recorded video, the robot has travelled to X diffrent white and gray disks, thus there are X subgoals. The task subgoal sequence could be: 1, 2, 3, ...",
    "subgoal": {{"action":"predict the first subgoal", "point": null}}
}}
```

If the input is ongoing new video clips showing the robot current execution, the output should be:
```json
{{
    "description": "According to the pre-recorded video and the pre-planned subgoal sequence, the robot need to travel X subgoals, and it has already travelled Y subgoals, thus the next subgoal is the (Y+1)th subgoal. Thus the next subgoal is ...",
    "subgoal": {{"action":"predict the next subgoal", "point": null}}
}}
```

Notes:
- PatternLock does not require grounding; always set "point" to null.
- Keep the JSON structure strictly consistent with the above.
- When outputing the subgoal sequence, you need to first think about how many subgoals are there, then output the subgoal sequence accordingly. Be sure to follow this reasoning process.
- There could not be more than 10 subgoals! Be sure to follow this rule.
"""