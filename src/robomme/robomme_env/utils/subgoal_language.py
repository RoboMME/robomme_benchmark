from ...logging_utils import logger


def get_subgoal_with_index(idx, template, **kwargs):
    if idx == 0:
        subgoal = template.format(idx="first", **kwargs)
    elif idx == 1:
        subgoal = template.format(idx="second", **kwargs)
    elif idx == 2:
        subgoal = template.format(idx="third", **kwargs)
    elif idx == 3:
        subgoal = template.format(idx="fourth", **kwargs)
    elif idx == 4:
        subgoal = template.format(idx="fifth", **kwargs)
    elif idx == 5:
        subgoal = template.format(idx="sixth", **kwargs)
    elif idx == 6:
        subgoal = template.format(idx="seventh", **kwargs)
    elif idx == 7:
        subgoal = template.format(idx="eighth", **kwargs)
    elif idx == 8:
        subgoal = template.format(idx="ninth", **kwargs)
    elif idx == 9:
        subgoal = template.format(idx="tenth", **kwargs)
    else:
        raise ValueError(f"Invalid index: {idx}")
    return subgoal



if __name__ == "__main__":
    logger.debug(get_subgoal_with_index(0, "pick up the {idx} {color} cube", color="red"))
