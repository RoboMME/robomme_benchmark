# Public timing and termination rules

This is shared benchmark documentation for both methods, not a task solution.
Read the current episode's language instruction, observation and overall step
limit. The pinned benchmark uses 20 control ticks per second. Each task can end
before its overall horizon when a native success or failure condition is met.
No additional wall time or repeated observation queries extend the control clock.

| Task | Button press deadline |
| --- | --- |
| ButtonUnmask | The required press must register before control step 90. |
| ButtonUnmaskSwap | The required press must register before control step 180. |

The last allowed press steps are 89 and 179 respectively. These are execution
steps; an initial conditioning video does not consume this execution allowance.
ButtonUnmaskSwap's separate 480-step motion limit is not its press deadline.
An early press does not by itself complete the remaining language goal.

All other task-specific success/failure decisions remain with the native
simulator. The public sensor interface provides no hidden object state, oracle
policy, privileged subgoal or evaluator source. Development sees only its allowed
recordings, public interface and budgeted training interactions.
