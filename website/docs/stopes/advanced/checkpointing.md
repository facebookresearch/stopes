---
sidebar_position: 1
---

# Checkpointing (advanced)

When using SLURM, the StopesModule system uses submitit to schedule the jobs.
This means that you can leverage the checkpointing feature it offers. This
allows you to store the state of the current module when its job gets preempted
or times out. See the [submitit
doc](https://github.com/facebookincubator/submitit/blob/main/docs/checkpointing.md)
for more details.
