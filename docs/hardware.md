# TORO integration

The public offline workflow plans footholds and replays saved plans locally.
The real-world experiment additionally used the DLR TORO motion-planning,
whole-body-control, and communication stack cited in the paper. Those systems
are not installed or recreated by this repository.

The original `main_pump_TORO_optapi.py` and `src/run_main_DDR_sliding_TORO.py`
are preserved for integration reference. They contain historical robot host/port
settings and use `src.utils.udp` to serialize the footstep sequence. UDP sends
return without creating a socket unless `FOOTSTEP_ENABLE_ROBOT_UDP=1` is set.
An operator must configure the correct receiver and external controller before
enabling that mode. This setting is not needed for any README example.

Use `python -m experiments.pump_deadlines run --smoke-test` for offline Pump
planning comparisons. It does not import the network runtime entry point.

The MuJoCo viewer drives an analytical inverse-kinematics pose sequence and
does not validate dynamic stability or contact forces. Its plan input is
`plan.json` produced by the GCS demo; the initial two foot poses must contain
one left and one right foot, and yaw is expressed in degrees.
