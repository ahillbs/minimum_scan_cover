This is the code repository for minimum scan cover.

All implemented solvers are available in the solver folder.
Data will be stored in a sqlite database.
Currently, no other database is supported but can easily be implemented.

For testing, several python programs are available.

To create random instances three programs are available:
- create_cel_instances_script.py create random celestial instances.
- create_gen_instances_script.py generates random point 2D instances.
- instance_evolve.py generated instances and evolves to a goal, like a maximal runtime for a solver.
- instance_evolve_greedy.py generates instances where the performance relative to the lower bound will be maximized.
For more information about the parameter, see task_config folder and help parameter.

Instance tests can be created with instance_tests.py.
Note, that it also tests the instances if not stated otherwise.
For the tests to work, the celeryconfig.py needs to be filled.

Tasks can also be processed by usting task_processor.py.
There, it is also possible to locally perform the tasks (not tested and only for instance tests).

revisit_error_instances.py can be used to debug instances that threw error messages.

reset_task.py can be used to reset tasks and subtasks and can also reset task jobs.