# Command Line Interface (CLI)
## Argument Format
py main.py [visualise_cnt=<value>] [db=<value>] [folds=<value>] [seed=<value>]

### (1) visualise_cnt
Type: integer

Description: number of trees you want to visualise, starting from the 1st, capped at the maximum of trees available (no error is raised if you exceed the maximum)

### (2) db
Type: filepath

Description: a relative filepath to this program's current working directory which must have the format shown below.

#### Dataset Row Format
64	-56	-61	-66	-71	-82	-81	1

Where, the last element ('1' in the above example), is the class label.

### (3) folds
Type: integer

Description: number of folds to be used when performing k-cross validation.

### (4) seed
Type: integer

Description: Random Number Generator seed value.
