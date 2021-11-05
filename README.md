# Quick Start

Note that the CLI arguments defaults are shown below.

## Pruning nested 10 cross validation

python3 main.py prune=true visualise_cnt=1

## 10 cross validation

python3 main.py visualise_cnt=1

# Command Line Interface (CLI)

## Argument Format

python3 main.py [latex=<value>] [prune=<value>] [visualise_cnt=<value>] [db=<value>] [folds=<value>] [seed=<value>]

### (1) visualise_cnt

Type: integer
Default = 0

Description: number of trees you want to visualise, starting from the 1st, capped at the maximum of trees available (no error is raised if you exceed the maximum)

### (2) db

Type: filepath

Default: wifi_db/clean_dataset.txt

Description: a relative filepath to this program's current working directory which must have the format shown below.

#### Dataset Row Format

64 -56 -61 -66 -71 -82 -81 1

Where, the last element ('1' in the above example), is the class label.

### (3) folds

Type: integer

Default: 10

Description: number of folds to be used when performing k-cross validation.

### (4) seed

Type: integer

Default: 666

Description: Random Number Generator seed value.

### (5) prune

Type: bool

Default: False

Description: Enables the prune vs. unpruned comparison if specified.

### (6) latex

Type: bool

Default: False

Description: Enables the output of nparrays as latex for copy-paste.
