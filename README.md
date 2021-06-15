# An experiment launcher for hyperparameter tuning

## Installation

### Requirements

[Gin-config](https://github.com/google/gin-config)

```
python setup.py install
```

## Usage

First of all, the script to run must have the following structure

```
import gin
import sys


@gin.configurable
def foo(name, experiment, hp1, hp2, hp3, ...):
    ...


if __name__ == '__main__':
    gin.parse_config_file(sys.argv[1])
    foo()

```
The first argument to the script must be reserved for config file, which is 
supplied by `launcher`.
`launcher` supports other optional arguments. If there exist such arguments, 
it should be after the config file argument.

Then, make a python script, for e.g., `launch.py` with the following contents.

```
from launcher import Launcher

lc = Launcher('bar', 'foo', sync='.', ignores=('trash.py', 'garbage.py'), server=1)
lc.add_hyperparameters('hp1', value=10)
lc.add_hyperparameters('hp2', value=[1e-3, 3e-3, 1e-2], tunable=True)
lc.add_hyperparameters('hp3', value=100)
lc.add_hyperparameters('hp4', value=[32, 435, 456, 567, 234], tunable=True)
lc.launch('toy.py')
```

If `server=None`, `launcher` will use `ts` instead of `ms` and the queued jobs will
run locally.

For each hyperparameter that function `foo` accepts except for `name` and `experiment`, 
create a hyperparameter in the `launcher`.
If a hyperparameter is supposed to be tuned, indicate they are tunable
by setting `tunable=True`.
In this case, the `value` argument should be a list or tuples of values.
If the main script requires additional arguments, one can pass those arguments to
`launch.py` and call like `lc.launch('toy.py', sys.argv[1:])`.

If some experiments for some hyperparameters should be skipped, 
add
```
lc.skip_for('hp4', (32, 456))
```
before launching.

Executing this will queue multiple jobs with multiple configurations that we specify.
