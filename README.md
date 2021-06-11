# An experiment launcher for hyperparameter tuning

## Installation

### Requirements

[Gin-config]()

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

Then, make a python script, for e.g., `launch.py` with the following contents.

```
from launcher import Launcher

launcher = Launcher('bar', 'foo', sync='.', ignores=('tmp', 'launcher.py', '.idea', '.git'), server=1)
launcher.add_hyperparameters('hp1', value=10)
launcher.add_hyperparameters('hp2', value=[1e-3, 3e-3, 1e-2], tunable=True)
launcher.add_hyperparameters('hp3', value=100)
launcher.add_hyperparameters('hp4', value=[32, 435, 456, 567, 234], tunable=True)
launcher.launch('toy.py')
```
For each hyperparameter that function `foo` accepts except for `name` and `experiment`, 
create a hyperparameter in the `launcher`.
If a hyperparameter is supposed to be tuned, indicate they are tunable
by setting `tunable=True`.
In this case, the `value` argument should be a list or tuples of values.

Executing this will queue multiple jobs with multiple configurations that we specify.
