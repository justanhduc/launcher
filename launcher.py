import random
import string
import os
import subprocess
from typing import Union, Optional, Tuple

__VERSION__ = '0.1.0'
__all__ = ['Launcher']


class Hyperparameter:
    def __init__(self, name, value, tunable=False):
        self.name = name
        self.value = value
        self.tunable = tunable


class Launcher:
    """
    Launch a series of experiments against tunable hyperparameters.

    Parameters
    ----------
     name : str
        Name of the model
     configurable : str
        The name that is used for `gin.configurable`.
     sync : Optional[Union[List[str], str]]
        A list of folders/files to sync in order to execute the script.
        Default: `None`.
     ignores : Optional[Union[List[str], str]]
        Specify what folders/files/patterns to ignore.
        Default: `None`.
     server : int
        Choose which server to perform the experiment.
        The number depends on `messenger`.
        Default: `None`.
     num_gpus : int
        The number of GPUs required for the experiment.
        Default: `0`.
     tmp_configs_folder : str
        The temporary folder to store all the generated config files.
        This folder should be in the local machine.
        Default: `None`.
     experiment_root : str
        The temporary folder to execute the experiment from.
        This folder should be in the remote machine.
        By default, it will upload the experiment to the
        `messenger` temp dir.
        Default: `None`.
     interpreter : str
        Path to the Python interpreter used for the experiment.
        This Python interpreter should be in the remote machine.
        Default: `None`.

    """
    def __init__(self,
                 name: str,
                 configurable: str,
                 sync: Optional[Union[Tuple[str, ...], str]] = None,
                 ignores: Optional[Union[Tuple[str, ...], str]] = None,
                 server: int = None,
                 num_gpus: int = 0,
                 tmp_configs_folder: str = None,
                 experiment_root: str = None,
                 interpreter: str = None):
        if sync is None:
            sync = []
        else:
            if isinstance(sync, str):
                sync = [sync]

        if ignores is None:
            ignores = []

        self.name = name
        self.configurable = configurable
        self.sync = sync
        self.ignores = list(ignores)
        self.num_gpus = num_gpus
        self.__hyperparameters = {}
        if tmp_configs_folder is None:
            tmp_configs_folder = 'tmp'

        if not os.path.exists(tmp_configs_folder):
            os.mkdir(tmp_configs_folder)

        self.tmp_folder = tmp_configs_folder
        self.ignores.append(self.tmp_folder)
        self.server = 0 if server is None else server
        self.tmp_root = '/tmp/messenger-tmp' if experiment_root is None else experiment_root
        self.interpreter = 'python' if interpreter is None else interpreter

    def set_tunable(self, hyperparameter: str):
        """
        Sets a hyperparameter as tunable.
        One can read all hyperparameters from a config file,
        and use this method to mark some hyperparameter as tunable.

        :param hyperparameter:
            name of the hyperparameter
        """
        self.__hyperparameters[hyperparameter].tunable = True

    def add_hyperparameters(self, name: str, value, tunable=False):
        """
        Adds a hyperparameter.

        :param name:
            name of the hyperparameter
        :param value:
            value of the hyperparameter.
            If this hyperparameter is tunable,
            the value should be a list/tuple of values
        :param tunable:
            whether this hyperparameter is tunable.
            Default: `False`.
        """
        if tunable:
            assert isinstance(value, (list, tuple))
        self.__hyperparameters[name] = Hyperparameter(name, value, tunable)

    def hyperparameters_from_config(self, config: dict):
        """
        Reads hyperparameters from a config dictionary.

        :param config:
            a config dictionary
        """
        for k, v in config.items():
            self.add_hyperparameters(k, v)

    def generate_configs(self):
        """
        Generates a matrix of configurations.
        """
        hyperparameters = list(self.__hyperparameters.values())

        def _generate(i, name):
            if i == len(hyperparameters):
                return {name: {}}

            configs = {}
            hp = hyperparameters[i]
            values = hp.value if hp.tunable else [hp.value]
            for value in values:
                if hp.tunable:
                    if name:
                        tmp_name = f'{name}-{hp.name}-{value}'
                    else:
                        tmp_name = f'{hp.name}-{value}'
                else:
                    tmp_name = name

                for key, conf in _generate(i + 1, tmp_name).items():
                    conf[hp.name] = value
                    configs[key] = conf

            return configs

        return _generate(0, '')

    def launch(self, script: str, extra_args=None):
        """
        Launches the script based on the given hyperparameters.

        :param script:
            name of the script to launch
        """
        all_configs = self.generate_configs()
        for config_name, config in all_configs.items():
            if not config_name:
                config_name = 'default'

            config_filename = f'{config_name}.gin'
            config_file = os.path.join(self.tmp_folder, config_filename)
            headers = [
                f'{self.configurable}.name = "{self.name}" \n',
                f'{self.configurable}.experiment = "{config_name}" \n'
            ]
            contents = []
            for k, v in config.items():
                if isinstance(v, str):
                    contents.append(f'{self.configurable}.{k} = "{v}" \n')
                else:
                    contents.append(f'{self.configurable}.{k} = {v} \n')

            with open(config_file, 'w') as f:
                f.writelines(headers)
                f.writelines(contents)
                f.close()

            to_sync = list(self.sync)
            to_sync.append(config_file)
            to_sync = ':'.join(to_sync)
            cmd = ['ms', '-H', str(self.server), '--sync', to_sync]
            if self.ignores:
                exclude = ':'.join(self.ignores)
                exclude = ['--exclude', f'{exclude}']
                cmd.extend(exclude)

            tmpdir = os.path.join(self.tmp_root, config_name)
            tmpdir += '-'
            tmpdir += ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            cmd += ['--sync_dest', tmpdir]
            if self.num_gpus:
                cmd.append(f'-G {str(self.num_gpus)}')

            cmd += [self.interpreter, script, config_filename]
            if extra_args is not None:
                cmd += list(extra_args)
            subprocess.call(cmd)
