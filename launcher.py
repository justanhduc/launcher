import random
import string
import os
import subprocess
from typing import Union, Optional, Tuple, Any, List
from shutil import rmtree
from collections import defaultdict

__VERSION__ = '0.1.0'
__all__ = ['Launcher']


class Hyperparameter:
    def __init__(self, name, value, tunable=False):
        self.name = name
        self.value = value
        self.tunable = tunable


def _spawn_set():
    return set()


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
        else:
            if isinstance(ignores, str):
                ignores = [ignores]

        self.name = name
        self.configurable = configurable
        self.sync = sync
        self.ignores = list(ignores)
        self.num_gpus = num_gpus
        self.__hyperparameters = {}
        if tmp_configs_folder is None:
            tmp_configs_folder = '__tmp__'

        if not os.path.exists(tmp_configs_folder):
            os.mkdir(tmp_configs_folder)

        self.tmp_folder = tmp_configs_folder
        self.ignores.append(self.tmp_folder)
        self.server = server
        if experiment_root is None:
            if self.server is None:  # work locally
                self.tmp_root = '/tmp/launcher-tmp'
                if not os.path.exists(self.tmp_root):
                    os.mkdir(self.tmp_root)
            else:
                self.tmp_root = '/tmp/messenger-tmp'
        else:
            self.tmp_root = experiment_root
            if self.server is None:
                if not os.path.exists(self.tmp_root):
                    os.mkdir(self.tmp_root)
            else:
                assert os.path.isabs(self.tmp_root), f'Relative experiment root is not allowed. Got {experiment_root}'

        self.interpreter = 'python' if interpreter is None else interpreter
        self._skips = defaultdict(_spawn_set)

    def set_tunable(self, hyperparameter: str) -> None:
        """
        Sets a hyperparameter as tunable.
        One can read all hyperparameters from a config file,
        and use this method to mark some hyperparameter as tunable.

        :param hyperparameter:
            name of the hyperparameter
        """
        self.__hyperparameters[hyperparameter].tunable = True

    def add_hyperparameters(self, name: str, value, tunable=False) -> None:
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

    def hyperparameters_from_config(self, config: dict) -> None:
        """
        Reads hyperparameters from a config dictionary.

        :param config:
            a config dictionary
        """
        for k, v in config.items():
            self.add_hyperparameters(k, v)

    def skip_for(self, name: str, value: Union[Tuple[Any, ...], Any]) -> None:
        if not isinstance(value, (list, tuple)):
            value = [value]
        for v in value:
            self._skips[name].add(v)

    def _skip_this(self, config):
        for k, v in config.items():
            if k in self._skips:
                if v in self._skips[k]:
                    return True
        return False

    def generate_configs(self) -> dict:
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

    def _sync(self, files_and_folders: List[str], target: str):
        if target is None:
            tmpdir = os.path.join(self.tmp_root, 'tmp-')
            tmpdir += ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        else:
            tmpdir = target

        exclude = [f'--exclude={exclude}' for exclude in self.ignores] if self.ignores else []
        for to_sync in files_and_folders:
            cmd = ['rsync', '-uar', to_sync]
            cmd += exclude
            cmd += [f'{tmpdir}/']
            subprocess.call(cmd)  # sync using rsync

        return tmpdir

    def launch(self, script: str, extra_args: List[str] = None) -> None:
        """
        Launches the script based on the given hyperparameters.

        :param script:
            name of the script to launch
        :param extra_args:
            extra arguments to be passed to script.
        """
        all_configs = self.generate_configs()
        for config_name, config in all_configs.items():
            if self._skip_this(config):
                continue

            if not config_name:
                config_name = 'default'

            config_filename = f'{self.name}-{config_name}.gin'
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
            tmpdir = os.path.join(self.tmp_root, config_name)
            tmpdir += '-'
            tmpdir += ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            if self.server is None:  # work locally
                working_dir = self._sync(to_sync, tmpdir)
                cmd = ['ts']
                current_dir = os.getcwd()
                os.chdir(working_dir)  # cd here to launch ts inside
            else:
                to_sync = ':'.join(to_sync)
                cmd = ['ms', '-H', str(self.server), '--sync', to_sync]
                if self.ignores:
                    exclude = ':'.join(self.ignores)
                    exclude = ['--exclude', f'{exclude}']
                    cmd.extend(exclude)
                cmd += ['--sync_dest', tmpdir]

            if self.num_gpus:
                cmd += ['-G', f'{self.num_gpus}']

            cmd += ['-L', f'{self.name}-{config_name}']
            script_cmd = [self.interpreter, script, config_filename]
            if extra_args is not None:
                script_cmd += list(extra_args)
            if cmd[0] == 'ms':
                cmd.append(' '.join(script_cmd))
            elif cmd[0] == 'ts':
                cmd += script_cmd
            else:
                raise ValueError  # should never end up here

            subprocess.call(cmd)
            if self.server is None:
                os.chdir(current_dir)  # cd back
        rmtree(self.tmp_folder)
