import sys

import argparse
import os
import subprocess
import typing
import traceback


from .logging import get_logger


logger = get_logger()


def python_interpreter():
    res = os.environ.get('PYTHON_INTERPRETER', 'python3')
    logger.info("python interpreter:{}".format(res))
    return res


def command(cmd, *args, raise_exception=True, print_error=True,
            use_system=True):
    if len(args) != 0:
        cmd = cmd.format(*args)
    logger.info(cmd)
    if not use_system:
        popen = subprocess.Popen(
            cmd,
            shell=True,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        _, err = popen.communicate()
        if popen.returncode != 0:
            if print_error:
                sys.stderr.write(err)
            if raise_exception:
                raise RuntimeError(f"The cmd '{cmd}' run failed")
        else:
            popen.stderr.close()
        return_code = popen.returncode
    else:
        return_code = os.system(cmd)
        if return_code != 0 and raise_exception:
            raise RuntimeError(f"The cmd '{cmd}' run failed")

    return return_code


def popen_command(cmd: typing.Union[typing.List[str], str], shell: bool=False,
                  stdin=None):
    logger.info(cmd)
    pipe = subprocess.Popen(cmd,
                            stdin=stdin,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            encoding="utf-8",
                            shell=shell)
    std, err = pipe.communicate(stdin)
    if pipe.returncode != 0:
        sys.stderr.write(err)
        raise RuntimeError(f"The cmd {cmd} run failed")
    else:
        res = std.strip().split("\n")
        return res


_cpu_number = None


def get_cpu_num():
    return _cpu_number


def pipe(*cmd, need_output=False):
    p = None
    for i, c in enumerate(cmd):
        in_pipe = None
        if p is not None:
            in_pipe = p.stdout

        out_pipe = subprocess.PIPE
        if i == len(cmd) - 1 and not need_output:
            out_pipe = subprocess.DEVNULL
        _p = subprocess.Popen(c, shell=True,
                              stdin=in_pipe,
                              stdout=out_pipe)
        if in_pipe is not None:
            p.stdout.close()
        p = _p
    if need_output:
        return p.communicate()[0]
    else:
        p.wait()


def print_exception(e: Exception):
    traceback.print_tb(e.__traceback__)

