import os
import shlex
from abc import abstractmethod

import tempfile
from io import StringIO
from datetime import datetime

from pathlib import Path

import subprocess
import warnings
from contextlib import contextmanager
from pathlib import Path

import typing

from .common import command, popen_command
from .logging import get_logger
import glob as _glob
import portalocker


def file_lock(path: str):  # type: ignore
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.

    This is useful to make sure workers don't cache files to the same location.

    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`

    Examples:

        filename = "/path/to/file"
        with file_lock(filename):
            if not os.path.isfile(filename):
                do_create_file()
    """
    dirname = os.path.dirname(path)
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        # makedir is not atomic. Exceptions can happen when multiple workers try
        # to create the same dir, despite exist_ok=True.
        # When this happens, we assume the dir is created and proceed to creating
        # the lock. If failed to create the directory, the next line will raise
        # exceptions.
        pass
    return portalocker.Lock(path + ".lock", timeout=1800)  # type: ignore


logger = get_logger()


class PathManagerRegister(object):
    def __init__(self):
        self.d = {}
        self.local_path_manager = LocalPathManager

    def register(self, c):
        assert issubclass(c, PathManager)
        self.d[c.prefix] = c

    def get_path_manager(self, path: str):
        for k, c in self.d.items():
            if path.startswith(k):
                return c
        return self.local_path_manager


hdfs_prefix = "hdfs://"


def is_hdfs_path(p: str):
    if p.startswith(hdfs_prefix):
        return True
    else:
        return False


class PathManager(object):
    def __init__(self, *path: str):
        self.path = self._join_paths(*path)

    @staticmethod
    def build(*path: str):
        assert len(path) > 0, \
            f"The path input can not be empty"
        return path_manager_register.get_path_manager(path[0])(*path)

    @abstractmethod
    def _join_paths(self, *path: str):
        pass

    @abstractmethod
    def make_dir(self):
        pass

    @abstractmethod
    def dir_exists(self):
        pass

    @abstractmethod
    def file_exists(self):
        pass

    def check_file_exist(self):
        path = self.path
        if not self.file_exists():
            raise ValueError(f"The file {path} is not existed")

    def check_dir_exist(self):
        path = self.path
        if not self.dir_exists():
            raise ValueError(f"directory {path} does not exist")

    @abstractmethod
    def ls_dir(self):
        pass

    @abstractmethod
    def file_lines_number(self):
        pass

    @abstractmethod
    def remove_file(self):
        pass

    @abstractmethod
    def sync_file_to(self, target_path: str, source_no_existed_ignore=False,
                     overwrite_target=True):
        pass

    @abstractmethod
    def symlink_to(self, target, force_empty=False):
        pass

    def join_paths(self):
        return self.path

    @abstractmethod
    def split(self):
        pass

    def build_contain_dir(self):
        """
        create the directory contains the file self.path
        """
        container_dir = self.split()[0]
        if len(container_dir.strip()) == 0:
            return
        PathManager.build(container_dir).make_dir()

    @abstractmethod
    def open(self, mode):
        pass

    @abstractmethod
    def cat(self):
        """
        When use this function, please ensure read all strings
        from the stream.
        :return: A stream of string
        """
        pass

    @abstractmethod
    def glob(self):
        pass

    @abstractmethod
    def modified_time(self):
        pass

    @abstractmethod
    def append_to_file(self, s: str):
        pass


class LocalPathManager(PathManager):

    def append_to_file(self, s: str):
        s = shlex.quote(s)
        command(f"echo {s} >> {self.path}")

    def modified_time(self):
        return datetime.fromtimestamp(
            os.path.getmtime(self.path)
        )

    def glob(self):
        return _glob.glob(self.path)

    @contextmanager
    def cat(self):
        logger.info(f"To cat the path {self.path}")
        p = subprocess.Popen(
            f"cat {self.path}",
            stdout=subprocess.PIPE,
            shell=True,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        try:
            yield p.stdout
        finally:
            for _ in p.stdout:
                pass
            error_msg = p.stderr.readlines()
            p.wait()
            if p.returncode != 0:
                raise ValueError("".join(error_msg))
            else:
                p.stderr.close()

    @contextmanager
    def open(self, mode):
        if self.file_exists():
            pass
        elif self.dir_exists():
            raise ValueError(f"Can not open the directory {self.path}")
        elif "r" in mode:
            raise ValueError(f"Can not read a non-existed file! {self.path}")
        else:
            self.build_contain_dir()
        with open(self.path, mode) as f:
            yield f

    def _join_paths(self, *path: str):
        return os.path.join(*path)

    def make_dir(self):
        command(f"mkdir -p {self.path}")

    def dir_exists(self):
        path = self.path
        if os.path.exists(path):
            if os.path.isdir(path):
                return True
        return False

    def file_exists(self):
        path = self.path
        if os.path.exists(path):
            if os.path.isfile(path):
                return True
        return False

    def ls_dir(self):
        self.check_dir_exist()
        return os.listdir(self.path)

    def file_lines_number(self):
        file_path = self.path
        check_file_exist(file_path)
        p = subprocess.Popen(f"wc -l {file_path}", shell=True, stdout=subprocess.PIPE)
        file_length = p.stdout.readlines()[0]
        return int(file_length.strip().split()[0])

    def remove_file(self):
        # self.check_file_exist()
        assert self.dir_exists() or self.file_exists()
        command(f"rm -r {self.path}")

    def sync_file_to(self, target_path: str, source_no_existed_ignore=False,
                     overwrite_target=True):
        target_path = PathManager.build(target_path)
        target_path.build_contain_dir()
        t_path = target_path.path
        if isinstance(target_path, LocalPathManager):
            command(f"cp {self.path} {t_path}")
        elif isinstance(target_path, HdfsPathManager):
            overwrite = "-f"
            command(f"hadoop fs -put {overwrite} {self.path} {t_path}", use_system=False)

    def symlink_to(self, target, force_empty=False):
        target_path = self.build(target)
        assert isinstance(target_path, self.__class__)
        source = self.path
        if os.path.exists(target):
            if force_empty:
                raise ValueError(f"The target {target} existed")
            assert Path(target).resolve() == Path(source).resolve()
            return
        make_dir(os.path.split(target)[0])
        os.symlink(
            source,
            target,
        )

    def split(self):
        return os.path.split(self.path)

    def sync_to_local(self):
        return self.path


class HdfsPathManager(PathManager):
    prefix: str = hdfs_prefix

    def append_to_file(self, s: str):
        s = shlex.quote(s)
        command(f"echo {s} | hadoop fs -appendToFile - {self.path}")

    def modified_time(self):
        res = popen_command(["hadoop", "fs", "-stat", "%y", self.path])
        return datetime.strptime(res[0].strip(), "%Y-%m-%d %H:%M:%S")

    def glob(self):
        res = popen_command(["hadoop", "fs", "-ls", self.path])
        # res = res[1:]
        # print(res)
        res = map(lambda x: x.strip().split(), res)
        res = filter(lambda x: len(x) == 8, res)
        res = [t[-1] for t in res]
        # res = [t.split("/")[-1] for t in res]
        return res

    @contextmanager
    def cat(self):
        logger.info(f"To cat the path {self.path}")
        p = subprocess.Popen(
            f"hadoop fs -text {self.path}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        try:
            yield p.stdout
        finally:
            # read all lines in stdout and stderr
            # Avoid the deadlock for wait
            for l in p.stdout:
                pass
            p.stdout.close()
            error_msg = "".join(p.stderr.readlines())
            p.stderr.close()
            p.wait()
            if p.returncode == 0:
                pass
            else:
                raise ValueError(error_msg)


    @staticmethod
    def _cache_path(p):
        home = str(Path.home())
        return LocalPathManager(home, "__cached_dir", p[len(hdfs_prefix):])

    @staticmethod
    def _cache_persist_path(p):
        home = str(Path.home())
        return LocalPathManager(home, "__cached_persist_dir", p[len(hdfs_prefix):])

    @contextmanager
    def open(self, mode):
        cache_path = self._cache_path(self.path)
        if self.file_exists():
            sync_file(cache_path.path, self.path)
        elif self.dir_exists():
            raise ValueError("Can not open a directory")
        elif "r" in mode:
            raise ValueError(f"Can not open a non-existed file {self.path}")
        else:
            cache_path.build_contain_dir()
        try:
            with cache_path.open(mode) as f:
                yield f
            if "a" in mode or "w" in mode:
                sync_file(self.path, cache_path.path)
        finally:
            cache_path.remove_file()

    def _join_paths(self, *path: str):
        return "/".join(path)

    def make_dir(self):
        command(f"hadoop fs -mkdir -p {self.path}", use_system=False)

    def dir_exists(self):
        res = command(f"hadoop fs -test -d {self.path}", raise_exception=False, print_error=False, use_system=False)
        if res != 0:
            return False
        else:
            return True

    def file_exists(self):
        res = command(f"hadoop fs -test -f {self.path}", raise_exception=False, print_error=False, use_system=False)
        if res != 0:
            return False
        else:
            return True

    def ls_dir(self):
        try:
            res = popen_command(["hadoop", "fs", "-ls", self.path])
        except RuntimeError as e:
            return []
        # res = res[1:]
        # print(res)
        res = map(lambda x: x.strip().split(), res)
        res = filter(lambda x: len(x) == 8, res)
        res = [t[-1] for t in res]
        res = [t.split("/")[-1] for t in res]
        return res

    def file_lines_number(self):
        res = popen_command([f"hadoop fs -cat {self.path} | wc -l"], shell=True)
        return int(res[0])

    def remove_file(self):
        command(f"hadoop fs -rm -f {self.path}", use_system=False)

    def sync_file_to(self, target_path: str, source_no_existed_ignore=False,
                     overwrite_target=True):
        target_path = PathManager.build(target_path)
        target_path.build_contain_dir()

        if overwrite_target and target_path.file_exists():
            target_path.remove_file()

        t_path = target_path.path

        if isinstance(target_path, HdfsPathManager):
            command(f"hadoop fs -cp {self.path} {t_path}", use_system=False)
        elif isinstance(target_path, LocalPathManager):
            command(f"hadoop fs -copyToLocal {self.path} {t_path}", use_system=False)

    def symlink_to(self, target, force_empty=False):
        raise ValueError("The hdfs path can not create symbol link")

    def split(self):
        paths = self.path.split("/")
        return "/".join(paths[:-1]), paths[-1]

    def sync_to_local(self):
        local_path = self._cache_persist_path(self.path)
        with file_lock(local_path.path):
            sync_file(local_path.path, self.path)
        return local_path.path


path_manager_register = PathManagerRegister()
path_manager_register.register(HdfsPathManager)


def make_dir(*path: str):
    """
    This method will recursively create the directory
    :param path: A variable length parameter. The function will use the os.path.join to the path list
    :return:
    """
    PathManager.build(*path).make_dir()


def dir_exists(path: str):
    return PathManager.build(path).dir_exists()


def file_exists(path: str):
    return PathManager.build(path).file_exists()


def check_file_exist(path: str):
    PathManager.build(path).check_file_exist()


def check_dir_exist(path: str):
    PathManager.build(path).check_dir_exist()


def ls_dir(path: str):
    return PathManager.build(path).ls_dir()


def file_lines_number(file_path: str):
    return PathManager.build(file_path).file_lines_number()


def split_file(file_name, splited_dir, buckets):
    # with open(file_name, "r") as f:
    #     lines = f.readlines()
    #     lines = more_itertools.distribute(buckets, lines)
    # file_name = os.path.split(file_name)[1]
    # res = []
    # logger.info("TO split file:{}".format(file_name))
    # for idx, l in enumerate(lines):
    #     split_path = os.path.join(splited_dir, "{}-{}".format(file_name, idx))
    #     with open(split_path, "w") as f:
    #         l = map(lambda x: x.strip(), l)
    #         f.write("\n".join(l))
    #     logger.info("split to {}".format(split_path))
    #     res.append(split_path)
    # return res
    buckets = int(buckets)
    name = os.path.split(file_name)[1]
    target_file_pattern = os.path.join(splited_dir, f"{name}.")
    command(f"split -n l/{buckets} {file_name} {target_file_pattern}")
    return _glob.glob(target_file_pattern+"*")


def merge_files(target_file, source_files):
    logger.info("Begin merge {} to {}".format(" ".join(source_files), target_file))
    target_dir = os.path.split(target_file)[0]
    if not dir_exists(target_dir):
        warnings.warn("The target dir {} is not existed, it will be created".format(target_file))
        make_dir(target_dir)

    os.system("cat {} > {}".format(" ".join(source_files), target_file))
    logger.info("End merge {} to {}".format(" ".join(source_files), target_file))


def remove_files(*args: str):
    for path in args:
        PathManager.build(path).remove_file()


def temp_dir(d: str=None):
    if d is not None:
        assert isinstance(PathManager.build(d), LocalPathManager)
        if not dir_exists(d):
            make_dir(d)
    return tempfile.TemporaryDirectory(dir=d)


@contextmanager
def temp_file(d=None):
    with temp_dir(d) as t_d:
        yield os.path.join(t_d, "tmp")


def _sync_file_ignore_existed(target_path: str, source_path: str, source_no_existed_ignore: bool = False,
                              overwrite_old_file: bool = False, overwrite_target: bool = False):
    target_path = PathManager.build(target_path)
    source_path = PathManager.build(source_path)
    if source_path.file_exists():
        if target_path.file_exists():
            if overwrite_target:
                source_path.sync_file_to(target_path.path, )
            elif overwrite_old_file and source_path.modified_time() > target_path.modified_time():
                source_path.sync_file_to(target_path.path, )
            else:
                logger.info(f"File {target_path.path} is existed. Skip it.")
                return
        elif target_path.dir_exists():
            raise ValueError(f"The source path {source_path.path} is a file,"
                             f"but the synced target path {target_path.path} is a directory")
        else:
            source_path.sync_file_to(target_path.path,)
    elif source_path.dir_exists():
        for sub_name in source_path.ls_dir():
            _sync_file_ignore_existed(
                target_path=join_paths(target_path.path, sub_name),
                source_path=join_paths(source_path.path, sub_name),
                source_no_existed_ignore=source_no_existed_ignore,
                overwrite_old_file=overwrite_old_file,
                overwrite_target=overwrite_target,
            )
    else:
        if not source_no_existed_ignore:
            raise ValueError(f"The source path {source_path.path} does not exist!")


def sync_file(target_path: str, source_path: str, source_no_existed_ignore: bool = False,
              overwrite_target: bool = True, overwrite_old_file: bool = False):
    """
    Copy the content in the source path to the target path.
    The target content will just at the target path, even if target path is a directory.
    It is different from cp command action.
    :param target_path:
    :param source_path:
    :param source_no_existed_ignore: If set and the source path does not exit, do nothing.
    :param overwrite_target: If set, overwrite the target.
    :param overwrite_old_file: If set, only overwrite the old file by the time stamp.
        This is not always successful, if two nodes use different time zone.
    :return:
    """
    _sync_file_ignore_existed(target_path=target_path, source_path=source_path,
                              source_no_existed_ignore=source_no_existed_ignore,
                              overwrite_old_file=overwrite_old_file,
                              overwrite_target=overwrite_target)


def symlink(source, target, force_empty=False):
    """
    :param target: The symbolic path
    :param source: The real path
    :param force_empty: If true, the target source should be empty.
    If False and target exists, source should point to the same as the target
    :return: None
    """
    PathManager.build(source).symlink_to(target, force_empty=force_empty)


def join_paths(*paths):
    return PathManager.build(*paths).path


def path_split(path: str):
    return PathManager.build(path).split()


@contextmanager
def open_file(path: str, mode: str):
    with PathManager.build(path).open(mode) as f:
        yield f


@contextmanager
def cat(path):
    with PathManager.build(path).cat() as f:
        yield f


def glob(path):
    return PathManager.build(path).glob()


def build_contain_dir(path):
    PathManager.build(path).build_contain_dir()


def home():
    return str(Path.home())


def modified_time(path):
    return PathManager.build(path).modified_time()


def append_to_file(path: str, s: str):
    PathManager.build(path).append_to_file(s)


def sync_to_local(path: str) -> str:
    return PathManager.build(path).sync_to_local()
