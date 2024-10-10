import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


def init_logger(level: str, logdir: str, console: bool) -> None:
    """Initiate and setup the root logger

    Parameters
    ----------
    console : bool
        A flag used to print logs to the console
    logdir : str
        Path to the logs directory.
        If None passed, logging to a file is disabled
    level : str
        Level of logging to the console.
        File logging always has DEBUG level
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not (console or logdir):
        logger.setLevel(logging.CRITICAL + 1)
        return

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))

        message_format = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(message_format)

        logger.addHandler(console_handler)

    if logdir:
        time_format = r"%Y-%m-%d_%H:%M:%S"

        logdir = Path(logdir)
        logdir.mkdir(parents=True, exist_ok=True)

        top_lvl_file = Path(sys.argv[0]).stem
        timestamp = datetime.now().strftime(time_format)

        logfile = logdir / f"{top_lvl_file}@{timestamp}.log"
        file_handler = logging.FileHandler(str(logfile))
        file_handler.setLevel(logging.DEBUG)

        message_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", time_format
        )
        file_handler.setFormatter(message_format)

        logger.addHandler(file_handler)


def get_logger(
    name: str, *, level="info", logdir="data/logs", console=True
) -> logging.Logger:
    """Get initated module level logger.
       Top-level function of this module. To be imported and used in other scripts

    Parameters
    ----------
    name : str
        Name of the loggger. For default consider using: __name__
    console : bool
        A flag used to print logs to the console (default is True)
    logdir : str
        Path to the logs directory. (default is 'data/logs')
        If None passed, logging to a file is disabled
    level : str
        Level of logging to the console. (default is 'info')
        File logging always has DEBUG level


    Returns
    -------
    logging.Logger
        Module level logger with given name. Set up and ready to use
    """

    init_logger(level=level, logdir=logdir, console=console)
    logger = logging.getLogger(name)

    return logger


def clear_logdir(logdir: Path) -> None:
    """Clear logs directory, removing empty files"""

    count = 0
    for item in logdir.glob("*.log"):
        if item.stat().st_size == 0:
            item.unlink()
            count += 1

    print(f"Deleted {count} empty files")


def compress_logdir(logdir: Path) -> None:
    """Compress logs directory by merging logs coming from the same script to a single file"""

    logfiles = set()
    compressed_files = set()
    count = 0

    for file in logdir.glob("*.log"):
        prefix = file.stem.split("@")[0]  # get executed file name

        if file.stem.endswith("compressed"):
            compressed_files.add(prefix.replace("_compressed", ""))
        else:
            logfiles.add(prefix)

    for prefix in logfiles:
        output_file = logdir / f"{prefix}_compressed.log"

        if prefix in compressed_files:
            filemode = "a"  # compressed file already exits, append to it
        else:
            filemode = "w"  # create compressed file

        with output_file.open(filemode, encoding="utf-8") as outfile:
            for file in sorted(logdir.glob("*.log")):
                filename = file.stem.split("@")
                if (filename[0] == prefix) and (
                    not filename[-1].endswith("compressed")
                ):
                    outfile.write(filename[1] + "\n")  # add execution date to the top
                    outfile.writelines(file.read_text(encoding="utf-8"))
                    outfile.write("\n")

                    file.unlink()  # delete file after merging
                    count += 1

    print(f"Merged {count} log files")


def get_timestamp():
    tag = datetime.now().strftime(r"%Y-%m-%d_%H:%M:%S")

    return tag


def test():
    logger = get_logger("test_logger", level="info", console=True, logdir="data/logs")

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provides logging utilities. Execute to clear and compress logs directory"
    )
    parser.add_argument(
        "logdir", nargs="?", type=str, default="data/logs", help="Logs directory"
    )

    parser.add_argument(
        "--no-purge", action="store_true", help="Flag to prevent purging empty files"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Flag to prevent directory compression",
    )

    parser.set_defaults(no_purge=False)
    parser.set_defaults(no_compress=False)

    return parser.parse_args()


def main(args):
    logdir = Path(args.logdir)

    if not logdir.is_dir():
        raise AttributeError(f"Log directory {str(logdir)} does not exist.")

    if not args.no_purge:
        clear_logdir(logdir)

    if not args.no_compress:
        compress_logdir(logdir)


if __name__ == "__main__":
    main(parse_args())
