from loguru import logger
import os

# remove the default handler that sends all log messages to stderr
logger.remove()
logger.add(os.getenv("LOGFILE"), colorize=True, enqueue=True)