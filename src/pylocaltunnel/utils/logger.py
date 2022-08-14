import logging
import os
from datetime import datetime

# create a time format for the log file
time_format = "%Y-%m-%d_%H-%M-%S"

now = datetime.now()

filename = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/logs/localtunnel-{now.strftime(time_format)}.log"

logging.basicConfig(level=logging.DEBUG,
                    filename= filename,
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)
