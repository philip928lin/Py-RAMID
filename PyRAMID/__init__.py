

from PyRAMID.Setting import ConsoleLogParm
import logging

print("\n\nWelcome to Py-RAMID!\nA python package of a Riverware and Agent-based Modeling Interface for Developers.\n")

logger = logging.getLogger(__name__)        # This is the root of logging
logger.setLevel(ConsoleLogParm["Msglevel"])

# Clear all existed handlers and then add new console handler by default
logger.handlers.clear()
ch = logging.StreamHandler()
ch.setLevel(ConsoleLogParm["Msglevel"])
formatter_ch = logging.Formatter(ConsoleLogParm["MsgFormat"], datefmt=ConsoleLogParm["DateFormate"])
ch.setFormatter(formatter_ch)
logger.addHandler(ch)

