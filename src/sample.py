import logging

logger = logging.getLogger()  # name the logger as asl
logger.setLevel(logging.DEBUG)
f_hand = logging.FileHandler("tests.log")     # file where the custom logs needs to be handled
f_hand.setLevel(logging.DEBUG) # level to set for logging the errors
f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                             datefmt='%d-%b-%y %H:%M:%S')  # format in which the logs needs to be written
f_hand.setFormatter(f_format)  # setting the format of the logs
logger.addHandler(f_hand)  # setting the logging handler with the above formatter specification

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(f_format)
logger.addHandler(ch)

logger.info("CUda there")