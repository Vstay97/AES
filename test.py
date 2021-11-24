import logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.info('[Dev]   loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' %
            (
1, 2, 3, 4, 5, 6))

