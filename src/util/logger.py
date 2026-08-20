import logging

logging.basicConfig(
        filename='log',
        encoding='utf8',
        filemode='a',
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
        )

log = logging.getLogger(__name__)
log.info('Logger Initiating 👾')
