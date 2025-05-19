import logging
import sys

def get_logger(name, level=logging.INFO):
    """
    Crea y devuelve un objeto logger configurado.

    Args:
        name (str): El nombre del logger (generalmente __name__).
        level (int): El nivel de logging (e.g., logging.DEBUG, logging.INFO, logging.ERROR).
                     Por defecto es logging.INFO.

    Returns:
        logging.Logger: Un objeto logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Crear un handler para la salida estándar si no existe ya
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger