# Logging setup
"""
Configuration du système de journalisation pour le système de gestion d'inventaire pharmaceutique.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

from src.config import LOG_LEVEL, DEBUG

# Création du répertoire de logs s'il n'existe pas
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configuration du format de journalisation
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# Nom du fichier de log basé sur la date
log_filename = os.path.join(logs_dir, f"smart_pharma_{datetime.now().strftime('%Y%m%d')}.log")

# Configuration du niveau de journalisation
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

# Configuration du gestionnaire de fichiers avec rotation
file_handler = RotatingFileHandler(
    log_filename,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(log_format, date_format))
file_handler.setLevel(log_level)

# Configuration du gestionnaire de console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(log_format, date_format))
console_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Configuration du logger racine
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Loggers spécifiques
db_logger = logging.getLogger("database")
agent_logger = logging.getLogger("agents")
ui_logger = logging.getLogger("ui")

# Fonction pour obtenir un logger
def get_logger(name):
    """
    Obtient un logger configuré pour le module spécifié.
    
    Args:
        name: Nom du module
        
    Returns:
        Logger configuré
    """
    return logging.getLogger(name)
