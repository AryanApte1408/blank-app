# # # # # """
# # # # # database_manager.py - Hotswappable database configuration manager
# # # # # Makes RAG independent of specific database instances
# # # # # """
# # # # # import os
# # # # # from typing import Dict, Optional, List
# # # # # from dataclasses import dataclass
# # # # # from enum import Enum


# # # # # class DatabaseMode(Enum):
# # # # #     """Available database modes."""
# # # # #     FULL = "full"
# # # # #     ABSTRACTS = "abstracts"
# # # # #     CUSTOM = "custom"


# # # # # @dataclass
# # # # # class DatabaseConfig:
# # # # #     """Configuration for a database setup."""
# # # # #     mode: str
# # # # #     sqlite_path: str
# # # # #     chroma_dir: str
# # # # #     chroma_collection: str
# # # # #     neo4j_uri: str
# # # # #     neo4j_user: str
# # # # #     neo4j_password: str
# # # # #     neo4j_database: str
# # # # #     description: str
    
# # # # #     def to_dict(self) -> Dict:
# # # # #         return {
# # # # #             "mode": self.mode,
# # # # #             "sqlite_path": self.sqlite_path,
# # # # #             "chroma_dir": self.chroma_dir,
# # # # #             "chroma_collection": self.chroma_collection,
# # # # #             "neo4j_uri": self.neo4j_uri,
# # # # #             "neo4j_user": self.neo4j_user,
# # # # #             "neo4j_password": self.neo4j_password,
# # # # #             "neo4j_database": self.neo4j_database,
# # # # #             "description": self.description
# # # # #         }


# # # # # class DatabaseManager:
# # # # #     """
# # # # #     Manages multiple database configurations and allows hotswapping.
# # # # #     RAG components query this manager instead of hardcoded config.
# # # # #     """
    
# # # # #     def __init__(self):
# # # # #         self.configs: Dict[str, DatabaseConfig] = {}
# # # # #         self.active_config_name: Optional[str] = None
# # # # #         self._load_default_configs()
    
# # # # #     def _load_default_configs(self):
# # # # #         """Load default database configurations."""
        
# # # # #         # Full papers database
# # # # #         self.register_config(
# # # # #             name="full",
# # # # #             config=DatabaseConfig(
# # # # #                 mode="full",
# # # # #                 sqlite_path=os.getenv("SQLITE_DB_FULL", r"D:\OSPO\KG-RAG1\researchers_fixed.db"),
# # # # #                 chroma_dir=os.getenv("CHROMA_DIR_FULL", r"D:\OSPO\KG-RAG1\chroma_store_full"),
# # # # #                 chroma_collection="papers_all",
# # # # #                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
# # # # #                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
# # # # #                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
# # # # #                 neo4j_database=os.getenv("NEO4J_DB", "syr-rag"),
# # # # #                 description="Full papers with complete text and metadata"
# # # # #             )
# # # # #         )
        
# # # # #         # Abstracts database
# # # # #         self.register_config(
# # # # #             name="abstracts",
# # # # #             config=DatabaseConfig(
# # # # #                 mode="abstracts",
# # # # #                 sqlite_path=os.getenv("SQLITE_DB_ABSTRACTS", r"D:\OSPO\KG-RAG1\abstracts_only.db"),
# # # # #                 chroma_dir=os.getenv("CHROMA_DIR_ABSTRACTS", r"D:\OSPO\KG-RAG1\chroma_store_abstracts"),
# # # # #                 chroma_collection="abstracts_all",
# # # # #                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
# # # # #                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
# # # # #                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
# # # # #                 neo4j_database=os.getenv("NEO4J_DB", "syr-rag"),
# # # # #                 description="Abstracts only from academic APIs"
# # # # #             )
# # # # #         )
        
# # # # #         # Set default active config
# # # # #         self.active_config_name = "full"
    
# # # # #     def register_config(self, name: str, config: DatabaseConfig):
# # # # #         """Register a new database configuration."""
# # # # #         self.configs[name] = config
# # # # #         print(f"✅ Registered database config: {name}")
    
# # # # #     def switch_config(self, name: str) -> bool:
# # # # #         """Switch to a different database configuration."""
# # # # #         if name not in self.configs:
# # # # #             print(f"❌ Config '{name}' not found")
# # # # #             return False
        
# # # # #         self.active_config_name = name
# # # # #         print(f"🔄 Switched to database config: {name}")
# # # # #         return True
    
# # # # #     def get_active_config(self) -> Optional[DatabaseConfig]:
# # # # #         """Get the currently active database configuration."""
# # # # #         if self.active_config_name:
# # # # #             return self.configs.get(self.active_config_name)
# # # # #         return None
    
# # # # #     def list_configs(self) -> List[str]:
# # # # #         """List all available database configurations."""
# # # # #         return list(self.configs.keys())
    
# # # # #     def get_config(self, name: str) -> Optional[DatabaseConfig]:
# # # # #         """Get a specific database configuration by name."""
# # # # #         return self.configs.get(name)
    
# # # # #     def add_custom_config(
# # # # #         self,
# # # # #         name: str,
# # # # #         sqlite_path: str,
# # # # #         chroma_dir: str,
# # # # #         chroma_collection: str,
# # # # #         neo4j_uri: str = "bolt://localhost:7687",
# # # # #         neo4j_user: str = "neo4j",
# # # # #         neo4j_password: str = "password",
# # # # #         neo4j_database: str = "neo4j",
# # # # #         description: str = "Custom database"
# # # # #     ):
# # # # #         """Add a custom database configuration at runtime."""
# # # # #         config = DatabaseConfig(
# # # # #             mode="custom",
# # # # #             sqlite_path=sqlite_path,
# # # # #             chroma_dir=chroma_dir,
# # # # #             chroma_collection=chroma_collection,
# # # # #             neo4j_uri=neo4j_uri,
# # # # #             neo4j_user=neo4j_user,
# # # # #             neo4j_password=neo4j_password,
# # # # #             neo4j_database=neo4j_database,
# # # # #             description=description
# # # # #         )
# # # # #         self.register_config(name, config)
    
# # # # #     def validate_config(self, name: str) -> Dict[str, bool]:
# # # # #         """Validate that all paths/connections exist for a config."""
# # # # #         config = self.get_config(name)
# # # # #         if not config:
# # # # #             return {"valid": False, "error": "Config not found"}
        
# # # # #         validation = {
# # # # #             "sqlite_exists": os.path.exists(config.sqlite_path),
# # # # #             "chroma_dir_exists": os.path.exists(config.chroma_dir),
# # # # #             "neo4j_connectable": self._test_neo4j_connection(config)
# # # # #         }
        
# # # # #         validation["valid"] = all(validation.values())
# # # # #         return validation
    
# # # # #     def _test_neo4j_connection(self, config: DatabaseConfig) -> bool:
# # # # #         """Test Neo4j connection."""
# # # # #         try:
# # # # #             from neo4j import GraphDatabase
# # # # #             driver = GraphDatabase.driver(
# # # # #                 config.neo4j_uri,
# # # # #                 auth=(config.neo4j_user, config.neo4j_password)
# # # # #             )
# # # # #             with driver.session(database=config.neo4j_database) as session:
# # # # #                 session.run("RETURN 1")
# # # # #             driver.close()
# # # # #             return True
# # # # #         except Exception as e:
# # # # #             print(f"❌ Neo4j connection failed: {e}")
# # # # #             return False


# # # # # # Global singleton instance
# # # # # _db_manager = None


# # # # # def get_db_manager() -> DatabaseManager:
# # # # #     """Get the global database manager instance."""
# # # # #     global _db_manager
# # # # #     if _db_manager is None:
# # # # #         _db_manager = DatabaseManager()
# # # # #     return _db_manager


# # # # # def get_active_db_config() -> DatabaseConfig:
# # # # #     """Quick access to active database configuration."""
# # # # #     manager = get_db_manager()
# # # # #     config = manager.get_active_config()
# # # # #     if config is None:
# # # # #         raise RuntimeError("No active database configuration set")
# # # # #     return config

# # # # """
# # # # database_manager.py - Hotswappable database configuration manager
# # # # """
# # # # import os
# # # # from typing import Dict, Optional, List
# # # # from dataclasses import dataclass
# # # # from enum import Enum


# # # # class DatabaseMode(Enum):
# # # #     """Available database modes."""
# # # #     FULL = "full"
# # # #     ABSTRACTS = "abstracts"
# # # #     CUSTOM = "custom"


# # # # @dataclass
# # # # class DatabaseConfig:
# # # #     """Configuration for a database setup."""
# # # #     mode: str
# # # #     chroma_dir: str
# # # #     chroma_collection: str
# # # #     neo4j_uri: str
# # # #     neo4j_user: str
# # # #     neo4j_password: str
# # # #     neo4j_database: str
# # # #     description: str
    
# # # #     def to_dict(self) -> Dict:
# # # #         return {
# # # #             "mode": self.mode,
# # # #             "chroma_dir": self.chroma_dir,
# # # #             "chroma_collection": self.chroma_collection,
# # # #             "neo4j_uri": self.neo4j_uri,
# # # #             "neo4j_user": self.neo4j_user,
# # # #             "neo4j_password": self.neo4j_password,
# # # #             "neo4j_database": self.neo4j_database,
# # # #             "description": self.description
# # # #         }


# # # # class DatabaseManager:
# # # #     """Manages multiple database configurations and allows hotswapping."""
    
# # # #     def __init__(self):
# # # #         self.configs: Dict[str, DatabaseConfig] = {}
# # # #         self.active_config_name: Optional[str] = None
# # # #         self._load_default_configs()
    
# # # #     def _load_default_configs(self):
# # # #         """Load default database configurations."""
        
# # # #         # Full papers database
# # # #         self.register_config(
# # # #             name="full",
# # # #             config=DatabaseConfig(
# # # #                 mode="full",
# # # #                 chroma_dir=os.getenv("CHROMA_DIR_FULL", r"D:\OSPO\KG-RAG1\chroma_store_full"),
# # # #                 chroma_collection="papers_all",
# # # #                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
# # # #                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
# # # #                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
# # # #                 neo4j_database="syr-rag",  # Different database for full mode
# # # #                 description="Full papers with complete text and metadata"
# # # #             )
# # # #         )
        
# # # #         # Abstracts database
# # # #         self.register_config(
# # # #             name="abstracts",
# # # #             config=DatabaseConfig(
# # # #                 mode="abstracts",
# # # #                 chroma_dir=os.getenv("CHROMA_DIR_ABSTRACTS", r"D:\OSPO\KG-RAG1\chroma_store_abstracts"),
# # # #                 chroma_collection="abstracts_all",
# # # #                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
# # # #                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
# # # #                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
# # # #                 neo4j_database="syr-rag-abstracts",  # Separate database for abstracts
# # # #                 description="Abstracts only from academic APIs"
# # # #             )
# # # #         )
        
# # # #         # Set default active config
# # # #         self.active_config_name = "abstracts"
    
# # # #     def register_config(self, name: str, config: DatabaseConfig):
# # # #         """Register a new database configuration."""
# # # #         self.configs[name] = config
# # # #         print(f"✅ Registered database config: {name} (Neo4j DB: {config.neo4j_database})")
    
# # # #     def switch_config(self, name: str) -> bool:
# # # #         """Switch to a different database configuration."""
# # # #         if name not in self.configs:
# # # #             print(f"❌ Config '{name}' not found")
# # # #             return False
        
# # # #         self.active_config_name = name
# # # #         config = self.configs[name]
# # # #         print(f"🔄 Switched to '{name}' (Neo4j: {config.neo4j_database})")
# # # #         return True
    
# # # #     def get_active_config(self) -> Optional[DatabaseConfig]:
# # # #         """Get the currently active database configuration."""
# # # #         if self.active_config_name:
# # # #             return self.configs.get(self.active_config_name)
# # # #         return None
    
# # # #     def list_configs(self) -> List[str]:
# # # #         """List all available database configurations."""
# # # #         return list(self.configs.keys())
    
# # # #     def get_config(self, name: str) -> Optional[DatabaseConfig]:
# # # #         """Get a specific database configuration by name."""
# # # #         return self.configs.get(name)
    
# # # #     def add_custom_config(
# # # #         self,
# # # #         name: str,
# # # #         chroma_dir: str,
# # # #         chroma_collection: str,
# # # #         neo4j_uri: str = "bolt://localhost:7687",
# # # #         neo4j_user: str = "neo4j",
# # # #         neo4j_password: str = "password",
# # # #         neo4j_database: str = "neo4j",
# # # #         description: str = "Custom database"
# # # #     ):
# # # #         """Add a custom database configuration at runtime."""
# # # #         config = DatabaseConfig(
# # # #             mode="custom",
# # # #             chroma_dir=chroma_dir,
# # # #             chroma_collection=chroma_collection,
# # # #             neo4j_uri=neo4j_uri,
# # # #             neo4j_user=neo4j_user,
# # # #             neo4j_password=neo4j_password,
# # # #             neo4j_database=neo4j_database,
# # # #             description=description
# # # #         )
# # # #         self.register_config(name, config)
    
# # # #     def validate_config(self, name: str) -> Dict[str, bool]:
# # # #         """Validate that all paths/connections exist for a config."""
# # # #         config = self.get_config(name)
# # # #         if not config:
# # # #             return {"valid": False, "error": "Config not found"}
        
# # # #         validation = {
# # # #             "chroma_dir_exists": os.path.exists(config.chroma_dir),
# # # #             "neo4j_connectable": self._test_neo4j_connection(config)
# # # #         }
        
# # # #         validation["valid"] = all(validation.values())
# # # #         return validation
    
# # # #     def _test_neo4j_connection(self, config: DatabaseConfig) -> bool:
# # # #         """Test Neo4j connection."""
# # # #         try:
# # # #             from neo4j import GraphDatabase
# # # #             driver = GraphDatabase.driver(
# # # #                 config.neo4j_uri,
# # # #                 auth=(config.neo4j_user, config.neo4j_password)
# # # #             )
# # # #             with driver.session(database=config.neo4j_database) as session:
# # # #                 session.run("RETURN 1")
# # # #             driver.close()
# # # #             return True
# # # #         except Exception as e:
# # # #             print(f"❌ Neo4j connection failed for {config.neo4j_database}: {e}")
# # # #             return False


# # # # # Global singleton instance
# # # # _db_manager = None


# # # # def get_db_manager() -> DatabaseManager:
# # # #     """Get the global database manager instance."""
# # # #     global _db_manager
# # # #     if _db_manager is None:
# # # #         _db_manager = DatabaseManager()
# # # #     return _db_manager


# # # # def get_active_db_config() -> DatabaseConfig:
# # # #     """Quick access to active database configuration."""
# # # #     manager = get_db_manager()
# # # #     config = manager.get_active_config()
# # # #     if config is None:
# # # #         raise RuntimeError("No active database configuration set")
# # # #     return config

# # # """
# # # database_manager.py - Hotswappable database configuration manager
# # # """
# # # import os
# # # from typing import Dict, Optional, List
# # # from dataclasses import dataclass
# # # from enum import Enum


# # # class DatabaseMode(Enum):
# # #     """Available database modes."""
# # #     FULL = "full"
# # #     ABSTRACTS = "abstracts"
# # #     CUSTOM = "custom"


# # # @dataclass
# # # class DatabaseConfig:
# # #     """Configuration for a database setup."""
# # #     mode: str
# # #     chroma_dir: str
# # #     chroma_collection: str
# # #     neo4j_uri: str
# # #     neo4j_user: str
# # #     neo4j_password: str
# # #     neo4j_database: str
# # #     description: str
    
# # #     def to_dict(self) -> Dict:
# # #         return {
# # #             "mode": self.mode,
# # #             "chroma_dir": self.chroma_dir,
# # #             "chroma_collection": self.chroma_collection,
# # #             "neo4j_uri": self.neo4j_uri,
# # #             "neo4j_user": self.neo4j_user,
# # #             "neo4j_password": self.neo4j_password,
# # #             "neo4j_database": self.neo4j_database,
# # #             "description": self.description
# # #         }


# # # class DatabaseManager:
# # #     """Manages multiple database configurations and allows hotswapping."""
    
# # #     def __init__(self):
# # #         self.configs: Dict[str, DatabaseConfig] = {}
# # #         self.active_config_name: Optional[str] = None
# # #         self._load_default_configs()
    
# # #     def _load_default_configs(self):
# # #         """Load default database configurations."""
        
# # #         # Full papers database
# # #         self.register_config(
# # #             name="full",
# # #             config=DatabaseConfig(
# # #                 mode="full",
# # #                 chroma_dir=os.getenv("CHROMA_DIR_FULL", r"D:\OSPO\KG-RAG1\chroma_store_full"),
# # #                 chroma_collection="papers_all",
# # #                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
# # #                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
# # #                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
# # #                 neo4j_database="syr-rag",
# # #                 description="Full papers with complete text and metadata"
# # #             )
# # #         )
        
# # #         # Abstracts database
# # #         self.register_config(
# # #             name="abstracts",
# # #             config=DatabaseConfig(
# # #                 mode="abstracts",
# # #                 chroma_dir=os.getenv("CHROMA_DIR_ABSTRACTS", r"D:\OSPO\KG-RAG1\chroma_store_abstracts"),
# # #                 chroma_collection="abstracts_all",
# # #                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
# # #                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
# # #                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
# # #                 neo4j_database="syr-rag-abstracts",
# # #                 description="Abstracts only from academic APIs"
# # #             )
# # #         )
        
# # #         self.active_config_name = "abstracts"
    
# # #     def register_config(self, name: str, config: DatabaseConfig):
# # #         """Register a new database configuration."""
# # #         self.configs[name] = config
# # #         print(f"✅ Registered database config: {name} (Neo4j DB: {config.neo4j_database})")
    
# # #     def switch_config(self, name: str) -> bool:
# # #         """Switch to a different database configuration."""
# # #         if name not in self.configs:
# # #             print(f"❌ Config '{name}' not found")
# # #             return False
        
# # #         self.active_config_name = name
# # #         config = self.configs[name]
# # #         print(f"🔄 Switched to '{name}' (Neo4j: {config.neo4j_database})")
# # #         return True
    
# # #     def get_active_config(self) -> Optional[DatabaseConfig]:
# # #         """Get the currently active database configuration."""
# # #         if self.active_config_name:
# # #             return self.configs.get(self.active_config_name)
# # #         return None
    
# # #     def list_configs(self) -> List[str]:
# # #         """List all available database configurations."""
# # #         return list(self.configs.keys())
    
# # #     def get_config(self, name: str) -> Optional[DatabaseConfig]:
# # #         """Get a specific database configuration by name."""
# # #         return self.configs.get(name)
    
# # #     def add_custom_config(
# # #         self,
# # #         name: str,
# # #         chroma_dir: str,
# # #         chroma_collection: str,
# # #         neo4j_uri: str = "bolt://localhost:7687",
# # #         neo4j_user: str = "neo4j",
# # #         neo4j_password: str = "password",
# # #         neo4j_database: str = "neo4j",
# # #         description: str = "Custom database"
# # #     ):
# # #         """Add a custom database configuration at runtime."""
# # #         config = DatabaseConfig(
# # #             mode="custom",
# # #             chroma_dir=chroma_dir,
# # #             chroma_collection=chroma_collection,
# # #             neo4j_uri=neo4j_uri,
# # #             neo4j_user=neo4j_user,
# # #             neo4j_password=neo4j_password,
# # #             neo4j_database=neo4j_database,
# # #             description=description
# # #         )
# # #         self.register_config(name, config)
    
# # #     def validate_config(self, name: str) -> Dict[str, bool]:
# # #         """Validate that all paths/connections exist for a config."""
# # #         config = self.get_config(name)
# # #         if not config:
# # #             return {"valid": False, "error": "Config not found"}
        
# # #         validation = {
# # #             "chroma_dir_exists": os.path.exists(config.chroma_dir),
# # #             "neo4j_connectable": self._test_neo4j_connection(config)
# # #         }
        
# # #         validation["valid"] = all(validation.values())
# # #         return validation
    
# # #     def _test_neo4j_connection(self, config: DatabaseConfig) -> bool:
# # #         """Test Neo4j connection."""
# # #         try:
# # #             from neo4j import GraphDatabase
# # #             driver = GraphDatabase.driver(
# # #                 config.neo4j_uri,
# # #                 auth=(config.neo4j_user, config.neo4j_password)
# # #             )
# # #             with driver.session(database=config.neo4j_database) as session:
# # #                 session.run("RETURN 1")
# # #             driver.close()
# # #             return True
# # #         except Exception as e:
# # #             print(f"❌ Neo4j connection failed for {config.neo4j_database}: {e}")
# # #             return False


# # # # Global singleton instance
# # # _db_manager = None


# # # def get_db_manager() -> DatabaseManager:
# # #     """Get the global database manager instance."""
# # #     global _db_manager
# # #     if _db_manager is None:
# # #         _db_manager = DatabaseManager()
# # #     return _db_manager


# # # def get_active_db_config() -> DatabaseConfig:
# # #     """Quick access to active database configuration."""
# # #     manager = get_db_manager()
# # #     config = manager.get_active_config()
# # #     if config is None:
# # #         raise RuntimeError("No active database configuration set")
# # #     return config

# # """
# # database_manager.py - Hotswappable database configuration manager
# # (Updated Nov 2025)
# # - Adds DB-level chronological utilities so retrieval is newest → oldest
# # - Keeps original config manager API intact
# # """
# # import os
# # import re  # NEW: used by sorting helpers
# # from typing import Dict, Optional, List
# # from dataclasses import dataclass
# # from enum import Enum


# # class DatabaseMode(Enum):
# #     """Available database modes."""
# #     FULL = "full"
# #     ABSTRACTS = "abstracts"
# #     CUSTOM = "custom"


# # @dataclass
# # class DatabaseConfig:
# #     """Configuration for a database setup."""
# #     mode: str
# #     chroma_dir: str
# #     chroma_collection: str
# #     neo4j_uri: str
# #     neo4j_user: str
# #     neo4j_password: str
# #     neo4j_database: str
# #     description: str
    
# #     def to_dict(self) -> Dict:
# #         return {
# #             "mode": self.mode,
# #             "chroma_dir": self.chroma_dir,
# #             "chroma_collection": self.chroma_collection,
# #             "neo4j_uri": self.neo4j_uri,
# #             "neo4j_user": self.neo4j_user,
# #             "neo4j_password": self.neo4j_password,
# #             "neo4j_database": self.neo4j_database,
# #             "description": self.description
# #         }


# # # ================================
# # # Chronological sorting utilities
# # # ================================
# # def _extract_year_from_meta(meta: Dict) -> int:
# #     if not meta:
# #         return 0
# #     for k in ["year", "pub_year", "publication_year", "published"]:
# #         if k in meta and meta[k] not in (None, "", "N/A", "Unknown"):
# #             try:
# #                 return int(str(meta[k])[:4])
# #             except Exception:
# #                 pass
# #     # fallback: scan string fields for YYYY
# #     text = " ".join(str(v) for v in meta.values() if isinstance(v, str))
# #     m = re.findall(r"(19|20)\d{2}", text)
# #     return int(m[0]) if m else 0


# # def sort_fused_result_newest_first(result: Dict) -> Dict:
# #     """
# #     Sorts fused_text_blocks (+ fused_metadata if present) by descending publication year.
# #     This runs in the DB layer so upstream callers (RAG pipeline) get pre-sorted context.
# #     """
# #     if not result or "fused_text_blocks" not in result:
# #         return result

# #     texts = result.get("fused_text_blocks", [])
# #     metas = result.get("fused_metadata", [])

# #     # If we have metadata, sort by it; otherwise infer from text
# #     if metas and len(metas) == len(texts):
# #         pairs = list(zip(metas, texts))
# #         pairs.sort(key=lambda p: _extract_year_from_meta(p[0]), reverse=True)
# #         metas_sorted, texts_sorted = zip(*pairs) if pairs else ([], [])
# #         result["fused_metadata"] = list(metas_sorted)
# #         result["fused_text_blocks"] = list(texts_sorted)
# #     else:
# #         # No metadata: try to infer year from text itself
# #         def year_from_text(t: str) -> int:
# #             m = re.findall(r"(19|20)\d{2}", str(t))
# #             return int(m[0]) if m else 0
# #         result["fused_text_blocks"] = sorted(texts, key=year_from_text, reverse=True)

# #     print("🕒 Database-level sorting applied: newest → oldest")
# #     return result


# # class DatabaseManager:
# #     """Manages multiple database configurations and allows hotswapping."""
    
# #     def __init__(self):
# #         self.configs: Dict[str, DatabaseConfig] = {}
# #         self.active_config_name: Optional[str] = None
# #         self._load_default_configs()
    
# #     def _load_default_configs(self):
# #         """Load default database configurations."""
        
# #         # Full papers database
# #         self.register_config(
# #             name="full",
# #             config=DatabaseConfig(
# #                 mode="full",
# #                 chroma_dir=os.getenv("CHROMA_DIR_FULL", r"D:\OSPO\KG-RAG1\chroma_store_full"),
# #                 chroma_collection="papers_all",
# #                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
# #                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
# #                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
# #                 neo4j_database="syr-rag",
# #                 description="Full papers with complete text and metadata"
# #             )
# #         )
        
# #         # Abstracts database
# #         self.register_config(
# #             name="abstracts",
# #             config=DatabaseConfig(
# #                 mode="abstracts",
# #                 chroma_dir=os.getenv("CHROMA_DIR_ABSTRACTS", r"D:\OSPO\KG-RAG1\chroma_store_abstracts"),
# #                 chroma_collection="abstracts_all",
# #                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
# #                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
# #                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
# #                 neo4j_database="syr-rag-abstracts",
# #                 description="Abstracts only from academic APIs"
# #             )
# #         )
        
# #         self.active_config_name = "abstracts"
    
# #     def register_config(self, name: str, config: DatabaseConfig):
# #         """Register a new database configuration."""
# #         self.configs[name] = config
# #         print(f"✅ Registered database config: {name} (Neo4j DB: {config.neo4j_database})")
    
# #     def switch_config(self, name: str) -> bool:
# #         """Switch to a different database configuration."""
# #         if name not in self.configs:
# #             print(f"❌ Config '{name}' not found")
# #             return False
        
# #         self.active_config_name = name
# #         config = self.configs[name]
# #         print(f"🔄 Switched to '{name}' (Neo4j: {config.neo4j_database})")
# #         return True
    
# #     def get_active_config(self) -> Optional[DatabaseConfig]:
# #         """Get the currently active database configuration."""
# #         if self.active_config_name:
# #             return self.configs.get(self.active_config_name)
# #         return None
    
# #     def list_configs(self) -> List[str]:
# #         """List all available database configurations."""
# #         return list(self.configs.keys())
    
# #     def get_config(self, name: str) -> Optional[DatabaseConfig]:
# #         """Get a specific database configuration by name."""
# #         return self.configs.get(name)
    
# #     def add_custom_config(
# #         self,
# #         name: str,
# #         chroma_dir: str,
# #         chroma_collection: str,
# #         neo4j_uri: str = "bolt://localhost:7687",
# #         neo4j_user: str = "neo4j",
# #         neo4j_password: str = "password",
# #         neo4j_database: str = "neo4j",
# #         description: str = "Custom database"
# #     ):
# #         """Add a custom database configuration at runtime."""
# #         config = DatabaseConfig(
# #             mode="custom",
# #             chroma_dir=chroma_dir,
# #             chroma_collection=chroma_collection,
# #             neo4j_uri=neo4j_uri,
# #             neo4j_user=neo4j_user,
# #             neo4j_password=neo4j_password,
# #             neo4j_database=neo4j_database,
# #             description=description
# #         )
# #         self.register_config(name, config)
    
# #     def validate_config(self, name: str) -> Dict[str, bool]:
# #         """Validate that all paths/connections exist for a config."""
# #         config = self.get_config(name)
# #         if not config:
# #             return {"valid": False, "error": "Config not found"}
        
# #         validation = {
# #             "chroma_dir_exists": os.path.exists(config.chroma_dir),
# #             "neo4j_connectable": self._test_neo4j_connection(config)
# #         }
        
# #         validation["valid"] = all(validation.values())
# #         return validation
    
# #     def _test_neo4j_connection(self, config: DatabaseConfig) -> bool:
# #         """Test Neo4j connection."""
# #         try:
# #             from neo4j import GraphDatabase
# #             driver = GraphDatabase.driver(
# #                 config.neo4j_uri,
# #                 auth=(config.neo4j_user, config.neo4j_password)
# #             )
# #             with driver.session(database=config.neo4j_database) as session:
# #                 session.run("RETURN 1")
# #             driver.close()
# #             return True
# #         except Exception as e:
# #             print(f"❌ Neo4j connection failed for {config.neo4j_database}: {e}")
# #             return False


# # # Global singleton instance
# # _db_manager = None


# # def get_db_manager() -> DatabaseManager:
# #     """Get the global database manager instance."""
# #     global _db_manager
# #     if _db_manager is None:
# #         _db_manager = DatabaseManager()
# #     return _db_manager


# # def get_active_db_config() -> DatabaseConfig:
# #     """Quick access to active database configuration."""
# #     manager = get_db_manager()
# #     config = manager.get_active_config()
# #     if config is None:
# #         raise RuntimeError("No active database configuration set")
# #     return config


# """
# database_manager.py - Hotswappable database configuration manager
# (Updated Nov 2025)
# - Adds DB-level chronological utilities so retrieval is newest → oldest
# - Keeps original config manager API intact
# """
# import os
# import re  # NEW: used by sorting helpers
# from typing import Dict, Optional, List
# from dataclasses import dataclass
# from enum import Enum


# class DatabaseMode(Enum):
#     """Available database modes."""
#     FULL = "full"
#     ABSTRACTS = "abstracts"
#     CUSTOM = "custom"


# @dataclass
# class DatabaseConfig:
#     """Configuration for a database setup."""
#     mode: str
#     chroma_dir: str
#     chroma_collection: str
#     neo4j_uri: str
#     neo4j_user: str
#     neo4j_password: str
#     neo4j_database: str
#     description: str
    
#     def to_dict(self) -> Dict:
#         return {
#             "mode": self.mode,
#             "chroma_dir": self.chroma_dir,
#             "chroma_collection": self.chroma_collection,
#             "neo4j_uri": self.neo4j_uri,
#             "neo4j_user": self.neo4j_user,
#             "neo4j_password": self.neo4j_password,
#             "neo4j_database": self.neo4j_database,
#             "description": self.description
#         }


# # ================================
# # Chronological sorting utilities
# # ================================
# def _extract_year_from_meta(meta: Dict) -> int:
#     if not meta:
#         return 0
#     for k in ["year", "pub_year", "publication_year", "published"]:
#         if k in meta and meta[k] not in (None, "", "N/A", "Unknown"):
#             try:
#                 return int(str(meta[k])[:4])
#             except Exception:
#                 pass
#     # fallback: scan string fields for YYYY
#     text = " ".join(str(v) for v in meta.values() if isinstance(v, str))
#     m = re.findall(r"(19|20)\d{2}", text)
#     return int(m[0]) if m else 0


# def sort_fused_result_newest_first(result: Dict) -> Dict:
#     """
#     Sorts fused_text_blocks (+ fused_metadata if present) by descending publication year.
#     This runs in the DB layer so upstream callers (RAG pipeline) get pre-sorted context.
#     """
#     if not result or "fused_text_blocks" not in result:
#         return result

#     texts = result.get("fused_text_blocks", [])
#     metas = result.get("fused_metadata", [])

#     # If we have metadata, sort by it; otherwise infer from text
#     if metas and len(metas) == len(texts):
#         pairs = list(zip(metas, texts))
#         pairs.sort(key=lambda p: _extract_year_from_meta(p[0]), reverse=True)
#         metas_sorted, texts_sorted = zip(*pairs) if pairs else ([], [])
#         result["fused_metadata"] = list(metas_sorted)
#         result["fused_text_blocks"] = list(texts_sorted)
#     else:
#         # No metadata: try to infer year from text itself
#         def year_from_text(t: str) -> int:
#             m = re.findall(r"(19|20)\d{2}", str(t))
#             return int(m[0]) if m else 0
#         result["fused_text_blocks"] = sorted(texts, key=year_from_text, reverse=True)

#     print("🕒 Database-level sorting applied: newest → oldest")
#     return result


# class DatabaseManager:
#     """Manages multiple database configurations and allows hotswapping."""
    
#     def __init__(self):
#         self.configs: Dict[str, DatabaseConfig] = {}
#         self.active_config_name: Optional[str] = None
#         self._load_default_configs()
    
#     def _load_default_configs(self):
#         """Load default database configurations."""
        
#         # Full papers database
#         self.register_config(
#             name="full",
#             config=DatabaseConfig(
#                 mode="full",
#                 chroma_dir=os.getenv("CHROMA_DIR_FULL", r"D:\OSPO\KG-RAG1\chroma_store_full"),
#                 chroma_collection="papers_all",
#                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
#                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
#                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
#                 neo4j_database="syr-rag",
#                 description="Full papers with complete text and metadata"
#             )
#         )
        
#         # Abstracts database
#         self.register_config(
#             name="abstracts",
#             config=DatabaseConfig(
#                 mode="abstracts",
#                 chroma_dir=os.getenv("CHROMA_DIR_ABSTRACTS", r"D:\OSPO\KG-RAG1\chroma_store_abstracts"),
#                 chroma_collection="abstracts_all",
#                 neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
#                 neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
#                 neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
#                 neo4j_database="syr-rag-abstracts",
#                 description="Abstracts only from academic APIs"
#             )
#         )
        
#         self.active_config_name = "abstracts"
    
#     def register_config(self, name: str, config: DatabaseConfig):
#         """Register a new database configuration."""
#         self.configs[name] = config
#         print(f"✅ Registered database config: {name} (Neo4j DB: {config.neo4j_database})")
    
#     def switch_config(self, name: str) -> bool:
#         """Switch to a different database configuration."""
#         if name not in self.configs:
#             print(f"❌ Config '{name}' not found")
#             return False
        
#         self.active_config_name = name
#         config = self.configs[name]
#         print(f"🔄 Switched to '{name}' (Neo4j: {config.neo4j_database})")
#         return True
    
#     def get_active_config(self) -> Optional[DatabaseConfig]:
#         """Get the currently active database configuration."""
#         if self.active_config_name:
#             return self.configs.get(self.active_config_name)
#         return None
    
#     def list_configs(self) -> List[str]:
#         """List all available database configurations."""
#         return list(self.configs.keys())
    
#     def get_config(self, name: str) -> Optional[DatabaseConfig]:
#         """Get a specific database configuration by name."""
#         return self.configs.get(name)
    
#     def add_custom_config(
#         self,
#         name: str,
#         chroma_dir: str,
#         chroma_collection: str,
#         neo4j_uri: str = "bolt://localhost:7687",
#         neo4j_user: str = "neo4j",
#         neo4j_password: str = "password",
#         neo4j_database: str = "neo4j",
#         description: str = "Custom database"
#     ):
#         """Add a custom database configuration at runtime."""
#         config = DatabaseConfig(
#             mode="custom",
#             chroma_dir=chroma_dir,
#             chroma_collection=chroma_collection,
#             neo4j_uri=neo4j_uri,
#             neo4j_user=neo4j_user,
#             neo4j_password=neo4j_password,
#             neo4j_database=neo4j_database,
#             description=description
#         )
#         self.register_config(name, config)
    
#     def validate_config(self, name: str) -> Dict[str, bool]:
#         """Validate that all paths/connections exist for a config."""
#         config = self.get_config(name)
#         if not config:
#             return {"valid": False, "error": "Config not found"}
        
#         validation = {
#             "chroma_dir_exists": os.path.exists(config.chroma_dir),
#             "neo4j_connectable": self._test_neo4j_connection(config)
#         }
        
#         validation["valid"] = all(validation.values())
#         return validation
    
#     def _test_neo4j_connection(self, config: DatabaseConfig) -> bool:
#         """Test Neo4j connection."""
#         try:
#             from neo4j import GraphDatabase
#             driver = GraphDatabase.driver(
#                 config.neo4j_uri,
#                 auth=(config.neo4j_user, config.neo4j_password)
#             )
#             with driver.session(database=config.neo4j_database) as session:
#                 session.run("RETURN 1")
#             driver.close()
#             return True
#         except Exception as e:
#             print(f"❌ Neo4j connection failed for {config.neo4j_database}: {e}")
#             return False


# # Global singleton instance
# _db_manager = None


# def get_db_manager() -> DatabaseManager:
#     """Get the global database manager instance."""
#     global _db_manager
#     if _db_manager is None:
#         _db_manager = DatabaseManager()
#     return _db_manager


# def get_active_db_config() -> DatabaseConfig:
#     """Quick access to active database configuration."""
#     manager = get_db_manager()
#     config = manager.get_active_config()
#     if config is None:
#         raise RuntimeError("No active database configuration set")
#     return config


"""
database_manager.py - Hotswappable database configuration manager
(Updated: adds year-aware helper so retriever can sort once per request)
"""

import os
import re   # ✅ needed for sort_docs_by_year_desc
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class DatabaseMode(Enum):
    FULL = "full"
    ABSTRACTS = "abstracts"
    CUSTOM = "custom"


@dataclass
class DatabaseConfig:
    mode: str
    chroma_dir: str
    chroma_collection: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    description: str
    # prefer newer docs first
    prefer_newer: bool = True

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "chroma_dir": self.chroma_dir,
            "chroma_collection": self.chroma_collection,
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": self.neo4j_password,
            "neo4j_database": self.neo4j_database,
            "description": self.description,
            "prefer_newer": self.prefer_newer,
        }


class DatabaseManager:
    def __init__(self):
        self.configs: Dict[str, DatabaseConfig] = {}
        self.active_config_name: Optional[str] = None
        self._load_default_configs()

    def _load_default_configs(self):
        # Full papers
        self.register_config(
            name="full",
            config=DatabaseConfig(
                mode="full",
                chroma_dir=os.getenv(
                    "CHROMA_DIR_FULL", r"D:\OSPO\KG-RAG1\chroma_store_full"
                ),
                chroma_collection="papers_all",
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
                neo4j_database="syr-rag",
                description="Full papers with complete text and metadata",
                prefer_newer=True,
            ),
        )

        # Abstracts
        self.register_config(
            name="abstracts",
            config=DatabaseConfig(
                mode="abstracts",
                chroma_dir=os.getenv(
                    "CHROMA_DIR_ABSTRACTS", r"D:\OSPO\KG-RAG1\chroma_store_abstracts"
                ),
                chroma_collection="abstracts_all",
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                neo4j_password=os.getenv("NEO4J_PASS", "OSPOlol@1234"),
                neo4j_database="syr-rag-abstracts",
                description="Abstracts only from academic APIs",
                prefer_newer=True,
            ),
        )

        self.active_config_name = "abstracts"

    def register_config(self, name: str, config: DatabaseConfig):
        self.configs[name] = config
        print(f"✅ Registered database config: {name} (Neo4j DB: {config.neo4j_database})")

    def switch_config(self, name: str) -> bool:
        if name not in self.configs:
            print(f"❌ Config '{name}' not found")
            return False
        self.active_config_name = name
        cfg = self.configs[name]
        print(f"🔄 Switched to '{name}' (Neo4j: {cfg.neo4j_database})")
        return True

    def get_active_config(self) -> Optional[DatabaseConfig]:
        if self.active_config_name:
            return self.configs.get(self.active_config_name)
        return None

    def list_configs(self) -> List[str]:
        return list(self.configs.keys())

    def get_config(self, name: str) -> Optional[DatabaseConfig]:
        return self.configs.get(name)

    def add_custom_config(
        self,
        name: str,
        chroma_dir: str,
        chroma_collection: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        description: str = "Custom database",
        prefer_newer: bool = True,
    ):
        cfg = DatabaseConfig(
            mode="custom",
            chroma_dir=chroma_dir,
            chroma_collection=chroma_collection,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            description=description,
            prefer_newer=prefer_newer,
        )
        self.register_config(name, cfg)

    def validate_config(self, name: str) -> Dict[str, bool]:
        cfg = self.get_config(name)
        if not cfg:
            return {"valid": False, "error": "Config not found"}

        validation = {
            "chroma_dir_exists": os.path.exists(cfg.chroma_dir),
            "neo4j_connectable": self._test_neo4j_connection(cfg),
        }
        validation["valid"] = all(validation.values())
        return validation

    def _test_neo4j_connection(self, cfg: DatabaseConfig) -> bool:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
            )
            with driver.session(database=cfg.neo4j_database) as session:
                session.run("RETURN 1")
            driver.close()
            return True
        except Exception as e:
            print(f"❌ Neo4j connection failed for {cfg.neo4j_database}: {e}")
            return False


# global singleton
_db_manager = None


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_active_db_config() -> DatabaseConfig:
    mgr = get_db_manager()
    cfg = mgr.get_active_config()
    if cfg is None:
        raise RuntimeError("No active database configuration set")
    return cfg


def sort_docs_by_year_desc(blocks: list[str]) -> list[str]:
    """
    Sort arbitrary doc strings by the first 4-digit year we can find, newest first.
    Called from rag_pipeline BEFORE context packing so we get newest papers first.
    """

    def extract_year(s: str) -> int:
        m = re.search(r"(19|20)\d{2}", s or "")
        if m:
            return int(m.group(0))
        return 0

    return sorted(blocks, key=lambda b: extract_year(b), reverse=True)
