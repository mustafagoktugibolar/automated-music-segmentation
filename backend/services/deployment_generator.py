import json
import yaml
import os
from shared.logger import get_logger

logger = get_logger()

class DeploymentGenerator:
    def __init__(self, config_path: str = "deployment.json", output_path: str = "docker-compose.yml"):
        self.config_path = config_path
        self.output_path = output_path

    def generate(self, config_data: dict = None):
        """
        Generates docker-compose.yml from the provided config_data or by reading the config file.
        """
        if config_data is None:
            if not os.path.exists(self.config_path):
                logger.error(f"Deployment config file not found: {self.config_path}")
                raise FileNotFoundError(f"{self.config_path} not found")
            
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

        logger.info("Generating docker-compose configuration...")
        
        services = {}
        volumes = config_data.get("volumes", {})

        for service_config in config_data.get("services", []):
            name = service_config.pop("name")
            # Remove 'type' metadata (not docker-compose standard)
            service_config.pop("type", None) 
            
            # Enforce: service name == container name
            service_config["container_name"] = name
            
            services[name] = service_config

        compose_data = {
            "services": services,
            "volumes": volumes
        }

        # Custom representer to avoid aliases in YAML
        yaml.Dumper.ignore_aliases = lambda *args : True

        try:
            with open(self.output_path, 'w') as f:
                f.write("# This file is auto-generated from deployment.json. DO NOT EDIT MANUALLY.\n")
                yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Successfully generated {self.output_path}")
            
            # Update source JSON file if new data provided
            if config_data:
                with open(self.config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)

        except Exception:
            logger.error("Failed to write docker-compose.yml", exc_info=True)
            raise

if __name__ == "__main__":
    DeploymentGenerator().generate()
