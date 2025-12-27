
from shared.logger import get_logger

logger = get_logger()

# get deployment json
def create_compose_file():
    try:
        with open("deployment.json", "r") as f:
            deployment_json = json.load(f)
            
            
        return deployment_json
    except Exception as e:
        logger.error(f"An error occurred while reading deployment file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred while reading deployment file: {str(e)}"
        )