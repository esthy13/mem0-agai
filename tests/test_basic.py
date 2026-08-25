
from project_config import ENV_PATH, PROJECT_ROOT


def test_project_config_uses_repository_root() -> None:
    """Shared configuration should resolve files from the flattened root."""
    assert PROJECT_ROOT == ENV_PATH.parent
    assert PROJECT_ROOT.name == "mem0-agai"

