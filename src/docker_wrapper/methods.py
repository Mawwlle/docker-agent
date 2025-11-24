"""Wrappers to make docker client fully typed"""

from src.docker_wrapper.schemas import ContainerStats, DockerContainer, DockerSystemDF
import docker


def get_containers(
    client: docker.DockerClient, all: bool = False
) -> list[DockerContainer]:
    """
    Get a list of containers

    Args:
        client: docker client
        all: returns only running containers by default. If all set to True - then give all containers (stopeed also)

    Returns:
        List of docker containers
    """

    return [
        DockerContainer.model_validate(container)
        for container in client.containers.list(all)
    ]


def get_container_stats(
    client: docker.DockerClient, container: DockerContainer
) -> ContainerStats:
    """
    Get container stats

    Args:
        client: Docker client
        container: DockerContainer instance

    Returns:
        Container statistics
    """

    raw_stats = client.containers.get(container.id).stats(stream=False)

    return ContainerStats.from_api_response(
        container_id=container.id, name=container.name, stats=raw_stats
    )


def get_system_df(client: docker.DockerClient) -> DockerSystemDF:
    """
    Get container stats

    Args:
        client: Docker client
        container: DockerContainer instance

    Returns:
        Container statistics
    """
    return DockerSystemDF.from_api_response(client.df())
