"""
Tools for working with Docker.

This module provides LangChain tool wrappers for Docker operations,
handling Docker client initialization and providing type-safe interfaces
for container management and monitoring.
"""

from typing import Annotated
from langchain.tools import tool
from pydantic import Field

from src.docker_wrapper.schemas import ContainerStats, DockerContainer, DockerSystemDF
from src.docker_wrapper.methods import (
    get_container_stats,
    get_containers,
    get_system_df,
)
from src.utils import get_docker_client


@tool
def list_containers(
    all: Annotated[
        bool,
        Field(
            description="Returns only running containers by default. "
            "If set to True, returns all containers including stopped ones."
        ),
    ] = False,
) -> list[DockerContainer]:
    """
    List Docker containers with their basic information.

    This tool retrieves information about Docker containers including their
    ID, name, status, image, ports, and creation time. By default, only
    running containers are returned.

    Args:
        all: If False (default), returns only running containers.
             If True, returns all containers including stopped and exited ones.

    Returns:
        List of DockerContainer objects containing container metadata.

    Example:
        >>> # Get only running containers
        >>> running = list_containers(all=False)
        >>> # Get all containers including stopped
        >>> all_containers = list_containers(all=True)

    Raises:
        docker.errors.APIError: If Docker daemon is not accessible.
    """
    client = get_docker_client()
    return get_containers(client, all)


@tool
def container_stats(
    container: Annotated[
        DockerContainer,
        Field(description="Docker container data model from list_containers tool"),
    ],
) -> ContainerStats:
    """
    Get real-time resource usage statistics for a specific container.

    This tool retrieves detailed performance metrics including CPU usage,
    memory consumption, network I/O, and block I/O for a running container.

    Args:
        container: A DockerContainer object obtained from list_containers tool.
                   Must contain valid container ID and name.

    Returns:
        ContainerStats object with resource usage metrics including:
        - CPU percentage
        - Memory usage and limit
        - Memory percentage
        - Network I/O (bytes sent/received)
        - Block I/O (bytes read/written)

    Example:
        >>> containers = list_containers(all=False)
        >>> stats = container_stats(container=containers[0])
        >>> print(f"CPU: {stats.cpu_percent}%, Memory: {stats.memory_percent}%")

    Raises:
        docker.errors.NotFound: If container ID is invalid or container was removed.
        docker.errors.APIError: If Docker daemon is not accessible.

    Note:
        Container must be running to retrieve stats. Stopped containers will
        raise an error.
    """

    client = get_docker_client()

    return get_container_stats(client, container)


@tool
def system_df() -> DockerSystemDF:
    """
    Get Docker system-wide disk usage information.

    This tool retrieves comprehensive disk space usage across all Docker
    resources including images, containers, volumes, and build cache.
    Useful for diagnosing disk space issues and identifying cleanup opportunities.

    Returns:
        DockerSystemDF object containing:
        - Images: Total size, count, and reclaimable space
        - Containers: Total size, count, and reclaimable space
        - Volumes: Total size, count, and reclaimable space
        - Build Cache: Total size and reclaimable space

    Example:
        >>> df = system_df()
        >>> print(f"Total images size: {df.images_size}")
        >>> print(f"Reclaimable space: {df.reclaimable}")

    Raises:
        docker.errors.APIError: If Docker daemon is not accessible.

    Note:
        Reclaimable space indicates resources that can be freed with
        `docker system prune` command.
    """

    client = get_docker_client()

    return get_system_df(client)
