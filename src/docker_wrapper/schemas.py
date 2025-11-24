"""
Docker models for container management, statistics, and system information.

Fully typed Pydantic models for parsing Docker CLI and API outputs.
"""

import re
import docker
from typing import Any, ClassVar
from pydantic import BaseModel, Field, field_validator, computed_field


class PortBinding(BaseModel):
    """Represents a single port binding with host IP and port."""

    host_ip: str = Field(
        ..., alias="HostIp", description="Host IP address (e.g., '0.0.0.0')"
    )
    host_port: str = Field(
        ..., alias="HostPort", description="Host port number as string"
    )

    model_config = {"populate_by_name": True}


class ContainerPorts(BaseModel):
    """
    Model for Docker container ports mapping.

    Keys are container ports (e.g., '5672/tcp')
    Values are either None (not exposed) or list of PortBinding objects
    """

    ports: dict[str, list[PortBinding] | None] = Field(
        default_factory=dict, description="Mapping of container ports to host bindings"
    )

    @field_validator("ports", mode="before")
    @classmethod
    def parse_ports(
        cls, v: dict[str, list[dict[str, Any]] | None] | None
    ) -> dict[str, list[PortBinding] | None]:
        """Convert raw dict to properly typed port bindings."""
        if not isinstance(v, dict):
            return v

        result = {}
        for port_key, bindings in v.items():
            if bindings is None:
                result[port_key] = None
            else:
                result[port_key] = [
                    PortBinding(**binding) if isinstance(binding, dict) else binding
                    for binding in bindings
                ]
        return result

    @computed_field
    @property
    def exposed_ports(self) -> dict[str, list[PortBinding]]:
        """Get only ports that are exposed (not None)."""
        return {
            port: bindings
            for port, bindings in self.ports.items()
            if bindings is not None
        }

    def get_host_port(self, container_port: str) -> str | None:
        """
        Get the host port for a given container port.

        Args:
            container_port: Container port string (e.g., '5672/tcp')

        Returns:
            Host port string or None if not exposed
        """
        bindings = self.ports.get(container_port)
        if bindings and len(bindings) > 0:
            return bindings[0].host_port
        return None

    def is_exposed(self, container_port: str) -> bool:
        """Check if a container port is exposed to host."""
        return self.ports.get(container_port) is not None


class DockerContainer(BaseModel):
    """Complete Docker container model with ports."""

    id: str = Field(..., description="Container ID")
    name: str = Field(..., description="Container name")
    image: str = Field(..., description="Image name and tag")
    status: str = Field(..., description="Container status (running, exited, etc.)")
    ports: ContainerPorts = Field(..., description="Port mappings")

    @field_validator("ports", mode="before")
    @classmethod
    def parse_container_ports(cls, v):
        """Handle raw ports dict from Docker API."""
        if isinstance(v, dict) and not isinstance(v, ContainerPorts):
            return ContainerPorts(ports=v)
        return v

    @field_validator("image", mode="before")
    @classmethod
    def parse_image(cls, v: docker.models.images.Image) -> str:
        if not v.tags:
            return "no tags found need to fix parse image method"
        return v.tags[0]  # TODO: fix this

    model_config = {"from_attributes": True}


class SizeParser:
    """Utility class for parsing human-readable sizes to bytes."""

    UNITS: ClassVar[dict[str, int]] = {
        "B": 1,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }

    @classmethod
    def parse(cls, size_str: str) -> int:
        """
        Convert human readable size to bytes.

        Args:
            size_str: Size string like '144.6MiB', '5.787GiB', '1.64kB'

        Returns:
            Size in bytes

        Raises:
            ValueError: If size format is invalid or unit is unknown
        """
        size_str = size_str.strip()
        match = re.match(r"^([\d.]+)\s*([A-Za-z]+)$", size_str)

        if not match:
            raise ValueError(f"Invalid size format: {size_str}")

        value = float(match.group(1))
        unit = match.group(2)

        if unit not in cls.UNITS:
            raise ValueError(f"Unknown unit: {unit}")

        return int(value * cls.UNITS[unit])


class MemoryUsage(BaseModel):
    """Memory usage with used and limit values."""

    used_bytes: int = Field(..., description="Memory used in bytes")
    limit_bytes: int = Field(..., description="Memory limit in bytes")
    used_human: str = Field(
        ..., description="Human readable used memory (e.g., '144.6MiB')"
    )
    limit_human: str = Field(..., description="Human readable limit (e.g., '5.787GiB')")

    @computed_field
    @property
    def percentage(self) -> float:
        """Calculate memory usage percentage."""
        if self.limit_bytes == 0:
            return 0.0
        return round((self.used_bytes / self.limit_bytes) * 100, 2)

    @classmethod
    def from_string(cls, usage_str: str) -> "MemoryUsage":
        """
        Parse memory string like '144.6MiB / 5.787GiB'.

        Args:
            usage_str: Memory usage string from docker stats

        Returns:
            MemoryUsage instance

        Raises:
            ValueError: If format is invalid
        """
        parts = usage_str.split(" / ")
        if len(parts) != 2:
            raise ValueError(f"Invalid memory usage format: {usage_str}")

        used_human = parts[0].strip()
        limit_human = parts[1].strip()

        used_bytes = SizeParser.parse(used_human)
        limit_bytes = SizeParser.parse(limit_human)

        return cls(
            used_bytes=used_bytes,
            limit_bytes=limit_bytes,
            used_human=used_human,
            limit_human=limit_human,
        )


class NetworkIO(BaseModel):
    """Network I/O statistics."""

    rx_bytes: int = Field(..., description="Received bytes")
    tx_bytes: int = Field(..., description="Transmitted bytes")
    rx_human: str = Field(..., description="Human readable received (e.g., '1.64kB')")
    tx_human: str = Field(..., description="Human readable transmitted (e.g., '138B')")

    @computed_field
    @property
    def total_bytes(self) -> int:
        """Total network traffic (RX + TX)."""
        return self.rx_bytes + self.tx_bytes

    @classmethod
    def from_string(cls, io_str: str) -> "NetworkIO":
        """
        Parse network I/O string like '1.64kB / 138B'.

        Args:
            io_str: Network I/O string from docker stats

        Returns:
            NetworkIO instance

        Raises:
            ValueError: If format is invalid
        """
        parts = io_str.split(" / ")
        if len(parts) != 2:
            raise ValueError(f"Invalid network I/O format: {io_str}")

        rx_human = parts[0].strip()
        tx_human = parts[1].strip()

        rx_bytes = SizeParser.parse(rx_human)
        tx_bytes = SizeParser.parse(tx_human)

        return cls(
            rx_bytes=rx_bytes, tx_bytes=tx_bytes, rx_human=rx_human, tx_human=tx_human
        )


class BlockIO(BaseModel):
    """Block I/O statistics."""

    read_bytes: int = Field(..., description="Bytes read from disk")
    write_bytes: int = Field(..., description="Bytes written to disk")
    read_human: str = Field(..., description="Human readable read (e.g., '54.4MB')")
    write_human: str = Field(..., description="Human readable write (e.g., '152kB')")

    @computed_field
    @property
    def total_bytes(self) -> int:
        """Total disk I/O (read + write)."""
        return self.read_bytes + self.write_bytes

    @classmethod
    def from_string(cls, io_str: str) -> "BlockIO":
        """
        Parse block I/O string like '54.4MB / 152kB'.

        Args:
            io_str: Block I/O string from docker stats

        Returns:
            BlockIO instance

        Raises:
            ValueError: If format is invalid
        """
        parts = io_str.split(" / ")
        if len(parts) != 2:
            raise ValueError(f"Invalid block I/O format: {io_str}")

        read_human = parts[0].strip()
        write_human = parts[1].strip()

        read_bytes = SizeParser.parse(read_human)
        write_bytes = SizeParser.parse(write_human)

        return cls(
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            read_human=read_human,
            write_human=write_human,
        )


class ContainerStats(BaseModel):
    """Complete Docker container statistics."""

    container_id: str = Field(..., description="Container ID")
    name: str = Field(..., description="Container name")
    cpu_percent: float = Field(..., ge=0, description="CPU usage percentage")
    memory: MemoryUsage = Field(..., description="Memory usage statistics")
    memory_percent: float = Field(
        ..., ge=0, le=100, description="Memory usage percentage"
    )
    network_io: NetworkIO = Field(..., description="Network I/O statistics")
    block_io: BlockIO = Field(..., description="Block I/O statistics")
    pids: int = Field(..., ge=0, description="Number of PIDs")

    @field_validator("cpu_percent", mode="before")
    @classmethod
    def parse_cpu_percent(cls, v) -> float:
        """Parse CPU percentage from string like '0.75%'."""
        if isinstance(v, str):
            return float(v.rstrip("%"))
        return float(v)

    @field_validator("memory_percent", mode="before")
    @classmethod
    def parse_mem_percent(cls, v) -> float:
        """Parse memory percentage from string like '2.44%'."""
        if isinstance(v, str):
            return float(v.rstrip("%"))
        return float(v)

    @field_validator("memory", mode="before")
    @classmethod
    def parse_memory(cls, v):
        """Parse memory usage string."""
        if isinstance(v, str):
            return MemoryUsage.from_string(v)
        return v

    @field_validator("network_io", mode="before")
    @classmethod
    def parse_network(cls, v):
        """Parse network I/O string."""
        if isinstance(v, str):
            return NetworkIO.from_string(v)
        return v

    @field_validator("block_io", mode="before")
    @classmethod
    def parse_block(cls, v):
        """Parse block I/O string."""
        if isinstance(v, str):
            return BlockIO.from_string(v)
        return v

    def is_healthy(
        self, cpu_threshold: float = 80.0, memory_threshold: float = 80.0
    ) -> bool:
        """
        Check if container resource usage is within healthy thresholds.

        Args:
            cpu_threshold: Maximum healthy CPU percentage
            memory_threshold: Maximum healthy memory percentage

        Returns:
            True if both CPU and memory are below thresholds
        """
        return (
            self.cpu_percent < cpu_threshold and self.memory_percent < memory_threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with human-readable values."""
        return {
            "container_id": self.container_id,
            "name": self.name,
            "cpu_percent": self.cpu_percent,
            "memory_used": self.memory.used_human,
            "memory_limit": self.memory.limit_human,
            "memory_percent": self.memory_percent,
            "network_rx": self.network_io.rx_human,
            "network_tx": self.network_io.tx_human,
            "block_read": self.block_io.read_human,
            "block_write": self.block_io.write_human,
            "pids": self.pids,
        }

    @classmethod
    def from_api_response(
        cls, container_id: str, name: str, stats: dict[str, Any]
    ) -> "ContainerStats":
        """
        Parse Docker API stats() response.

        Args:
            container_id: Container ID
            name: Container name
            stats: Dictionary returned from container.stats(stream=False)

        Returns:
            ContainerStats instance
        """

        def bytes_to_human(bytes_size: int) -> str:
            """Convert bytes to human-readable format."""
            for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
                if bytes_size < 1024.0:
                    return f"{bytes_size:.1f}{unit}"
                bytes_size /= 1024.0
            return f"{bytes_size:.1f}PiB"

        # Calculate CPU percentage
        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})

        cpu_delta = cpu_stats.get("cpu_usage", {}).get(
            "total_usage", 0
        ) - precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
        system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get(
            "system_cpu_usage", 0
        )

        online_cpus = cpu_stats.get("online_cpus", 1)
        cpu_percent = 0.0
        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * online_cpus * 100.0

        # Calculate memory usage
        memory_stats = stats.get("memory_stats", {})
        memory_usage = memory_stats.get("usage", 0)
        memory_limit = memory_stats.get("limit", 0)

        # Subtract cache on Linux (more accurate)
        cache = memory_stats.get("stats", {}).get("cache", 0)
        memory_usage = max(0, memory_usage - cache)

        memory_percent = (
            (memory_usage / memory_limit * 100.0) if memory_limit > 0 else 0.0
        )

        # Calculate network I/O
        networks = stats.get("networks", {})
        rx_bytes = sum(net.get("rx_bytes", 0) for net in networks.values())
        tx_bytes = sum(net.get("tx_bytes", 0) for net in networks.values())

        # Calculate block I/O
        blkio_stats = stats.get("blkio_stats", {})
        io_service_bytes = blkio_stats.get("io_service_bytes_recursive", [])

        read_bytes = sum(
            entry.get("value", 0)
            for entry in io_service_bytes
            if entry.get("op") == "read"
        )
        write_bytes = sum(
            entry.get("value", 0)
            for entry in io_service_bytes
            if entry.get("op") == "write"
        )

        # Get PIDs
        pids = stats.get("pids_stats", {}).get("current", 0)

        # Create memory usage object
        memory_obj = MemoryUsage(
            used_bytes=memory_usage,
            limit_bytes=memory_limit,
            used_human=bytes_to_human(memory_usage),
            limit_human=bytes_to_human(memory_limit),
        )

        # Create network I/O object
        network_obj = NetworkIO(
            rx_bytes=rx_bytes,
            tx_bytes=tx_bytes,
            rx_human=bytes_to_human(rx_bytes),
            tx_human=bytes_to_human(tx_bytes),
        )

        # Create block I/O object
        block_obj = BlockIO(
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            read_human=bytes_to_human(read_bytes),
            write_human=bytes_to_human(write_bytes),
        )

        return cls(
            container_id=container_id,
            name=name,
            cpu_percent=round(cpu_percent, 2),
            memory=memory_obj,
            memory_percent=round(memory_percent, 2),
            network_io=network_obj,
            block_io=block_obj,
            pids=pids,
        )


class SizeInfo(BaseModel):
    """Size information with bytes and human-readable format."""

    bytes: int = Field(..., description="Size in bytes")
    human: str = Field(..., description="Human readable size (e.g., '11.77GB')")

    @classmethod
    def from_string(cls, size_str: str) -> "SizeInfo":
        """
        Parse size string like '11.77GB', '200.7kB', '1.868MB'.

        Args:
            size_str: Size string from docker system df

        Returns:
            SizeInfo instance
        """
        size_bytes = SizeParser.parse(size_str)
        return cls(bytes=size_bytes, human=size_str)


class ReclaimableInfo(BaseModel):
    """Information about reclaimable space."""

    size: SizeInfo = Field(..., description="Reclaimable size")
    percentage: float = Field(
        ..., ge=0, le=100, description="Percentage of total that is reclaimable"
    )

    @classmethod
    def from_string(cls, reclaim_str: str) -> "ReclaimableInfo":
        """
        Parse reclaimable string like '9.76GB (82%)' or '0B (0%)'.

        Args:
            reclaim_str: Reclaimable string from docker system df

        Returns:
            ReclaimableInfo instance

        Raises:
            ValueError: If format is invalid
        """
        match = re.match(r"^(.+?)\s+\((\d+(?:\.\d+)?)%\)$", reclaim_str.strip())
        if not match:
            raise ValueError(f"Invalid reclaimable format: {reclaim_str}")

        size_str = match.group(1)
        percentage = float(match.group(2))

        return cls(size=SizeInfo.from_string(size_str), percentage=percentage)


class ResourceInfo(BaseModel):
    """Base class for Docker resource usage information."""

    type: str = Field(..., description="Resource type")
    total: int = Field(..., ge=0, description="Total number of resources")
    active: int = Field(..., ge=0, description="Number of active resources")
    size: SizeInfo = Field(..., description="Total size")
    reclaimable: ReclaimableInfo = Field(..., description="Reclaimable space")

    @computed_field
    @property
    def inactive(self) -> int:
        """Number of inactive resources."""
        return max(0, self.total - self.active)

    def needs_cleanup(self, threshold: float = 50.0) -> bool:
        """
        Check if resource needs cleanup based on reclaimable percentage.

        Args:
            threshold: Minimum reclaimable percentage to trigger cleanup

        Returns:
            True if cleanup is recommended
        """
        return self.reclaimable.percentage >= threshold


class ImagesInfo(ResourceInfo):
    """Docker images disk usage information."""

    type: str = Field(default="Images", description="Resource type")


class ContainersInfo(ResourceInfo):
    """Docker containers disk usage information."""

    type: str = Field(default="Containers", description="Resource type")

    @computed_field
    @property
    def stopped(self) -> int:
        """Number of stopped containers (alias for inactive)."""
        return self.inactive


class VolumesInfo(ResourceInfo):
    """Docker volumes disk usage information."""

    type: str = Field(default="Local Volumes", description="Resource type")

    @computed_field
    @property
    def unused(self) -> int:
        """Number of unused volumes (alias for inactive)."""
        return self.inactive


class BuildCacheInfo(ResourceInfo):
    """Docker build cache disk usage information."""

    type: str = Field(default="Build Cache", description="Resource type")

    @computed_field
    @property
    def unused(self) -> int:
        """Number of unused cache entries (alias for inactive)."""
        return self.inactive


class CleanupRecommendation(BaseModel):
    """Cleanup recommendation for a Docker resource."""

    resource_type: str = Field(
        ..., description="Type of resource (Images, Containers, etc.)"
    )
    reclaimable_size: str = Field(..., description="Human-readable reclaimable size")
    reclaimable_percentage: float = Field(
        ..., description="Percentage that can be reclaimed"
    )
    count: int = Field(..., description="Number of items that can be removed")
    command: str = Field(..., description="Docker command to run for cleanup")
    priority: str = Field(..., description="Priority level: high, medium, low")

    @classmethod
    def from_resource(
        cls, resource: ResourceInfo, command: str
    ) -> "CleanupRecommendation":
        """Create recommendation from resource info."""
        # Determine priority based on reclaimable percentage
        if resource.reclaimable.percentage >= 75:
            priority = "high"
        elif resource.reclaimable.percentage >= 50:
            priority = "medium"
        else:
            priority = "low"

        return cls(
            resource_type=resource.type,
            reclaimable_size=resource.reclaimable.size.human,
            reclaimable_percentage=resource.reclaimable.percentage,
            count=resource.inactive,
            command=command,
            priority=priority,
        )


class DockerSystemDF(BaseModel):
    """Complete Docker system disk usage information."""

    images: ImagesInfo = Field(..., description="Images disk usage")
    containers: ContainersInfo = Field(..., description="Containers disk usage")
    volumes: VolumesInfo = Field(..., description="Volumes disk usage")
    build_cache: BuildCacheInfo = Field(..., description="Build cache disk usage")

    @computed_field
    @property
    def total_size_bytes(self) -> int:
        """Total disk usage in bytes across all types."""
        return (
            self.images.size.bytes
            + self.containers.size.bytes
            + self.volumes.size.bytes
            + self.build_cache.size.bytes
        )

    @computed_field
    @property
    def total_reclaimable_bytes(self) -> int:
        """Total reclaimable space in bytes."""
        return (
            self.images.reclaimable.size.bytes
            + self.containers.reclaimable.size.bytes
            + self.volumes.reclaimable.size.bytes
            + self.build_cache.reclaimable.size.bytes
        )

    @computed_field
    @property
    def total_reclaimable_percentage(self) -> float:
        """Overall reclaimable percentage."""
        if self.total_size_bytes == 0:
            return 0.0
        return round((self.total_reclaimable_bytes / self.total_size_bytes) * 100, 2)

    @classmethod
    def from_api_response(cls, df_data: dict) -> "DockerSystemDF":
        """
        Parse Docker API df() response.

        Args:
            df_data: Dictionary returned from client.df()

        Returns:
            DockerSystemDF instance
        """

        def bytes_to_human(bytes_size: int) -> str:
            """Convert bytes to human-readable format."""
            for unit in ["B", "kB", "MB", "GB", "TB"]:
                if bytes_size < 1000.0:
                    return f"{bytes_size:.2f}{unit}".rstrip("0").rstrip(".")
                bytes_size /= 1000.0
            return f"{bytes_size:.2f}PB"

        images = df_data.get("Images", [])
        total_images = len(images)
        active_images = sum(1 for img in images if img.get("Containers", 0) > 0)
        images_size = sum(img.get("Size", 0) for img in images)
        images_reclaimable = sum(
            img.get("Size", 0) for img in images if img.get("Containers", 0) == 0
        )
        images_reclaim_pct = (
            (images_reclaimable / images_size * 100) if images_size > 0 else 0
        )

        containers = df_data.get("Containers", [])
        total_containers = len(containers)
        active_containers = sum(1 for c in containers if c.get("State") == "running")
        containers_size = sum(c.get("SizeRw", 0) for c in containers)
        containers_reclaimable = sum(
            c.get("SizeRw", 0) for c in containers if c.get("State") != "running"
        )
        containers_reclaim_pct = (
            (containers_reclaimable / containers_size * 100)
            if containers_size > 0
            else 0
        )

        volumes = df_data.get("Volumes", [])
        total_volumes = len(volumes)
        active_volumes = sum(
            1 for v in volumes if v.get("UsageData", {}).get("RefCount", 0) > 0
        )
        volumes_size = sum(v.get("UsageData", {}).get("Size", 0) for v in volumes)
        volumes_reclaimable = sum(
            v.get("UsageData", {}).get("Size", 0)
            for v in volumes
            if v.get("UsageData", {}).get("RefCount", 0) == 0
        )
        volumes_reclaim_pct = (
            (volumes_reclaimable / volumes_size * 100) if volumes_size > 0 else 0
        )

        build_cache = df_data.get("BuildCache", [])
        total_cache = len(build_cache)
        active_cache = sum(1 for c in build_cache if c.get("InUse", False))
        cache_size = sum(c.get("Size", 0) for c in build_cache)
        cache_reclaimable = sum(
            c.get("Size", 0) for c in build_cache if not c.get("InUse", False)
        )
        cache_reclaim_pct = (
            (cache_reclaimable / cache_size * 100) if cache_size > 0 else 0
        )

        return cls(
            images=ImagesInfo(
                total=total_images,
                active=active_images,
                size=SizeInfo(bytes=images_size, human=bytes_to_human(images_size)),
                reclaimable=ReclaimableInfo(
                    size=SizeInfo(
                        bytes=images_reclaimable,
                        human=bytes_to_human(images_reclaimable),
                    ),
                    percentage=round(images_reclaim_pct, 2),
                ),
            ),
            containers=ContainersInfo(
                total=total_containers,
                active=active_containers,
                size=SizeInfo(
                    bytes=containers_size, human=bytes_to_human(containers_size)
                ),
                reclaimable=ReclaimableInfo(
                    size=SizeInfo(
                        bytes=containers_reclaimable,
                        human=bytes_to_human(containers_reclaimable),
                    ),
                    percentage=round(containers_reclaim_pct, 2),
                ),
            ),
            volumes=VolumesInfo(
                total=total_volumes,
                active=active_volumes,
                size=SizeInfo(bytes=volumes_size, human=bytes_to_human(volumes_size)),
                reclaimable=ReclaimableInfo(
                    size=SizeInfo(
                        bytes=volumes_reclaimable,
                        human=bytes_to_human(volumes_reclaimable),
                    ),
                    percentage=round(volumes_reclaim_pct, 2),
                ),
            ),
            build_cache=BuildCacheInfo(
                total=total_cache,
                active=active_cache,
                size=SizeInfo(bytes=cache_size, human=bytes_to_human(cache_size)),
                reclaimable=ReclaimableInfo(
                    size=SizeInfo(
                        bytes=cache_reclaimable, human=bytes_to_human(cache_reclaimable)
                    ),
                    percentage=round(cache_reclaim_pct, 2),
                ),
            ),
        )

    def get_cleanup_recommendations(
        self, threshold: float = 50.0, include_low_priority: bool = False
    ) -> list[CleanupRecommendation]:
        """
        Get cleanup recommendations based on reclaimable space.

        Args:
            threshold: Minimum reclaimable percentage to include
            include_low_priority: Whether to include low priority recommendations

        Returns:
            List of cleanup recommendations sorted by priority
        """

        recommendations: list[CleanupRecommendation] = []

        if self.images.needs_cleanup(threshold):
            rec = CleanupRecommendation.from_resource(
                self.images, "docker image prune -a"
            )
            recommendations.append(rec)

        if self.containers.reclaimable.percentage > 0:
            rec = CleanupRecommendation.from_resource(
                self.containers, "docker container prune"
            )
            recommendations.append(rec)

        if self.volumes.needs_cleanup(threshold):
            rec = CleanupRecommendation.from_resource(
                self.volumes, "docker volume prune"
            )
            recommendations.append(rec)

        if self.build_cache.needs_cleanup(threshold):
            rec = CleanupRecommendation.from_resource(
                self.build_cache, "docker builder prune"
            )
            recommendations.append(rec)

        if not include_low_priority:
            recommendations = [r for r in recommendations if r.priority != "low"]

        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recommendations
