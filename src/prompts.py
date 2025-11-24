from langchain_core.prompts import ChatPromptTemplate
from textwrap import dedent

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                """
        You are a Docker monitoring coordinator. Your task is to analyze the monitoring request and create a structured monitoring plan.

        Based on the user's task, generate:
        1. Specific container analysis tasks (which containers to monitor, what metrics to focus on)
        2. System-wide analysis tasks (disk usage, resource allocation, cleanup opportunities)
        3. Key focus areas for monitoring (CPU usage, memory consumption, network I/O, block I/O, disk space, container status)
        4. Which tools to use and in what order (list_containers, container_stats, system_df)

        Be specific about what metrics to collect and which containers are most relevant to the task.

        {format_instructions}
        /no_think
        """
            ).strip(),
        ),
        (
            "human",
            dedent(
                """
        Task: {task}
        Include stopped containers: {include_stopped}
        Max containers to analyze: {max_containers}
        """
            ).strip(),
        ),
    ]
)

CONTAINER_STATS_PARSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                """
        You are a helpful assistant that extracts detailed container statistics from agent search results.
        Extract comprehensive metrics for each container including:
        - CPU usage percentage
        - Memory usage (used/limit) with percentage
        - Network I/O (bytes received/transmitted)
        - Block I/O (bytes read/written)
        - PID count
        - Container status and health indicators

        Use the exact ContainerStats model structure with all available metrics.

        {format_instructions}
        /no_think
        """
            ).strip(),
        ),
        (
            "human",
            dedent(
                """
        Extract container statistics from these agent results:

        {agent_output}
        """
            ).strip(),
        ),
    ]
)


SYSTEM_DF_PARSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                """
        You are a helpful assistant that extracts system disk usage information from agent search results.
        Extract detailed information for each resource type including:
        - Images: total count, active count, total size, reclaimable space and percentage
        - Containers: total count, running count, stopped count, total size, reclaimable space
        - Volumes: total count, active count, unused count, total size, reclaimable space
        - Build Cache: total count, active count, unused count, total size, reclaimable space

        Use the exact DockerSystemDF model structure with all available metrics.

        {format_instructions}
        /no_think
        """
            ).strip(),
        ),
        (
            "human",
            dedent(
                """
        Extract system disk usage information from these agent results:

        {agent_output}
        """
            ).strip(),
        ),
    ]
)

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                """
        You are a Docker system analyst and technical writer.

        Your task is to synthesize all collected monitoring data into a comprehensive report.

        You should:
        1. Identify and summarize key findings across all containers and system resources
        2. Write a coherent summary of system health and performance
        3. Highlight containers with concerning resource usage patterns (high CPU/memory, excessive I/O)
        4. Identify optimization opportunities and potential bottlenecks
        5. Provide specific, actionable recommendations including cleanup commands
        6. Determine if additional analysis is needed to complete the report

        If the current data is insufficient for a complete analysis, set needs_additional_analysis to true and specify what additional data is required.
        Otherwise, set it to false and provide the final report.

        Create a well-structured, informative Docker monitoring report that helps system administrators make informed decisions. Focus on practical insights and actionable recommendations.

        {format_instructions}
        /no_think
        """
            ).strip(),
        ),
        (
            "human",
            dedent(
                """
        {context}
        """
            ).strip(),
        ),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                """
        You are a technical writer specializing in Docker system reports.

        Your task is to transform the structured Docker monitoring report into a beautiful, concise, and human-readable summary.

        Format the summary with:
        1. A clear title and overall health status
        2. Bullet points for key findings (max 5)
        3. Clear section for recommendations with actionable items
        4. Brief technical details only where essential
        5. Professional but accessible language

        The summary should be easy to read and understand for both technical and non-technical stakeholders.

        {format_instructions}
        /no_think
        """
            ).strip(),
        ),
        (
            "human",
            dedent(
                """
        Docker Report:
        Task: {task}
        Summary: {summary}
        Key Findings: {key_findings}
        Recommendations: {recommendations}
        """
            ).strip(),
        ),
    ]
)