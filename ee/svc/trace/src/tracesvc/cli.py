import asyncio

import click

from qbrixlog import configure_logging

from tracesvc.config import TraceSettings
from tracesvc.server import serve


@click.command()
@click.option("--host", default="0.0.0.0", help="gRPC server host")
@click.option("--port", default=50053, type=int, help="gRPC server port")
@click.option(
    "--consumer-name", default="worker-0", help="Consumer name for Redis Streams"
)
def run(host: str, port: int, consumer_name: str) -> None:
    configure_logging("trace")
    settings = TraceSettings(
        grpc_host=host, grpc_port=port, consumer_name=consumer_name
    )
    asyncio.run(serve(settings))


if __name__ == "__main__":
    run()
