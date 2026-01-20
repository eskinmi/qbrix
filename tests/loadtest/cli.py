from __future__ import annotations

import sys

import click


@click.command()
@click.option(
    "--scenario",
    "-s",
    type=click.Choice(["single", "multi"]),
    default="single",
    help="load test scenario to run",
)
@click.option(
    "--web",
    is_flag=True,
    default=False,
    help="start with web interface (default: headless)",
)
@click.option(
    "--host",
    "-h",
    default="localhost",
    help="proxy service host",
)
@click.option(
    "--port",
    "-p",
    default=50050,
    type=int,
    help="proxy service port",
)
@click.option(
    "--users",
    "-u",
    default=10,
    type=int,
    help="number of concurrent users",
)
@click.option(
    "--spawn-rate",
    "-r",
    default=1,
    type=float,
    help="users to spawn per second",
)
@click.option(
    "--run-time",
    "-t",
    default=None,
    type=str,
    help="run time (e.g., 60s, 5m, 1h). required for headless mode",
)
@click.option(
    "--web-host",
    default="localhost",
    help="web interface host",
)
@click.option(
    "--web-port",
    default=8089,
    type=int,
    help="web interface port",
)
@click.option(
    "--num-experiments",
    default=10,
    type=int,
    help="number of experiments for multi scenario",
)
@click.option(
    "--max-users-per-experiment",
    default=100,
    type=int,
    help="max users per experiment for multi scenario",
)
def run(
    scenario: str,
    web: bool,
    host: str,
    port: int,
    users: int,
    spawn_rate: float,
    run_time: str | None,
    web_host: str,
    web_port: int,
    num_experiments: int,
    max_users_per_experiment: int,
) -> None:
    """
    run qbrix load tests using locust.

    examples:

        # run single experiment scenario with web ui
        uv run loadtest --web

        # run headless for 60 seconds with 100 users
        uv run loadtest -u 100 -r 10 -t 60s

        # run multi-experiment scenario
        uv run loadtest -s multi --num-experiments 5 -u 50 -t 2m

        # connect to remote proxy
        uv run loadtest -h proxy.example.com -p 50050 --web
    """
    import os

    # set environment variables for config
    os.environ["LOADTEST_PROXY_HOST"] = host
    os.environ["LOADTEST_PROXY_PORT"] = str(port)
    os.environ["LOADTEST_NUM_EXPERIMENTS"] = str(num_experiments)
    os.environ["LOADTEST_MAX_USERS_PER_EXPERIMENT"] = str(max_users_per_experiment)

    # build locust command args
    locust_args = ["locust"]

    # select scenario file
    if scenario == "single":
        locust_args.extend(["-f", "tests/loadtest/scenarios/single_experiment.py"])
    else:
        locust_args.extend(["-f", "tests/loadtest/scenarios/multi_experiment.py"])

    if web:
        locust_args.extend(["--web-host", web_host, "--web-port", str(web_port)])
        click.echo(f"starting locust web interface at http://{web_host}:{web_port}")
        click.echo(f"scenario: {scenario}")
        click.echo(f"target: {host}:{port}")
    else:
        if not run_time:
            click.echo("error: --run-time is required for headless mode", err=True)
            click.echo(
                "use --web for interactive mode or specify -t/--run-time", err=True
            )
            sys.exit(1)

        locust_args.extend(
            [
                "--headless",
                "-u",
                str(users),
                "-r",
                str(spawn_rate),
                "-t",
                run_time,
            ]
        )
        click.echo(f"running headless load test")
        click.echo(f"scenario: {scenario}")
        click.echo(f"target: {host}:{port}")
        click.echo(f"users: {users}, spawn rate: {spawn_rate}/s, duration: {run_time}")

    # replace current process with locust
    sys.argv = locust_args
    from locust.main import main as locust_main

    locust_main()


if __name__ == "__main__":
    run()
