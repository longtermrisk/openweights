"""Broker command - starts the credential broker (openweights/broker)."""


def add_broker_parser(parser):
    parser.add_argument("--port", type=int, default=8125, help="Port (default: 8125)")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)"
    )


def handle_broker(args):
    """Start the credential broker using uvicorn."""
    import logging

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # httpx logs every request URL at INFO; the broker's own line says the same.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    print(f"Starting the OpenWeights credential broker on {args.host}:{args.port}")
    # One process: the endpoints are async and the caches live in it.
    uvicorn.run(
        "openweights.broker.app:app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=False,
    )
    return 0
